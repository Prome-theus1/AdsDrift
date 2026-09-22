"""Two-GPU exact-candidate parallel training for the Drifting generator."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from AdsDrift.model.generator import build_generator
from AdsDrift.model.utils.data_loader import ConditionBank, move_batch
from AdsDrift.model.drifting.distributed_objective import DistributedCandidateDriftingObjective
from AdsDrift.model.drifting.loss import build_drifting_loss
from AdsDrift.model.drifting.mace_features import build_mace_features, default_mace_config
from AdsDrift.model.utils.train import (
    _configure_numerics,
    _metric_values,
    _resolve_path,
    _seed_everything,
    _update_latest,
    load_config,
)


def _split_r0_for_rank(batch: dict, rank: int, world_size: int) -> dict:
    positions = batch["r0_positions"]
    generated_count = positions.shape[1]
    if generated_count % world_size:
        raise ValueError(
            f"R0 count {generated_count} must be divisible by world size {world_size}"
        )
    per_rank = generated_count // world_size
    start = rank * per_rank
    stop = start + per_rank
    result = dict(batch)
    result["r0_positions"] = positions[:, start:stop].contiguous()
    return result


def _advance_rng_through_completed_epochs(
    rng: np.random.Generator,
    bank: ConditionBank,
    completed_epochs: int,
    positives_per_mode: int,
    max_modes: int | None,
) -> None:
    """Reconstruct the v1 sampler state for legacy epoch-boundary checkpoints."""
    for _ in range(completed_epochs):
        order = rng.permutation(len(bank))
        for system_index in order:
            ConditionBank._sample_positive_indices(
                bank.systems[int(system_index)],
                positives_per_mode,
                max_modes,
                rng,
            )


def _save_checkpoint(
    path: Path,
    generator: DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: dict,
    epoch: int,
    step: int,
    rng: np.random.Generator,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "format_version": 3,
            "distributed_candidate_parallel": True,
            "coordinate_gradient_routing": "joint_unrestricted",
            "world_size": dist.get_world_size(),
            "generator": generator.module.state_dict(),
            "generator_config": config["generator"],
            "mace_config": default_mace_config(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "numpy_rng_state": rng.bit_generator.state,
            "epoch": epoch,
            "step": step,
            "config": {
                key: value for key, value in config.items() if not key.startswith("_")
            },
        },
        temporary,
    )
    temporary.replace(path)


def train_distributed(config: dict) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("Distributed candidate training requires CUDA")
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    training = config["training"]
    allow_tf32 = _configure_numerics(training)
    seed = int(training.get("seed", 0))
    _seed_everything(seed)
    rng = np.random.default_rng(seed)

    roots = [_resolve_path(value) for value in config["data"]["roots"]]
    movable_value = config["data"].get("movable_feature_root")
    movable_root = _resolve_path(movable_value) if movable_value else None
    bank = ConditionBank(roots, movable_root)
    expected_r0 = int(config["data"].get("r0_per_system", 100))
    if expected_r0 % world_size:
        raise ValueError("r0_per_system must be divisible by distributed world size")
    invalid = {
        system.system_id: system.random_count
        for system in bank.systems
        if system.random_count != expected_r0
    }
    if invalid:
        raise ValueError(f"Every system must contain exactly {expected_r0} R0: {invalid}")

    positives_per_mode = int(config["data"].get("positives_per_mode", 1))
    max_modes = config["data"].get("max_positive_modes")
    max_modes = None if max_modes is None else int(max_modes)
    resume = training.get("resume")
    checkpoint = None
    generator = build_generator(config["generator"]).to(device)
    if resume:
        checkpoint = torch.load(
            _resolve_path(resume), map_location=device, weights_only=False
        )
        generator.load_state_dict(checkpoint["generator"])
    generator = DistributedDataParallel(
        generator,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
    )

    drifting_config = dict(config["drifting"])
    feature_encoder = build_mace_features().to(device)
    objective = DistributedCandidateDriftingObjective(
        generator,
        feature_encoder,
        build_drifting_loss(drifting_config).to(device),
    ).to(device)
    objective.train()

    optimizer = torch.optim.AdamW(
        generator.module.parameters(),
        lr=float(training["learning_rate"]),
        betas=tuple(float(value) for value in training.get("betas", [0.9, 0.95])),
        weight_decay=float(training.get("weight_decay", 0.01)),
    )
    epochs = int(training["epochs"])
    if epochs < 1:
        raise ValueError("epochs must be positive")
    total_steps = epochs * len(bank)
    warmup_epochs = int(training.get("warmup_epochs", 10))
    if warmup_epochs < 0:
        raise ValueError("warmup_epochs must be nonnegative")
    # Ranks split candidates for the same condition, not the condition list.
    warmup_steps = warmup_epochs * len(bank)

    def schedule(step_index: int) -> float:
        if warmup_steps and step_index < warmup_steps:
            return max(step_index, 1) / warmup_steps
        progress = (step_index - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    start_epoch = 0
    step = 0
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        step = int(checkpoint["step"])
        if "numpy_rng_state" in checkpoint:
            rng.bit_generator.state = checkpoint["numpy_rng_state"]
        else:
            _advance_rng_through_completed_epochs(
                rng, bank, start_epoch, positives_per_mode, max_modes
            )
        del checkpoint

    run_directory = _resolve_path(training["run_directory"])
    log_path = run_directory / "metrics.jsonl"
    if rank == 0:
        run_directory.mkdir(parents=True, exist_ok=True)
        if resume and not log_path.exists():
            source_log = _resolve_path(resume).parent / "metrics.jsonl"
            if source_log.is_file() and source_log.resolve() != log_path.resolve():
                shutil.copyfile(source_log, log_path)
        (run_directory / "resolved_config.json").write_text(
            json.dumps(
                {key: value for key, value in config.items() if not key.startswith("_")},
                indent=2,
                ensure_ascii=False,
            )
            + "\n"
        )
        print(json.dumps({
            "event": "distributed_start",
            "world_size": world_size,
            "candidate_parallel": True,
            "systems": len(bank),
            "global_r0_per_batch": expected_r0,
            "local_r0_per_rank": expected_r0 // world_size,
            "start_epoch": start_epoch,
            "target_epochs": epochs,
            "step": step,
            "allow_tf32": allow_tf32,
            "mace_microbatch_size": feature_encoder.config.microbatch_size,
            "mace_activation_checkpointing": feature_encoder.config.activation_checkpointing,
            "coordinate_gradient_routing": "joint_unrestricted",
        }), flush=True)
    dist.barrier()

    log_every = int(training.get("log_every_steps", 1))
    checkpoint_every = int(training.get("checkpoint_every_epochs", 1))
    gradient_clip = float(training.get("gradient_clip_norm", 2.0))
    max_steps = training.get("max_steps")
    max_steps = None if max_steps is None else int(max_steps)
    max_runtime_minutes = training.get("max_runtime_minutes")
    max_runtime_seconds = (
        None if max_runtime_minutes is None else float(max_runtime_minutes) * 60.0
    )
    steps_this_run = 0
    training_started = time.monotonic()

    try:
        for epoch in range(start_epoch, epochs):
            epoch_started = time.monotonic()
            order = rng.permutation(len(bank))
            for batch_in_epoch, system_index in enumerate(order):
                system = bank.systems[int(system_index)]
                batch = bank.make_batch(
                    [system],
                    generated_per_condition=expected_r0,
                    positives_per_mode=positives_per_mode,
                    max_positive_modes=max_modes,
                    rng=rng,
                )
                batch = _split_r0_for_rank(batch, rank, world_size)
                batch = move_batch(batch, device)
                optimizer.zero_grad(set_to_none=True)
                started = time.perf_counter()
                learning_rate_used = optimizer.param_groups[0]["lr"]
                loss, metrics, _ = objective(batch)
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    generator.module.parameters(),
                    gradient_clip,
                    error_if_nonfinite=True,
                )
                optimizer.step()
                scheduler.step()
                step += 1
                steps_this_run += 1
                if rank == 0 and step % log_every == 0:
                    record = {
                        "epoch": epoch,
                        "batch_in_epoch": batch_in_epoch,
                        "step": step,
                        "system_id": system.system_id,
                        "positive_modes": system.mode_count,
                        "positive_samples_used": int(batch["positive_modes"].numel()),
                        "learning_rate": learning_rate_used,
                        "gradient_norm": float(gradient_norm.detach().cpu()),
                        "seconds": time.perf_counter() - started,
                        "world_size": world_size,
                        "local_r0_per_rank": expected_r0 // world_size,
                        **_metric_values(metrics),
                    }
                    with log_path.open("a") as handle:
                        handle.write(json.dumps(record) + "\n")
                    print(json.dumps(record), flush=True)
                if max_steps is not None and steps_this_run >= max_steps:
                    if rank == 0:
                        print(json.dumps({
                            "event": "max_steps_reached",
                            "steps_this_run": steps_this_run,
                            "global_step": step,
                        }), flush=True)
                    return run_directory

            if rank == 0 and (
                (epoch + 1) % checkpoint_every == 0 or epoch + 1 == epochs
            ):
                epoch_checkpoint = run_directory / f"epoch_{epoch + 1:04d}.pt"
                _save_checkpoint(
                    epoch_checkpoint,
                    generator,
                    optimizer,
                    scheduler,
                    config,
                    epoch,
                    step,
                    rng,
                )
                _update_latest(epoch_checkpoint)
                print(json.dumps({
                    "event": "epoch_complete",
                    "epoch": epoch,
                    "step": step,
                    "epoch_seconds": time.monotonic() - epoch_started,
                    "checkpoint": str(epoch_checkpoint),
                }), flush=True)
            dist.barrier()
            stop_for_runtime = (
                max_runtime_seconds is not None
                and time.monotonic() - training_started >= max_runtime_seconds
            )
            stop_tensor = torch.tensor(
                int(stop_for_runtime), device=device, dtype=torch.int32
            )
            dist.all_reduce(stop_tensor, op=dist.ReduceOp.MAX)
            if bool(stop_tensor.item()):
                if rank == 0:
                    print(json.dumps({
                        "event": "runtime_budget_reached_at_epoch_boundary",
                        "completed_epoch": epoch,
                        "step": step,
                    }), flush=True)
                return run_directory
        return run_directory
    finally:
        dist.barrier()
        dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-directory")
    parser.add_argument("--data-root")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-runtime-minutes", type=float)
    parser.add_argument("--checkpoint-every-epochs", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.run_directory:
        config["training"]["run_directory"] = args.run_directory
    if args.data_root:
        config["data"]["roots"] = [args.data_root]
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.resume:
        config["training"]["resume"] = args.resume
    if args.max_steps is not None:
        config["training"]["max_steps"] = args.max_steps
    if args.max_runtime_minutes is not None:
        config["training"]["max_runtime_minutes"] = args.max_runtime_minutes
    if args.checkpoint_every_epochs is not None:
        config["training"]["checkpoint_every_epochs"] = args.checkpoint_every_epochs
    output = train_distributed(config)
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"Distributed training complete: {output}", flush=True)


if __name__ == "__main__":
    main()
