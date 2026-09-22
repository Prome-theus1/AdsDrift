"""Three-condition DDP training with one complete condition on each GPU."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from AdsDrift.model.drifting.loss import build_drifting_loss
from AdsDrift.model.drifting.coordinate_gradient_balancing import (
    build_coordinate_gradient_balancer,
)
from AdsDrift.model.drifting.mace_features import build_mace_features
from AdsDrift.model.generator import build_generator
from AdsDrift.model.initialize.r0_resampling import OnlineR0Config, OnlineR0Resampler
from AdsDrift.model.model import AdsorptionDriftingObjective
from AdsDrift.model.utils.data_loader import ConditionBank, move_batch
from AdsDrift.model.utils.train import (
    _configure_numerics,
    _metric_values,
    _resolve_path,
    _seed_everything,
    _update_latest,
    load_config,
)


def _save_checkpoint(
    path: Path,
    generator: DistributedDataParallel,
    feature_encoder: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: dict,
    epoch: int,
    step: int,
    rank_system_ids: list[str],
    rng_states: list[dict],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "format_version": 3,
            "distributed_condition_parallel": True,
            "coordinate_gradient_routing": "mace_projected_balanced_first_order",
            "world_size": dist.get_world_size(),
            "rank_system_ids": rank_system_ids,
            "generator": generator.module.state_dict(),
            "generator_config": config["generator"],
            "mace_config": asdict(feature_encoder.config),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "numpy_rng_states_by_rank": rng_states,
            "epoch": epoch,
            "step": step,
            "config": {
                key: value for key, value in config.items() if not key.startswith("_")
            },
        },
        temporary,
    )
    temporary.replace(path)


def train_condition_parallel(config: dict) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("Condition-parallel training requires CUDA")
    if config["training"].get("resume"):
        raise ValueError("The first condition-parallel experiment starts from scratch")

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    training = config["training"]
    allow_tf32 = _configure_numerics(training)
    base_seed = int(training.get("seed", 0))
    # Each rank must see exactly the same per-system sampling stream that the
    # corresponding one-system reference run sees. DDP broadcasts rank-0 model
    # weights, so the common seed also gives an identical initial generator.
    _seed_everything(base_seed)
    rng = np.random.default_rng(base_seed)

    roots = [_resolve_path(value) for value in config["data"]["roots"]]
    movable_value = config["data"].get("movable_feature_root")
    movable_root = _resolve_path(movable_value) if movable_value else None
    bank = ConditionBank(roots, movable_root)
    if len(bank) != world_size:
        raise ValueError(
            f"Condition parallelism needs one system per rank: "
            f"{len(bank)} systems for {world_size} ranks"
        )
    system = bank.systems[rank]
    rank_system_ids = bank.system_ids
    expected_r0 = int(config["data"].get("r0_per_system", 100))
    if any(item.random_count != expected_r0 for item in bank.systems):
        raise ValueError("Every condition must contain the configured R0 count")

    online_config = OnlineR0Config.from_dict(
        config["data"].get("online_r0_resampling")
    )
    online_r0 = OnlineR0Resampler([system], online_config)
    positives_per_mode = int(config["data"].get("positives_per_mode", 1))
    max_modes = config["data"].get("max_positive_modes")
    max_modes = None if max_modes is None else int(max_modes)

    raw_generator = build_generator(config["generator"]).to(device)
    generator = DistributedDataParallel(
        raw_generator,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        find_unused_parameters=True,
    )
    feature_encoder = build_mace_features().to(device)
    objective = AdsorptionDriftingObjective(
        generator,
        feature_encoder,
        build_drifting_loss(dict(config["drifting"])).to(device),
        build_coordinate_gradient_balancer(
            config.get("coordinate_gradient_balancing")
        ).to(device),
    ).to(device)
    objective.train()

    optimizer = torch.optim.AdamW(
        generator.module.parameters(),
        lr=float(training["learning_rate"]),
        betas=tuple(float(value) for value in training.get("betas", [0.9, 0.95])),
        weight_decay=float(training.get("weight_decay", 0.01)),
    )
    epochs = int(training["epochs"])
    warmup_epochs = int(training.get("warmup_epochs", 10))
    if epochs < 1 or warmup_epochs < 0:
        raise ValueError("Epochs must be positive and warmup nonnegative")

    def schedule(step_index: int) -> float:
        if warmup_epochs and step_index < warmup_epochs:
            return max(step_index, 1) / warmup_epochs
        progress = (step_index - warmup_epochs) / max(epochs - warmup_epochs, 1)
        return 0.5 * (
            1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0))
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    run_directory = _resolve_path(training["run_directory"])
    log_path = run_directory / "metrics.jsonl"
    if rank == 0:
        run_directory.mkdir(parents=True, exist_ok=False)
        (run_directory / "resolved_config.json").write_text(
            json.dumps(
                {key: value for key, value in config.items() if not key.startswith("_")},
                indent=2,
                ensure_ascii=False,
            )
            + "\n"
        )
        print(
            json.dumps(
                {
                    "event": "condition_parallel_start",
                    "world_size": world_size,
                    "one_complete_condition_per_gpu": True,
                    "rank_system_ids": rank_system_ids,
                    "r0_per_rank": expected_r0,
                    "joint_updates_per_epoch": 1,
                    "target_epochs": epochs,
                    "warmup_epochs": warmup_epochs,
                    "generator_parameters": generator.module.parameter_count,
                    "allow_tf32": allow_tf32,
                    "mace_microbatch_size": feature_encoder.config.microbatch_size,
                    "online_r0_resampling": online_config.enabled,
                    "coordinate_gradient_routing": "mace_projected_balanced_first_order",
                }
            ),
            flush=True,
        )
    dist.barrier()

    log_every = int(training.get("log_every_steps", 10))
    checkpoint_every = int(training.get("checkpoint_every_epochs", 500))
    gradient_clip = float(training.get("gradient_clip_norm", 2.0))
    training_started = time.monotonic()
    try:
        for epoch in range(epochs):
            # Mirror the one-system trainer's ``rng.permutation(len(bank))``.
            # permutation(1) is intentionally retained for exact stream parity.
            rng.permutation(1)
            batch = bank.make_batch(
                [system],
                generated_per_condition=expected_r0,
                positives_per_mode=positives_per_mode,
                max_positive_modes=max_modes,
                rng=rng,
            )
            r0_diagnostics = {"r0_source": "stored_fixed_bank"}
            if online_config.enabled:
                r0_started = time.perf_counter()
                sampled_r0, r0_diagnostics = online_r0.sample(
                    system, epoch, expected_r0
                )
                r0_diagnostics["r0_sampling_seconds"] = (
                    time.perf_counter() - r0_started
                )
                batch["r0_positions"] = torch.from_numpy(sampled_r0[None])
            batch = move_batch(batch, device)

            optimizer.zero_grad(set_to_none=True)
            started = time.perf_counter()
            learning_rate_used = optimizer.param_groups[0]["lr"]
            loss, metrics, _ = objective(batch)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                generator.module.parameters(), gradient_clip, error_if_nonfinite=True
            )
            optimizer.step()
            scheduler.step()
            step = epoch + 1

            if step % log_every == 0:
                local_record = {
                    "epoch": epoch,
                    "step": step,
                    "rank": rank,
                    "system_id": system.system_id,
                    "positive_modes": system.mode_count,
                    "positive_samples_used": int(batch["positive_modes"].numel()),
                    "learning_rate": learning_rate_used,
                    "gradient_norm": float(gradient_norm.detach().cpu()),
                    "seconds": time.perf_counter() - started,
                    **r0_diagnostics,
                    **_metric_values(metrics),
                }
                gathered = [None] * world_size if rank == 0 else None
                dist.gather_object(local_record, gathered, dst=0)
                if rank == 0:
                    with log_path.open("a") as handle:
                        for record in gathered:
                            handle.write(json.dumps(record) + "\n")
                    print(
                        json.dumps(
                            {
                                "event": "joint_step",
                                "step": step,
                                "systems": gathered,
                            }
                        ),
                        flush=True,
                    )

            if step % checkpoint_every == 0 or step == epochs:
                gathered_states = [None] * world_size if rank == 0 else None
                dist.gather_object(rng.bit_generator.state, gathered_states, dst=0)
                if rank == 0:
                    checkpoint_path = run_directory / f"epoch_{step:04d}.pt"
                    _save_checkpoint(
                        checkpoint_path,
                        generator,
                        feature_encoder,
                        optimizer,
                        scheduler,
                        config,
                        epoch,
                        step,
                        rank_system_ids,
                        gathered_states,
                    )
                    _update_latest(checkpoint_path)
                    print(
                        json.dumps(
                            {
                                "event": "checkpoint",
                                "epoch": epoch,
                                "step": step,
                                "elapsed_seconds": time.monotonic() - training_started,
                                "path": str(checkpoint_path),
                            }
                        ),
                        flush=True,
                    )
                dist.barrier()
        return run_directory
    finally:
        dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-directory", required=True)
    parser.add_argument("--system-directories", nargs="+", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config["training"]["run_directory"] = args.run_directory
    config["training"]["resume"] = None
    config["data"]["roots"] = args.system_directories
    output = train_condition_parallel(config)
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"Condition-parallel training complete: {output}", flush=True)


if __name__ == "__main__":
    main()
