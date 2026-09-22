"""Train the test_18 joint-gradient one-pass adsorption generator."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch

from AdsDrift.model.utils.data_loader import ConditionBank, move_batch
from AdsDrift.model.model import AdsorptionDriftingObjective, build_model
from AdsDrift.model.initialize.r0_resampling import OnlineR0Config, OnlineR0Resampler


PROJECT_ROOT = Path(__file__).resolve().parents[4]


def _resolve_path(value: str) -> Path:
    path = Path(os.path.expandvars(value)).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_config(path: str | Path) -> dict:
    import yaml

    config_path = Path(path).expanduser().resolve()
    values = yaml.safe_load(config_path.read_text())
    if not isinstance(values, dict):
        raise ValueError("Training config must contain a mapping")
    values["_config_path"] = str(config_path)
    return values


def _select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _seed_everything(seed: int) -> None:
    if seed < 0:
        raise ValueError("Seed must be nonnegative")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _configure_numerics(training: dict) -> bool:
    """Apply the explicitly configured FP32 matrix-multiplication policy."""
    allow_tf32 = bool(training.get("allow_tf32", False))
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    return allow_tf32


def _metric_values(metrics: dict[str, torch.Tensor]) -> dict[str, float | int]:
    result = {}
    for name, value in metrics.items():
        scalar = value.detach().cpu().item()
        result[name] = int(scalar) if not value.dtype.is_floating_point else float(scalar)
    return result


def _save_checkpoint(
    path: Path,
    objective: AdsorptionDriftingObjective,
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
            "coordinate_gradient_routing": "mace_projected_balanced_first_order",
            "generator": objective.generator.state_dict(),
            "generator_config": config["generator"],
            "mace_config": asdict(objective.feature_encoder.config),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "step": step,
            "numpy_rng_state": rng.bit_generator.state,
            "config": {key: value for key, value in config.items() if not key.startswith("_")},
        },
        temporary,
    )
    temporary.replace(path)


def _update_latest(epoch_checkpoint: Path) -> None:
    """Update a small relative symlink without serializing a huge model twice."""
    latest = epoch_checkpoint.parent / "latest.pt"
    temporary = epoch_checkpoint.parent / ".latest.pt.tmp"
    if temporary.exists() or temporary.is_symlink():
        temporary.unlink()
    temporary.symlink_to(epoch_checkpoint.name)
    temporary.replace(latest)


def _advance_rng_through_completed_epochs(
    rng: np.random.Generator,
    bank: ConditionBank,
    completed_epochs: int,
    positives_per_mode: int,
    max_modes: int | None,
) -> None:
    """Reconstruct sampling state for legacy epoch-boundary checkpoints."""
    for _ in range(completed_epochs):
        order = rng.permutation(len(bank))
        for system_index in order:
            ConditionBank._sample_positive_indices(
                bank.systems[int(system_index)],
                positives_per_mode,
                max_modes,
                rng,
            )


def train(config: dict) -> Path:
    training = config["training"]
    allow_tf32 = _configure_numerics(training)
    seed = int(training.get("seed", 0))
    _seed_everything(seed)
    rng = np.random.default_rng(seed)
    device = _select_device(str(training.get("device", "auto")))

    roots = [_resolve_path(value) for value in config["data"]["roots"]]
    movable_value = config["data"].get("movable_feature_root")
    movable_root = _resolve_path(movable_value) if movable_value else None
    bank = ConditionBank(roots, movable_root)
    expected_r0 = int(config["data"].get("r0_per_system", 100))
    invalid = {
        system.system_id: system.random_count
        for system in bank.systems
        if system.random_count != expected_r0
    }
    if invalid:
        raise ValueError(
            f"Every system must contain exactly {expected_r0} R0 structures; found {invalid}"
        )
    online_r0_config = OnlineR0Config.from_dict(
        config["data"].get("online_r0_resampling")
    )
    online_r0 = OnlineR0Resampler(bank.systems, online_r0_config)

    drifting_config = dict(config["drifting"])
    objective = build_model(
        config["generator"],
        drifting_config,
        config.get("coordinate_gradient_balancing"),
    ).to(device)
    generator = objective.generator
    feature_encoder = objective.feature_encoder
    objective.train()

    optimizer = torch.optim.AdamW(
        objective.generator.parameters(),
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
    warmup_steps = warmup_epochs * len(bank)

    def schedule(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(step, 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    run_directory = _resolve_path(training["run_directory"])
    run_directory.mkdir(parents=True, exist_ok=True)
    (run_directory / "resolved_config.json").write_text(
        json.dumps({key: value for key, value in config.items() if not key.startswith("_")},
                   indent=2, ensure_ascii=False)
        + "\n"
    )
    log_path = run_directory / "metrics.jsonl"
    positives_per_mode = int(config["data"].get("positives_per_mode", 1))
    max_modes = config["data"].get("max_positive_modes")
    max_modes = None if max_modes is None else int(max_modes)
    start_epoch = 0
    step = 0
    resume = training.get("resume")
    if resume:
        checkpoint = torch.load(_resolve_path(resume), map_location=device, weights_only=False)
        objective.generator.load_state_dict(checkpoint["generator"])
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
        if not log_path.exists():
            source_log = _resolve_path(resume).parent / "metrics.jsonl"
            if source_log.is_file() and source_log.resolve() != log_path.resolve():
                shutil.copyfile(source_log, log_path)

    print(json.dumps({
        "event": "start",
        "device": str(device),
        "systems": len(bank),
        "r0_per_batch": expected_r0,
        "steps_per_epoch": len(bank),
        "generator_parameters": objective.generator.parameter_count,
        "allow_tf32": allow_tf32,
        "mace_microbatch_size": feature_encoder.config.microbatch_size,
        "mace_activation_checkpointing": feature_encoder.config.activation_checkpointing,
        "online_r0_resampling": online_r0_config.enabled,
        "online_r0_base_seed": (
            online_r0_config.base_seed if online_r0_config.enabled else None
        ),
        "coordinate_gradient_routing": "mace_projected_balanced_first_order",
        "coordinate_gradient_component_energy_fractions": list(
            objective.coordinate_gradient_balancer.config.component_energy_fractions
        ),
        "coordinate_gradient_preserve_total_energy": (
            objective.coordinate_gradient_balancer.config.preserve_total_energy
        ),
    }))
    log_every = int(training.get("log_every_steps", 1))
    checkpoint_every = int(training.get("checkpoint_every_epochs", 1))
    gradient_clip = float(training.get("gradient_clip_norm", 2.0))
    max_steps = training.get("max_steps")
    max_steps = None if max_steps is None else int(max_steps)
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be positive when provided")
    steps_this_run = 0
    max_runtime_minutes = training.get("max_runtime_minutes")
    max_runtime_seconds = (
        None if max_runtime_minutes is None else float(max_runtime_minutes) * 60.0
    )
    training_started = time.monotonic()

    for epoch in range(start_epoch, epochs):
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
            r0_diagnostics = {"r0_source": "stored_fixed_bank"}
            if online_r0_config.enabled:
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
                objective.generator.parameters(), gradient_clip, error_if_nonfinite=True
            )
            optimizer.step()
            scheduler.step()
            step += 1
            steps_this_run += 1
            if step % log_every == 0:
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
                    **r0_diagnostics,
                    **_metric_values(metrics),
                }
                with log_path.open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
                print(json.dumps(record))
            if max_steps is not None and steps_this_run >= max_steps:
                print(json.dumps({
                    "event": "max_steps_reached",
                    "steps_this_run": steps_this_run,
                    "global_step": step,
                }))
                return run_directory
        if (epoch + 1) % checkpoint_every == 0 or epoch + 1 == epochs:
            epoch_checkpoint = run_directory / f"epoch_{epoch + 1:04d}.pt"
            _save_checkpoint(
                epoch_checkpoint,
                objective,
                optimizer,
                scheduler,
                config,
                epoch,
                step,
                rng,
            )
            _update_latest(epoch_checkpoint)
        if max_runtime_seconds is not None and (
            time.monotonic() - training_started >= max_runtime_seconds
        ):
            print(json.dumps({
                "event": "runtime_budget_reached_at_epoch_boundary",
                "completed_epoch": epoch,
                "step": step,
            }))
            return run_directory
    return run_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-directory")
    parser.add_argument("--data-root")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--allow-tf32", choices=("true", "false"))
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
    if args.device:
        config["training"]["device"] = args.device
    if args.max_steps is not None:
        config["training"]["max_steps"] = args.max_steps
    if args.allow_tf32 is not None:
        config["training"]["allow_tf32"] = args.allow_tf32 == "true"
    if args.max_runtime_minutes is not None:
        config["training"]["max_runtime_minutes"] = args.max_runtime_minutes
    if args.checkpoint_every_epochs is not None:
        config["training"]["checkpoint_every_epochs"] = args.checkpoint_every_epochs
    output = train(config)
    print(f"Training complete: {output}")


if __name__ == "__main__":
    main()
