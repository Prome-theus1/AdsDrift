#!/usr/bin/env python3
"""Render training_dashboard.png from a training directory's metrics.jsonl."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


def number(value):
    if isinstance(value, bool):
        return math.nan
    try:
        value = float(value)
        return value if math.isfinite(value) else math.nan
    except (ValueError, TypeError, OverflowError):
        return math.nan


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path, help="training/test directory containing metrics.jsonl")
    parser.add_argument("--window", type=int, default=51, help="trailing median window (default: 51 records)")
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args(argv)
    args.run_directory = args.run_directory.expanduser().resolve()
    args.log = args.run_directory / "metrics.jsonl"
    args.output = args.run_directory / "training_dashboard.png"
    if not args.log.is_file():
        parser.error(f"Training log not found: {args.log}")
    if args.window < 1 or args.dpi < 1:
        parser.error("--window and --dpi must be positive integers")
    return args


def load_cached_oracle(run_directory):
    """Read optional existing oracle statistics; never load a model or rewrite them."""
    path = run_directory / "training_metrics_summary.json"
    if not path.is_file():
        return None
    try:
        report = json.loads(path.read_text(encoding="utf-8")).get("oracle_displacement")
        if report is None:
            return None
        balanced = report["equal_mode_weighted"]
        for key in ("surface_coordinate_rms_A", "ads_center_coordinate_rms_A"):
            value = number(balanced[key]["global_rms_A"])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid oracle value: {key}")
            balanced[key]["global_rms_A"] = value
        return report
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        warnings.warn(f"Skipping invalid cached oracle statistics in {path}: {exc}", stacklevel=2)
        return None


def load_training_records(path):
    """Read complete JSON records and keep the newest copy of every global step."""
    by_step, segment, previous_step = {}, 0, None
    with Path(path).open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict) or record.get("event") not in (None, "train"):
                continue
            step = number(record.get("step"))
            if not math.isfinite(step) or step < 0 or not step.is_integer():
                continue
            step = int(step)
            if previous_step is not None and step <= previous_step:
                segment += 1
                by_step = {old_step: old for old_step, old in by_step.items() if old_step < step}
            by_step[step] = dict(record, step=step, _segment=segment)
            previous_step = step
    if not by_step:
        raise ValueError(f"No complete training records yet: {path}")
    return [by_step[step] for step in sorted(by_step)]


def _elapsed_intervals(records, key):
    result = np.full(len(records), np.nan)
    for index in range(1, len(records)):
        previous, current = records[index - 1], records[index]
        elapsed = number(current.get(key)) - number(previous.get(key))
        step_delta = current["step"] - previous["step"]
        if current["_segment"] == previous["_segment"] and elapsed > 0 and step_delta > 0:
            result[index] = elapsed / step_delta
    return result


def step_seconds(records, mode):
    """Return optimizer-step duration without differencing current per-step logs."""
    if mode == "per-step":
        key = "seconds_per_step" if any("seconds_per_step" in row for row in records) else "seconds"
        return np.asarray([number(row.get(key)) for row in records]), f"direct {key}"
    if mode == "elapsed":
        key = "elapsed_seconds" if any("elapsed_seconds" in row for row in records) else "seconds"
        return _elapsed_intervals(records, key), f"delta {key} / delta step"

    if any("seconds_per_step" in row for row in records):
        return np.asarray([number(row.get("seconds_per_step")) for row in records]), "direct seconds_per_step"
    if any("elapsed_seconds" in row for row in records):
        return _elapsed_intervals(records, "elapsed_seconds"), "delta elapsed_seconds / delta step"
    if any("batch_in_epoch" in row and "system_id" in row for row in records):
        return np.asarray([number(row.get("seconds")) for row in records]), "direct seconds"
    values = np.asarray([number(row.get("seconds")) for row in records])
    finite = values[np.isfinite(values)]
    monotonic = len(finite) > 2 and np.mean(np.diff(finite) >= 0) > 0.95
    return (_elapsed_intervals(records, "seconds"), "delta seconds / delta step") if monotonic else (values, "direct seconds")


def trailing_median(data, window, segments):
    result, start = np.full(len(data), np.nan), 0
    for index in range(len(data)):
        if index and segments[index] != segments[index - 1]:
            start = index
        chunk = data[max(start, index - window + 1):index + 1]
        finite = chunk[np.isfinite(chunk)]
        if len(finite):
            result[index] = np.median(finite)
    return result


def render(args):
    records = load_training_records(args.log)
    durations, time_definition = step_seconds(records, "auto")
    for record, duration in zip(records, durations):
        record["seconds_per_step"] = duration
    keys = {key for record in records for key in record}
    if "raw_drift_loss" not in keys and "mean_raw_drift_rms" in keys:
        # Older runs only stored mean(RMS(V)). This square is a visual fallback,
        # not the exact mean(V^2) now recorded by the trainer.
        for record in records:
            value = number(record.get("mean_raw_drift_rms"))
            record["raw_drift_loss_proxy"] = value * value
        keys.add("raw_drift_loss_proxy")

    steps = np.asarray([record["step"] for record in records])
    segments = np.asarray([record["_segment"] for record in records])
    oracle_report = load_cached_oracle(args.run_directory)

    plt.rcParams.update({
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
    })
    fig, axes = plt.subplots(3, 2, figsize=(14.2, 11.2), sharex=True)
    axes = axes.ravel()
    fig.patch.set_facecolor("#f7f9fc")

    def draw(ax, key, label, color, *, smooth=True, linewidth=1.8):
        data = np.asarray([number(record.get(key)) for record in records])
        filtered = trailing_median(data, args.window, segments) if smooth else data
        for index, segment in enumerate(np.unique(segments)):
            mask = segments == segment
            if smooth:
                ax.plot(steps[mask], data[mask], color=color, alpha=0.18, linewidth=0.7)
            ax.plot(steps[mask], filtered[mask], color=color, linewidth=linewidth,
                    label=label if index == 0 else None)
        return data

    raw_key = "raw_drift_loss" if "raw_drift_loss" in keys else "raw_drift_loss_proxy"
    raw_label = "L_raw = mean(V²)" if raw_key == "raw_drift_loss" else "Approx. L_raw"
    raw_loss = draw(axes[0], raw_key, raw_label, "#111827", linewidth=2.1)
    if np.isfinite(raw_loss).any() and np.nanmin(raw_loss) > 0:
        axes[0].set_yscale("log")
    axes[0].set(title="Raw drifting loss", ylabel="L_raw = mean(V²)")

    draw(
        axes[1], "minimum_mode_attraction_relative_uniform",
        "Weakest positive mode", "#7c3aed",
    )
    axes[1].axhline(1.0, color="#64748b", linestyle="--", linewidth=1, label="Uniform baseline")
    axes[1].axhline(0.0, color="#dc2626", linestyle=":", linewidth=1)
    axes[1].set(title="Mode attraction / coverage", ylabel="Attraction relative to uniform", ylim=(0, None))

    surrogate = draw(axes[2], "loss", "Normalized surrogate", "#1e3a8a")
    if np.isfinite(surrogate).any() and np.nanmin(surrogate) > 0:
        axes[2].set_yscale("log")
    axes[2].set(title="Optimization loss", ylabel="Normalized loss")

    for key, label, color in (
        ("surface_displacement_rms_A", "Movable surface", "#ea580c"),
        ("ads_center_displacement_rms_A", "Adsorbate center", "#16a34a"),
        ("ads_internal_displacement_rms_A", "Adsorbate internal", "#db2777"),
    ):
        draw(axes[3], key, label, color)
    if oracle_report is not None:
        balanced = oracle_report["equal_mode_weighted"]
        surface_oracle = balanced["surface_coordinate_rms_A"]["global_rms_A"]
        center_oracle = balanced["ads_center_coordinate_rms_A"]["global_rms_A"]
        axes[3].axhline(
            surface_oracle,
            color="#ea580c",
            linestyle="--",
            linewidth=1.2,
            label=f"Surface oracle, mode-balanced ({surface_oracle:.3f} Å)",
        )
        axes[3].axhline(
            center_oracle,
            color="#16a34a",
            linestyle="--",
            linewidth=1.2,
            label=f"Ads-center oracle, mode-balanced ({center_oracle:.3f} Å)",
        )
    axes[3].set(title="Generated structural displacement", ylabel="RMS displacement (Å)", ylim=(0, None))

    gradient = draw(axes[4], "gradient_norm", "Gradient norm (pre-clip)", "#b91c1c")
    if np.isfinite(gradient).any() and np.nanmin(gradient) > 0:
        axes[4].set_yscale("log")
    axes[4].set(title="Optimization", ylabel="Gradient norm")
    learning_axis = axes[4].twinx()
    learning_rate = np.asarray([number(record.get("learning_rate")) for record in records])
    learning_smooth = trailing_median(learning_rate, args.window, segments)
    learning_axis.plot(steps, learning_smooth, color="#0891b2", linewidth=1.6, label="Learning rate")
    learning_axis.set_ylabel("Learning rate", color="#0891b2")
    learning_axis.tick_params(axis="y", colors="#0891b2")
    times = draw(axes[5], "seconds_per_step", "Optimizer step", "#c2410c")
    axes[5].set(title="Training speed", ylabel="Seconds / step", ylim=(0, None))
    if not np.isfinite(times).any():
        axes[5].text(0.5, 0.5, "No timing records", transform=axes[5].transAxes, ha="center")

    for ax in axes:
        ax.set_xlabel("Global optimizer step")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        ax.grid(alpha=0.18)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, frameon=False, fontsize=8, loc="best")
    handles, labels = learning_axis.get_legend_handles_labels()
    axes[4].legend(frameon=False, fontsize=8, loc="upper left")
    if handles:
        learning_axis.legend(handles, labels, frameon=False, fontsize=8, loc="upper right")

    latest = records[-1]
    epoch_value = number(latest.get("epoch"))
    epoch_label = int(epoch_value) + 1 if math.isfinite(epoch_value) else "?"
    fixed_errors = np.asarray([number(record.get("fixed_coordinate_max_error_A")) for record in records])
    max_fixed_error = float(np.nanmax(fixed_errors)) if np.isfinite(fixed_errors).any() else None
    fig.suptitle(
        f"{args.log.parent.name}  |  epoch {epoch_label}  |  step {steps[-1]:,}",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.015,
        f"Light lines: logged values; dark lines: trailing median ({args.window} records). "
        f"Timing: {time_definition}. Fixed-coordinate max error: {max_fixed_error if max_fixed_error is not None else 'N/A'} Å.",
        ha="center",
        fontsize=8,
        color="#475569",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.965))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name("." + args.output.name + ".tmp.png")
    fig.savefig(temporary, dpi=args.dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
    os.replace(temporary, args.output)

    return args.output


def main(argv=None):
    args = parse_args(argv)
    try:
        output = render(args)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
