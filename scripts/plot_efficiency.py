"""Aggregate saved timing runs and create efficiency comparison figures."""

from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.text import Text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "timing_results"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "efficiency"
PASTEL_COLORS = (
    "#5B8DB8",
    "#A7D8F0",
    "#A9D6C3",
    "#AE97D3",)


def make_figure_text_bold(figure) -> None:
    for text in figure.findobj(match=Text):
        text.set_fontweight("bold")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build plots from independent benchmark_efficiency.py runs."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--config-name",
        action="append",
        default=[],
        help="Include only this timing config; may be repeated.",
    )
    parser.add_argument(
        "--baseline",
        default="sasrec",
        help="Saved model name used for percentage deltas.",
    )
    parser.add_argument(
        "--model-label",
        action="append",
        default=[],
        metavar="NAME=LABEL",
        help="Optional display-name mapping; may be repeated.",
    )
    parser.add_argument(
        "--dataset-label",
        action="append",
        default=[],
        metavar="NAME=LABEL",
        help="Optional display-name mapping; may be repeated.",
    )
    return parser.parse_args()


def parse_label_map(values: list[str]) -> dict[str, str]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected NAME=LABEL, got {value!r}.")
        name, label = value.split("=", 1)
        result[name] = label
    return result


def load_runs(input_dir: Path, config_names: list[str]) -> list[dict[str, Any]]:
    result_dir = input_dir.resolve() / "results"
    if config_names:
        paths = sorted(
            path
            for config_name in config_names
            for path in (result_dir / config_name).glob("*.json")
        )
    else:
        paths = sorted(result_dir.glob("**/*.json"))
    runs = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") == 1 and "timing" in payload:
            payload["_path"] = str(path)
            runs.append(payload)
    if not runs:
        raise FileNotFoundError(f"No timing JSON files found below {result_dir}.")
    return runs


def validate_runs(runs: list[dict[str, Any]], baseline: str) -> None:
    hashes: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    configs: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    for run in runs:
        key = (run["model"], run["dataset"])
        hashes[key].add(run["config_hash"])
        configs[key].add(run["config_name"])

    inconsistent = {
        key: (hashes[key], configs[key])
        for key in hashes
        if len(hashes[key]) > 1
    }
    if inconsistent:
        details = "; ".join(
            f"{model}/{dataset}: configs={sorted(config_names)}, hashes={sorted(values)}"
            for (model, dataset), (values, config_names) in inconsistent.items()
        )
        raise ValueError(
            "Cannot average runs with different resolved parameters for the same "
            f"model/dataset pair. Separate them into different input directories. {details}"
        )

    environments = {
        (
            tuple(run["environment"].get("gpu_names", [])),
            run["environment"].get("platform"),
            run["environment"].get("torch"),
            run["environment"].get("numpy"),
            run["environment"].get("scipy"),
            run["source_config"]["model"].get("device"),
        )
        for run in runs
    }
    if len(environments) > 1:
        raise ValueError(
            "Timing samples were collected on different hardware/software environments. "
            "Put comparable runs in one input directory and move the others elsewhere."
        )

    inference_protocols = {run["protocol"].get("inference") for run in runs}
    if len(inference_protocols) > 1:
        raise ValueError(
            "Timing samples use different numbers of test holdouts and cannot be averaged."
        )

    models = {run["model"] for run in runs}
    datasets = {run["dataset"] for run in runs}
    if baseline not in models:
        raise ValueError(f"Baseline {baseline!r} is absent. Available models: {sorted(models)}")

    available = {(run["model"], run["dataset"]) for run in runs}
    missing = [
        (model, dataset)
        for model in models
        for dataset in datasets
        if (model, dataset) not in available
    ]
    if missing:
        formatted = ", ".join(f"{model}/{dataset}" for model, dataset in missing)
        raise ValueError(f"The comparison matrix is incomplete. Missing runs: {formatted}")


def default_model_label(name: str) -> str:
    normalized = name.lower()
    if normalized == "sasrec":
        return "SASRec"
    if normalized.startswith("sasrec_"):
        suffix = name[len("sasrec_") :].replace("_", " ")
        return f"SASRec + {suffix}"
    return name.replace("_", " ")


def default_dataset_label(name: str) -> str:
    for prefix in ("Amazon2014-", "Amazon2018-"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    return name.replace("-And-", " & ").replace("-", " ")


def grouped_stats(
    runs: Iterable[dict[str, Any]],
    extractor,
) -> dict[tuple[str, str], tuple[float, float, int]]:
    grouped: defaultdict[tuple[str, str], list[float]] = defaultdict(list)
    for run in runs:
        value = extractor(run)
        if value is not None:
            grouped[(run["model"], run["dataset"])].append(float(value))

    result = {}
    for key, values in grouped.items():
        array = np.asarray(values, dtype=float)
        result[key] = (
            float(array.mean()),
            float(array.std(ddof=1)) if len(array) > 1 else 0.0,
            len(array),
        )
    return result


def ordered_names(runs: list[dict[str, Any]], baseline: str) -> tuple[list[str], list[str]]:
    models = sorted({run["model"] for run in runs})
    models.remove(baseline)
    models.insert(0, baseline)
    datasets = sorted({run["dataset"] for run in runs})
    return models, datasets


def annotate_bars(
    ax,
    containers,
    models: list[str],
    datasets: list[str],
    stats: dict[tuple[str, str], tuple[float, float, int]],
    baseline: str,
) -> None:
    largest = max((row[0] for row in stats.values()), default=1.0)
    for dataset_index, container in enumerate(containers):
        dataset = datasets[dataset_index]
        baseline_value = stats[(baseline, dataset)][0]
        for model_index, bar in enumerate(container):
            model = models[model_index]
            value, _, _ = stats[(model, dataset)]
            if model == baseline:
                label = f"{value:.2f}s"
            else:
                delta = 100.0 * (value / baseline_value - 1.0)
                label = f"{value:.2f}s ({delta:+.1f}%)"
            ax.text(
                bar.get_width() + 0.012 * largest,
                bar.get_y() + bar.get_height() / 2,
                label,
                va="center",
                fontsize=8,
                fontweight="bold",
            )


def plot_main_comparison(
    runs: list[dict[str, Any]],
    output_dir: Path,
    baseline: str,
    model_labels: dict[str, str],
    dataset_labels: dict[str, str],
) -> None:
    models, datasets = ordered_names(runs, baseline)
    y = np.arange(len(models), dtype=float)
    height = 0.8 / len(datasets)
    colors = PASTEL_COLORS
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(15, max(4.2, 0.75 * len(models) + 2.2)),
    )

    metrics = (
        ("train_seconds_per_epoch", "Train time/epoch (s)"),
        ("inference_mean_seconds", "Inference time/test holdout (s)"),
    )
    for ax, (metric, xlabel) in zip(axes, metrics):
        stats = grouped_stats(runs, lambda run: run["timing"][metric])
        containers = []
        for dataset_index, dataset in enumerate(datasets):
            offset = (dataset_index - (len(datasets) - 1) / 2) * height
            means = [stats[(model, dataset)][0] for model in models]
            errors = [stats[(model, dataset)][1] for model in models]
            container = ax.barh(
                y + offset,
                means,
                height=height,
                xerr=errors if any(value > 0 for value in errors) else None,
                capsize=3,
                label=dataset_labels.get(dataset, default_dataset_label(dataset)),
                color=colors[dataset_index % len(colors)],
                alpha=0.8,
            )
            containers.append(container)

        annotate_bars(ax, containers, models, datasets, stats, baseline)
        largest = max(row[0] for row in stats.values())
        ax.set_xlim(0, largest * 1.5 if largest > 0 else 1.0)
        ax.set_xlabel(xlabel)
        ax.set_yticks(
            y,
            labels=[model_labels.get(name, default_model_label(name)) for name in models],
        )
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.25)
        ax.set_axisbelow(True)

    axes[0].legend(loc="best", frameon=True)
    make_figure_text_bold(figure)
    figure.tight_layout()
    figure.savefig(output_dir / "efficiency_comparison.png", dpi=220, bbox_inches="tight")
    figure.savefig(output_dir / "efficiency_comparison.pdf", bbox_inches="tight")
    plt.close(figure)


def train_val_preprocessing(run: dict[str, Any], key: str) -> float | None:
    for row in run.get("preprocessing", []):
        if row.get("phase") == "train_val":
            return float(row[key])
    return None


def plot_preprocessing(
    runs: list[dict[str, Any]],
    output_dir: Path,
    model_labels: dict[str, str],
    dataset_labels: dict[str, str],
) -> None:
    preprocessing_runs = [run for run in runs if run.get("preprocessing")]
    if not preprocessing_runs:
        return

    models = sorted({run["model"] for run in preprocessing_runs})
    datasets = sorted({run["dataset"] for run in preprocessing_runs})
    available = {(run["model"], run["dataset"]) for run in preprocessing_runs}
    missing = [
        (model, dataset)
        for model in models
        for dataset in datasets
        if (model, dataset) not in available
    ]
    if missing:
        formatted = ", ".join(f"{model}/{dataset}" for model, dataset in missing)
        raise ValueError(f"The preprocessing matrix is incomplete. Missing runs: {formatted}")

    y = np.arange(len(models), dtype=float)
    height = 0.8 / len(datasets)
    colors = PASTEL_COLORS
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(15, max(3.8, 0.75 * len(models) + 2.2)),
    )

    metric_specs = (
        ("eigendecomposition_seconds", "Eigendecomposition on train+val (s)"),
        ("total_preprocessing_seconds", "Total Laplacian preprocessing (s)"),
    )
    for ax, (metric, xlabel) in zip(axes, metric_specs):
        stats = grouped_stats(
            preprocessing_runs,
            lambda run, key=metric: train_val_preprocessing(run, key),
        )
        largest = max(row[0] for row in stats.values())
        for dataset_index, dataset in enumerate(datasets):
            offset = (dataset_index - (len(datasets) - 1) / 2) * height
            means = [stats[(model, dataset)][0] for model in models]
            errors = [stats[(model, dataset)][1] for model in models]
            bars = ax.barh(
                y + offset,
                means,
                height=height,
                xerr=errors if any(value > 0 for value in errors) else None,
                capsize=3,
                label=dataset_labels.get(dataset, default_dataset_label(dataset)),
                color=colors[dataset_index % len(colors)],
                alpha=0.8,
            )
            for model_index, bar in enumerate(bars):
                mean, _, _ = stats[(models[model_index], dataset)]
                ax.text(
                    bar.get_width() + 0.012 * largest,
                    bar.get_y() + bar.get_height() / 2,
                    f"{mean:.2f}s",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                )

        ax.set_xlim(0, largest * 1.35 if largest > 0 else 1.0)
        ax.set_xlabel(xlabel)
        ax.set_yticks(
            y,
            labels=[model_labels.get(name, default_model_label(name)) for name in models],
        )
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.25)
        ax.set_axisbelow(True)

    axes[0].legend(loc="best", frameon=True)
    make_figure_text_bold(figure)
    figure.tight_layout()
    figure.savefig(output_dir / "eigendecomposition_time.png", dpi=220, bbox_inches="tight")
    figure.savefig(output_dir / "eigendecomposition_time.pdf", bbox_inches="tight")
    plt.close(figure)


def write_summary(runs: list[dict[str, Any]], output_dir: Path) -> None:
    rows = []
    grouped: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[(run["model"], run["dataset"])].append(run)

    for (model, dataset), values in sorted(grouped.items()):
        train = np.asarray(
            [run["timing"]["train_seconds_per_epoch"] for run in values], dtype=float
        )
        inference = np.asarray(
            [run["timing"]["inference_mean_seconds"] for run in values], dtype=float
        )
        rows.append(
            {
                "model": model,
                "dataset": dataset,
                "config_name": values[0]["config_name"],
                "config_hash": values[0]["config_hash"],
                "n_runs": len(values),
                "train_seconds_per_epoch_mean": float(train.mean()),
                "train_seconds_per_epoch_std": (
                    float(train.std(ddof=1)) if len(train) > 1 else 0.0
                ),
                "inference_seconds_mean": float(inference.mean()),
                "inference_seconds_std": (
                    float(inference.std(ddof=1)) if len(inference) > 1 else 0.0
                ),
            }
        )
    (output_dir / "timing_summary.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    runs = load_runs(args.input_dir, args.config_name)
    validate_runs(runs, args.baseline)
    model_labels = parse_label_map(args.model_label)
    dataset_labels = parse_label_map(args.dataset_label)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_main_comparison(
        runs,
        output_dir,
        args.baseline,
        model_labels,
        dataset_labels,
    )
    plot_preprocessing(runs, output_dir, model_labels, dataset_labels)
    write_summary(runs, output_dir)
    print(f"Plots and timing_summary.json saved to {output_dir}")


if __name__ == "__main__":
    main()
