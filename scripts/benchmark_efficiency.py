"""Measure one model/dataset configuration using the run_model.py protocol.

Run this script once per timing sample. The Hydra root config passed with
``-cn`` determines both the model and the dataset, including every parameter
from ``configs/<config_name>.yaml`` and its ``configs/model/...`` default.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import scipy
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from scipy.sparse.linalg import eigsh

from scripts.precompute_laplacian_eigvecs import build_normalized_laplacian
from src.metrics import Summarizer
from src.utils import fix_random_seed


CONFIG_DIR = PROJECT_ROOT / "configs"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "timing_results"
SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure one Hydra model config. Re-run the command to collect "
            "independent timing samples."
        )
    )
    parser.add_argument(
        "-cn",
        "--config-name",
        required=True,
        help="Hydra root config name from configs/, without the .yaml suffix.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory in which independent result JSON files are accumulated.",
    )
    parser.add_argument(
        "--test-holdouts",
        type=int,
        default=10,
        help="Number of holdout_test_<id>.csv files; run_model.py uses 10.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Optional explicit Hydra overrides. They are saved in the result JSON.",
    )
    args = parser.parse_args()
    if args.test_holdouts < 1:
        parser.error("--test-holdouts must be at least 1.")
    return args


def compose_config(config_name: str, overrides: list[str]) -> DictConfig:
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name=config_name, overrides=overrides)


def clone_config(cfg: DictConfig) -> DictConfig:
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))


def resolved_config_hash(cfg: DictConfig) -> str:
    container = OmegaConf.to_container(cfg, resolve=True)
    if isinstance(container, dict):
        # Hydra output paths may contain ${now:...}; they do not affect the
        # measured model and must not make identical reruns look different.
        container.pop("hydra", None)
    serialized = json.dumps(container, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def sync_accelerator(model: Any) -> None:
    device = torch.device(str(getattr(model, "device", "cpu")))
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def timed_call(model: Any, function, *args, **kwargs):
    sync_accelerator(model)
    started = perf_counter()
    result = function(*args, **kwargs)
    sync_accelerator(model)
    return result, perf_counter() - started


def required_eigenvector_columns(cfg: DictConfig) -> int:
    k = cfg.model.get("k", None)
    if k is None:
        k = cfg.model.get("hidden_size", None)
    if k is None:
        raise ValueError(
            f"Cannot determine k for Laplacian model '{cfg.model.get('name', '<unnamed>')}'."
        )

    k = int(k)
    if k <= 0:
        raise ValueError(f"Invalid eigenvector count: k={k}.")
    return k


def precompute_eigenvectors(
    dataset_cfg: DictConfig,
    requested_k: int,
    merge_train_val: bool,
    destination: Path,
    seed: int,
) -> dict[str, Any]:
    dataset = instantiate(dataset_cfg, split="train", merge_train_val=merge_train_val)

    total_started = perf_counter()
    graph_started = perf_counter()
    laplacian, n_components = build_normalized_laplacian(dataset)
    graph_seconds = perf_counter() - graph_started

    if requested_k >= dataset.n_items:
        raise ValueError(
            f"eigsh requires k < n_items for {dataset.name}: "
            f"k={requested_k}, n_items={dataset.n_items}."
        )
    eigsh_k = requested_k

    # Match run_model.py, which resets the global RNG before both the
    # validation fit and the train+val retrain. Do not pass an explicit v0:
    # SciPy constructs it internally, just as in SASRecEinv._post_init().
    np.random.seed(seed)
    eig_started = perf_counter()
    eigvals, eigvecs = eigsh(laplacian, k=eigsh_k, which="SM")
    eig_seconds = perf_counter() - eig_started

    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    pos_enc = eigvecs[:, :requested_k].astype(np.float32, copy=False)

    metadata = {
        "dataset": str(dataset.name),
        "split": "train",
        "merge_train_val": merge_train_val,
        "n_users": int(dataset.n_users),
        "n_items": int(dataset.n_items),
        "n_components": int(n_components),
        "requested_k": int(requested_k),
        "eigsh_k": int(eigsh_k),
        "eigenvector_basis": "smallest_from_index_0",
        "dtype": "float32",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.save(destination, pos_enc)
    total_seconds = perf_counter() - total_started

    return {
        **metadata,
        "phase": "train_val" if merge_train_val else "train",
        "graph_build_seconds": graph_seconds,
        "eigendecomposition_seconds": eig_seconds,
        "postprocess_and_save_seconds": max(
            0.0, total_seconds - graph_seconds - eig_seconds
        ),
        "total_preprocessing_seconds": total_seconds,
        "cache_path": str(destination.resolve()),
    }


def prepare_eigenvector_caches(
    cfg: DictConfig,
    artifact_dir: Path,
) -> tuple[list[dict[str, Any]], dict[bool, Path]]:
    if "lap_eigvec_path" not in cfg.model:
        return [], {}

    configured_path = cfg.model.get("lap_eigvec_path", None)
    if configured_path is not None:
        path = Path(str(configured_path)).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Laplacian eigenvector file does not exist: {path}")
        return [], {False: path, True: path}

    requested_k = required_eigenvector_columns(cfg)
    seed = int(cfg.get("seed", 42))
    records = []
    paths = {}
    for merge_train_val in (False, True):
        phase = "train_val" if merge_train_val else "train"
        path = artifact_dir / f"{phase}_lap_eigvecs.npy"
        records.append(
            precompute_eigenvectors(
                cfg.dataset,
                requested_k=requested_k,
                merge_train_val=merge_train_val,
                destination=path,
                seed=seed,
            )
        )
        paths[merge_train_val] = path.resolve()
    return records, paths


def aggregate_preprocessing(records: list[dict[str, Any]]) -> dict[str, float]:
    fields = (
        "graph_build_seconds",
        "eigendecomposition_seconds",
        "postprocess_and_save_seconds",
        "total_preprocessing_seconds",
    )
    return {
        field: float(sum(float(record[field]) for record in records))
        for field in fields
    }


def set_eigenvector_cache(cfg: DictConfig, path: Path | None) -> None:
    if "lap_eigvec_path" not in cfg.model:
        return
    if path is None:
        raise ValueError("Laplacian cache path was not prepared.")
    with open_dict(cfg.model):
        cfg.model.lap_eigvec_path = str(path)


def instantiate_metrics(cfg: DictConfig, n_items: int) -> Summarizer:
    return Summarizer(
        [instantiate(metric_cfg, n_items=n_items) for metric_cfg in cfg.metrics]
    )


def evaluate_predictions(
    metrics: Summarizer,
    dataset: Any,
    predictions: np.ndarray,
) -> dict[str, float]:
    users = dataset.get_holdout_users()
    return {
        key: float(value)
        for key, value in metrics(
            predictions[users, :],
            dataset.get_holdout_array()[users],
        ).items()
    }


def aggregate_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    result = {}
    for name in rows[0]:
        values = np.asarray([row[name] for row in rows], dtype=float)
        result[name] = float(values.mean())
        result[f"{name}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return result


def apply_selected_model_params(cfg: DictConfig, model: Any) -> dict[str, Any]:
    selected = (
        model.suggest_additional_params()
        if hasattr(model, "suggest_additional_params")
        else {}
    )
    if not isinstance(selected, dict):
        return {}
    with open_dict(cfg.model):
        for name, value in selected.items():
            OmegaConf.update(cfg.model, name, value, merge=True)
    return selected


def run_protocol(
    source_cfg: DictConfig,
    test_holdouts: int,
    eigvec_paths: dict[bool, Path],
) -> dict[str, Any]:
    cfg = clone_config(source_cfg)
    seed = int(cfg.get("seed", 42))

    # Phase 1: identical to run_train(..., test_mode=False) in run_model.py.
    fix_random_seed(seed)
    set_eigenvector_cache(cfg, eigvec_paths.get(False))
    validation_model = instantiate(cfg.model)
    train_dataset = instantiate(cfg.dataset, split="train", merge_train_val=False)
    validation_dataset = instantiate(
        cfg.dataset,
        split="val",
        merge_train_val=False,
        holdout_filename="holdout_validation.csv",
    )
    # The first fit must use validation so that the model's early-stopping
    # logic selects the epoch count for the final train+val retrain.
    validation_model.fit(train_dataset, validation_dataset)
    validation_predictions = validation_model.predict(
        validation_dataset, top_n=int(cfg.max_top_n)
    )
    validation_metrics = evaluate_predictions(
        instantiate_metrics(cfg, train_dataset.n_items),
        validation_dataset,
        validation_predictions,
    )
    selected_params = apply_selected_model_params(cfg, validation_model)

    del validation_model, validation_predictions, validation_dataset, train_dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Phase 2: only this train+val fit contributes to reported training time.
    fix_random_seed(seed)
    set_eigenvector_cache(cfg, eigvec_paths.get(True))
    final_model = instantiate(cfg.model)
    train_val_dataset = instantiate(cfg.dataset, split="train", merge_train_val=True)
    final_epochs = int(getattr(final_model, "n_epochs", cfg.model.get("n_epochs", 1)))
    _, train_seconds = timed_call(final_model, final_model.fit, train_val_dataset, None)
    if final_epochs <= 0:
        raise ValueError(f"Final epoch count must be positive, got {final_epochs}.")

    inference_seconds = []
    test_metrics = []
    metrics = instantiate_metrics(cfg, train_val_dataset.n_items)
    for holdout_id in range(test_holdouts):
        test_dataset = instantiate(
            cfg.dataset,
            split="test",
            merge_train_val=True,
            holdout_filename=f"holdout_test_{holdout_id}.csv",
        )
        predictions, elapsed = timed_call(
            final_model,
            final_model.predict,
            test_dataset,
            top_n=int(cfg.max_top_n),
        )
        inference_seconds.append(elapsed)
        test_metrics.append(evaluate_predictions(metrics, test_dataset, predictions))

    result = {
        "seed": seed,
        "selected_model_params": selected_params,
        "final_epochs": final_epochs,
        "train_total_seconds": train_seconds,
        "train_seconds_per_epoch": train_seconds / final_epochs,
        "inference_seconds_per_holdout": inference_seconds,
        "inference_mean_seconds": float(np.mean(inference_seconds)),
        "inference_std_seconds": (
            float(np.std(inference_seconds, ddof=1))
            if len(inference_seconds) > 1
            else 0.0
        ),
        "validation_metrics": validation_metrics,
        "test_metrics": aggregate_metrics(test_metrics),
        "runtime_config": OmegaConf.to_container(cfg, resolve=True),
    }

    del final_model, train_val_dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def environment_metadata() -> dict[str, Any]:
    gpu_names = []
    if torch.cuda.is_available():
        gpu_names = [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ]
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_names": gpu_names,
    }


def main() -> None:
    args = parse_args()
    source_cfg = compose_config(args.config_name, args.overrides)
    # Fail on an invalid model config before starting an expensive eigensolve.
    probe_model = instantiate(source_cfg.model)
    del probe_model
    created_at = datetime.now().astimezone()
    run_id = created_at.strftime("%Y%m%d_%H%M%S_%f")
    output_root = args.output_dir.resolve()
    artifact_dir = output_root / "artifacts" / args.config_name / run_id
    result_dir = output_root / "results" / args.config_name
    result_dir.mkdir(parents=True, exist_ok=True)

    # Both split-specific caches are computed before the first model fit.
    preprocessing, eigvec_paths = prepare_eigenvector_caches(source_cfg, artifact_dir)
    timing = run_protocol(source_cfg, args.test_holdouts, eigvec_paths)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": created_at.isoformat(),
        "config_name": args.config_name,
        "config_hash": resolved_config_hash(source_cfg),
        "hydra_overrides": args.overrides,
        "model": str(source_cfg.model.name),
        "dataset": str(source_cfg.dataset.name),
        "protocol": {
            "selection_fit": "train",
            "timed_fit": "train+val",
            "training_timing_boundary": "model.fit(train_val_dataset, None)",
            "inference": f"mean of {args.test_holdouts} test holdouts",
            "inference_timing_boundary": "one model.predict(test_dataset) call",
            "dataset_loading_and_metric_computation_timed": False,
            "eigendecomposition_timed_separately": bool(preprocessing),
        },
        "environment": environment_metadata(),
        "preprocessing": preprocessing,
        "preprocessing_totals": aggregate_preprocessing(preprocessing),
        "timing": timing,
        "source_config": OmegaConf.to_container(source_cfg, resolve=True),
    }
    result_path = result_dir / f"{run_id}.json"
    result_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Timing result saved to {result_path}")


if __name__ == "__main__":
    main()
