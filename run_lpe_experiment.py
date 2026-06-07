import json
import numpy as np
from pathlib import Path
from tqdm import trange
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.utils import fix_random_seed
from src.metrics import Summarizer

from scipy.sparse import bmat, csr_matrix, diags, eye
from scipy.sparse.linalg import eigsh


def get_eigvecs_path(dataset_name):
    return Path(f"{dataset_name}_inter_eigvecs.npy")


def build_interaction_matrix(train_dataset):
    n_users = train_dataset.n_users
    n_items = train_dataset.n_items

    if hasattr(train_dataset, "get_coo_array"):
        coo = train_dataset.get_coo_array().tocoo()
        data = np.ones(coo.row.shape[0], dtype=np.float32)
        return csr_matrix((data, (coo.row, coo.col)), shape=(n_users, n_items))

    rows, cols = [], []
    for user in train_dataset:
        uid = user["user_id"]
        for item in user["history"].tolist():
            if 0 <= item < n_items:
                rows.append(uid)
                cols.append(item)

    data = np.ones(len(rows), dtype=np.float32)
    return csr_matrix((data, (rows, cols)), shape=(n_users, n_items))


def compute_and_save_interaction_eigvecs(train_dataset, dataset_name):
    path = get_eigvecs_path(dataset_name)

    if path.exists():
        print(f"Loading precomputed interaction eigvecs from {path}...")
        eigvecs = np.load(path)
        print(f"Loaded eigvecs shape: {eigvecs.shape}")
        if eigvecs.shape[0] != train_dataset.n_items:
            raise ValueError(
                f"Cached inter eigvec row count mismatch: got {eigvecs.shape[0]}, "
                f"expected {train_dataset.n_items}."
            )
        return eigvecs

    print("Computing interaction-graph eigenvectors (this may take a while)...")
    n_items = train_dataset.n_items
    n_users = train_dataset.n_users
    total_nodes = n_users + n_items

    R = build_interaction_matrix(train_dataset)
    zero_users = csr_matrix((n_users, n_users), dtype=np.float32)
    zero_items = csr_matrix((n_items, n_items), dtype=np.float32)
    adjacency = bmat([[zero_users, R], [R.T, zero_items]], format="csr")

    deg = np.asarray(adjacency.sum(axis=1)).ravel()
    with np.errstate(divide="ignore", invalid="ignore"):
        deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    d_inv_sqrt = diags(deg_inv_sqrt)
    laplacian = eye(total_nodes, format="csr") - d_inv_sqrt @ adjacency @ d_inv_sqrt

    eigsh_k = min(n_items, total_nodes - 1)
    eigvals, eigvecs = eigsh(laplacian, k=eigsh_k, which="SM")

    idx = np.argsort(eigvals)
    eigvecs = eigvecs[:, idx]
    eigvecs = eigvecs[:, 1:]
    item_eigvecs = eigvecs[n_users:, :]

    np.save(path, item_eigvecs)
    print(f"Interaction eigvecs saved to {path}, shape: {item_eigvecs.shape}")

    return item_eigvecs


def run_experiment(cfg, n_test_holdouts=10):
    all_metrics = []
    all_eigvec_indices = []
    all_val_runs = []

    dataset_name = cfg.dataset.name

    train_dataset = instantiate(
        cfg.dataset,
        split="train",
        merge_train_val=False
    )
    val_dataset = instantiate(
        cfg.dataset,
        split="val",
        merge_train_val=False,
        holdout_filename="holdout_validation.csv",
    )
    train_val_dataset = instantiate(
        cfg.dataset,
        split="train",
        merge_train_val=True
    )

    k = OmegaConf.select(cfg, "model.k")
    hidden_size = OmegaConf.select(cfg, "model.hidden_size")
    if k is None:
        k = hidden_size
    if hidden_size is not None and k != hidden_size:
        raise ValueError(
            f"run_lpe_experiment expects model.k == model.hidden_size for no-projection windows, "
            f"got k={k}, hidden_size={hidden_size}."
        )

    eigvecs = compute_and_save_interaction_eigvecs(train_dataset, dataset_name)
    total_available = eigvecs.shape[1]
    if k > total_available:
        raise ValueError(f"k={k} > total_available={total_available}")

    starts = list(range(0, total_available - k + 1, k))
    n_runs = len(starts)

    print(f"Total available eigvecs: {total_available}")
    print(f"Window size: {k}")
    print(f"Number of windows: {n_runs}")
    print(f"Number of test holdouts per run: {n_test_holdouts}")

    for run_idx in trange(n_runs, desc="Experiments"):
        start_idx = starts[run_idx]
        chosen_idx = np.arange(start_idx, start_idx + k)

        fix_random_seed(run_idx)

        model_val = instantiate(cfg.model)
        model_val.precomputed_eigvecs = eigvecs
        model_val.precomputed_chosen_idx = chosen_idx.tolist()
        model_val.fit(train_dataset, val_dataset)

        best_n_epochs = model_val.n_epochs
        print(f"\nRun {run_idx}: best epoch from val = {best_n_epochs}")

        fix_random_seed(run_idx)

        model_test = instantiate(cfg.model)
        model_test.precomputed_eigvecs = eigvecs
        model_test.precomputed_chosen_idx = chosen_idx.tolist()
        model_test.n_epochs = best_n_epochs
        model_test.max_patience = best_n_epochs + 1
        model_test.fit(train_val_dataset, None)

        chosen_indices = model_test.chosen_eigvec_indices

        test_metrics_list = []
        holdouts_used = []

        for test_idx in range(n_test_holdouts):
            holdout_filename = f"holdout_test_{test_idx}.csv"
            holdouts_used.append(holdout_filename)

            test_dataset_run = instantiate(
                cfg.dataset,
                split="test",
                merge_train_val=True,
                holdout_filename=holdout_filename,
            )

            predictions = model_test.predict(test_dataset_run, top_n=cfg.max_top_n)
            holdout_users = test_dataset_run.get_holdout_users()

            metrics = Summarizer([
                instantiate(metric_cfg, n_items=train_dataset.n_items)
                for metric_cfg in cfg.metrics
            ])
            metric_values = metrics(
                predictions[holdout_users, :],
                test_dataset_run.get_holdout_array()[holdout_users],
            )
            test_metrics_list.append(metric_values)

        metric_names = list(test_metrics_list[0].keys())
        averaged_metrics = {}
        for name in metric_names:
            values = np.array([m[name] for m in test_metrics_list], dtype=float)
            mean_val = float(np.mean(values))
            std_val = float(np.std(values, ddof=1))
            averaged_metrics[name] = mean_val
            averaged_metrics[name + "_std"] = std_val

        all_metrics.append(averaged_metrics)
        all_eigvec_indices.append(chosen_indices)
        all_val_runs.append({
            "run_idx": run_idx,
            "window": [int(start_idx), int(start_idx + k)],
            "best_n_epochs": best_n_epochs,
            "holdouts_used": holdouts_used,
            "test_metrics": test_metrics_list,
            "averaged_metrics": averaged_metrics,
            "eigenvector_indices": chosen_indices,
        })

    metric_names = [k for k in all_metrics[0].keys() if not k.endswith("_std")]
    aggregated = {}
    for name in metric_names:
        values = np.array([m[name] for m in all_metrics], dtype=float)
        aggregated[name] = float(np.mean(values))
        aggregated[name + "_std"] = float(np.std(values, ddof=1))

    return aggregated, all_metrics, all_eigvec_indices, all_val_runs


if __name__ == "__main__":
    @hydra.main(version_base=None, config_path="configs", config_name="sasrec_inter")
    def main(cfg):
        cfg = OmegaConf.to_container(cfg, resolve=True)
        cfg = OmegaConf.create(cfg)

        N_TEST_HOLDOUTS = 10

        aggregated, all_runs, all_eigvec_indices, all_val_runs = run_experiment(
            cfg, n_test_holdouts=N_TEST_HOLDOUTS
        )

        output = {
            "aggregated": aggregated,
            "runs": all_runs,
            "eigenvector_indices": all_eigvec_indices,
            "detailed_val_runs": all_val_runs,
            "config": {"n_test_holdouts": N_TEST_HOLDOUTS}
        }

        dataset_name = cfg.dataset.name
        out_file = f"lpe_window_{dataset_name}_inter_double_train.json"
        with open(out_file, "w") as f:
            json.dump(output, f, indent=2)

        print(f"\n=== FINAL RESULTS ===")
        for k, v in aggregated.items():
            print(f"{k}: {v:.6f}")

        print(f"\nResults saved to {out_file}")

    main()
