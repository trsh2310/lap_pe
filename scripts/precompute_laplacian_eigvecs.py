import argparse
import json
from pathlib import Path

import numpy as np
from hydra import compose, initialize
from hydra.utils import instantiate
from scipy.sparse import diags, eye
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh


def human_bytes(n_bytes: float) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(n_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024


def resolve_k(k_arg: str, n_items: int, n_components: int) -> int:
    if k_arg == "full":
        return n_items - n_components
    if k_arg == "n_items":
        return n_items
    if k_arg == "prev_power2":
        return 2 ** int(np.floor(np.log2(n_items)))
    if k_arg == "next_power2":
        return 2 ** int(np.ceil(np.log2(n_items)))
    if k_arg == "nearest_power2":
        prev_k = 2 ** int(np.floor(np.log2(n_items)))
        next_k = 2 ** int(np.ceil(np.log2(n_items)))
        return prev_k if abs(n_items - prev_k) <= abs(next_k - n_items) else next_k
    return int(k_arg)


def build_normalized_laplacian(dataset):
    r = dataset.get_coo_array().tocsr().astype(np.float32)
    a = (r.T @ r).tocsr()
    a.setdiag(0)
    a.eliminate_zeros()

    n_components, _ = connected_components(csgraph=a, directed=False)
    deg = np.asarray(a.sum(axis=1)).ravel()
    with np.errstate(divide="ignore", invalid="ignore"):
        deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)

    d_inv_sqrt = diags(deg_inv_sqrt)
    laplacian = eye(dataset.n_items, format="csr") - d_inv_sqrt @ a @ d_inv_sqrt
    return laplacian, n_components


def output_path(output_dir: Path, dataset_name: str, split: str, requested_k: int, computed_k: int) -> Path:
    safe_dataset = dataset_name.replace("/", "_")
    return output_dir / safe_dataset / f"{split}_lap_eigvecs_requested{requested_k}_computed{computed_k}.npz"


def main():
    parser = argparse.ArgumentParser(
        description="Precompute item Laplacian eigenvectors for SASRec LPE experiments."
    )
    parser.add_argument("--config-name", default="sasrec_mlp", help="Hydra root config name.")
    parser.add_argument("--dataset", required=True, help="Dataset config name, e.g. MovieLens-1M.")
    parser.add_argument(
        "--k",
        default="full",
        help="Integer, full, n_items, prev_power2, next_power2, or nearest_power2.",
    )
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--merge-train-val", action="store_true")
    parser.add_argument("--output-dir", default="data/lpe_cache")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--tol", type=float, default=0.0)
    parser.add_argument("--maxiter", type=int, default=None)
    parser.add_argument("--compressed", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with initialize(config_path="../configs", version_base=None):
        cfg = compose(config_name=args.config_name, overrides=[f"dataset={args.dataset}"])

    dataset = instantiate(
        cfg.dataset,
        split=args.split,
        merge_train_val=args.merge_train_val,
    )

    laplacian, n_components = build_normalized_laplacian(dataset)
    requested_k = resolve_k(args.k, dataset.n_items, n_components)
    if requested_k <= 0:
        raise ValueError(f"Resolved k must be positive, got {requested_k}.")

    eigsh_k = min(requested_k + n_components, dataset.n_items - 1)
    computed_k = max(0, eigsh_k - n_components)
    dtype = np.dtype(args.dtype)
    estimated_vec_bytes = dataset.n_items * max(computed_k, requested_k) * dtype.itemsize

    print(json.dumps({
        "dataset": dataset.name,
        "split": args.split,
        "merge_train_val": args.merge_train_val,
        "n_users": dataset.n_users,
        "n_items": dataset.n_items,
        "n_components": int(n_components),
        "requested_k": int(requested_k),
        "eigsh_k": int(eigsh_k),
        "computed_nontrivial_k": int(computed_k),
        "estimated_eigvec_storage": human_bytes(estimated_vec_bytes),
    }, indent=2))

    if requested_k + n_components >= dataset.n_items:
        print(
            "Warning: eigsh requires k < n_items. This will compute at most "
            f"{computed_k} non-trivial vectors, then downstream code can zero-pad to requested_k."
        )

    out_path = output_path(Path(args.output_dir), dataset.name, args.split, requested_k, computed_k)
    if args.dry_run:
        print(f"Dry run only. Output would be: {out_path}")
        return

    if out_path.exists() and not args.force:
        raise FileExistsError(f"{out_path} already exists. Use --force to overwrite.")

    eigvals, eigvecs = eigsh(laplacian, k=eigsh_k, which="SM", tol=args.tol, maxiter=args.maxiter)
    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    pos_enc = eigvecs[:, n_components:n_components + computed_k].astype(dtype, copy=False)
    nontrivial_eigvals = eigvals[n_components:n_components + computed_k].astype(dtype, copy=False)

    metadata = {
        "dataset": dataset.name,
        "split": args.split,
        "merge_train_val": args.merge_train_val,
        "n_users": dataset.n_users,
        "n_items": dataset.n_items,
        "n_components": int(n_components),
        "requested_k": int(requested_k),
        "eigsh_k": int(eigsh_k),
        "computed_nontrivial_k": int(computed_k),
        "dtype": str(dtype),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save = np.savez_compressed if args.compressed else np.savez
    save(
        out_path,
        eigvecs=pos_enc,
        eigvals=nontrivial_eigvals,
        metadata=np.array(json.dumps(metadata, sort_keys=True)),
    )
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
