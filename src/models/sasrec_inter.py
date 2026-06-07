import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import bmat, csr_matrix, diags, eye
from scipy.sparse.linalg import eigsh

from src.models.sasrec_einv import SASRecBackBone, SASRecEinv


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


def build_bipartite_lpe(train_dataset, k, chosen_idx=None, precomputed_eigvecs=None):
    n_users = train_dataset.n_users
    n_items = train_dataset.n_items

    if precomputed_eigvecs is not None and chosen_idx is not None:
        if precomputed_eigvecs.shape[0] != n_items:
            raise ValueError(
                f"Precomputed inter eigvec row count mismatch: got {precomputed_eigvecs.shape[0]}, "
                f"expected {n_items}."
            )
        pos_enc = precomputed_eigvecs[:, chosen_idx]
        return torch.FloatTensor(pos_enc), list(chosen_idx)

    r = build_interaction_matrix(train_dataset)
    zero_users = csr_matrix((n_users, n_users), dtype=np.float32)
    zero_items = csr_matrix((n_items, n_items), dtype=np.float32)
    adjacency = bmat([[zero_users, r], [r.T, zero_items]], format="csr")

    deg = np.asarray(adjacency.sum(axis=1)).ravel()
    with np.errstate(divide="ignore", invalid="ignore"):
        deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    d_inv_sqrt = diags(deg_inv_sqrt)
    laplacian = eye(n_users + n_items, format="csr") - d_inv_sqrt @ adjacency @ d_inv_sqrt

    eigvals, eigvecs = eigsh(laplacian, k=k + 1, which="SM")
    order = np.argsort(eigvals)
    eigvecs = eigvecs[:, order]
    item_lpe = eigvecs[n_users:, 1:k + 1]

    return torch.FloatTensor(item_lpe), list(range(k))


class SASRecInter(SASRecEinv):
    def __init__(self, name: str = "sasrec_inter", **kwargs):
        super().__init__(name=name, **kwargs)

    def _post_init(self, train_dataset, val_dataset):
        if self.k != self.hidden_size:
            raise ValueError(
                f"SASRecInter uses LPE windows without projection, so k must equal hidden_size. "
                f"Got k={self.k}, hidden_size={self.hidden_size}."
            )

        self.n_items = train_dataset.n_items

        self.bucket_size_x = int(2 * (self.batch_size * self.seq_len) ** 0.5)
        self.n_buckets = int(2 * (self.batch_size * self.seq_len) ** 0.5)

        print("Calculated bucket size:", self.bucket_size_x)
        print("Calculated n buckets:", self.n_buckets)

        self.model = SASRecBackBone(
            self.n_items,
            self.hidden_size,
            self.dropout,
            self.seq_len,
            self.num_blocks,
            self.num_heads,
            manual_seed=37,
        )

        self.model.to(self.device)

        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999))
        lap_pos, chosen_indices = build_bipartite_lpe(
            train_dataset,
            self.k,
            chosen_idx=self.precomputed_chosen_idx,
            precomputed_eigvecs=self.precomputed_eigvecs,
        )
        self.chosen_eigvec_indices = chosen_indices

        pad_vec = torch.zeros(1, lap_pos.shape[1])
        lap_pos = torch.cat([lap_pos, pad_vec], dim=0)

        self.model.lap_pos_emb = nn.Embedding.from_pretrained(lap_pos, freeze=True).to(self.device)
        with torch.no_grad():
            item_norm = self.model.item_emb.weight.norm(dim=1).mean()
            lap_norm = self.model.lap_pos_emb.weight.norm(dim=1).mean()
            scale = item_norm / (lap_norm + 1e-8)
            self.model.lpe_weight.data = scale


SASRecEinvInter = SASRecInter
