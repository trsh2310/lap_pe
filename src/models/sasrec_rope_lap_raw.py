from scipy.sparse import diags, eye, lil_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.sasrec import PointWiseFeedForward, SASRecModel, fix_torch_seed
from src.models.sasrec_rope import RotaryMultiheadAttention


def resolve_lap_eigvec_path(path):
    path = Path(path).expanduser()
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [path]
    if not path.is_absolute():
        candidates.extend(
            [
                Path.cwd() / path,
                Path.cwd().parent / "RecSys-BTL" / path.name,
                Path.cwd().parent / path,
                repo_root / path,
                repo_root.parent / "RecSys-BTL" / path.name,
                repo_root.parent / path,
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    candidates_text = "\n".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"Laplacian eigenvector file was not found. Checked:\n{candidates_text}"
    )


def load_laplacian_positional_encoding(path, k, n_items, eigvec_start=0, eigvec_stride=1):
    resolved_path = resolve_lap_eigvec_path(path)
    data = np.load(resolved_path, allow_pickle=True)
    if isinstance(data, np.lib.npyio.NpzFile) and "eigvecs" in data:
        eigvecs = data["eigvecs"]
    elif isinstance(data, np.lib.npyio.NpzFile) and "pos_enc" in data:
        eigvecs = data["pos_enc"]
    elif isinstance(data, np.ndarray):
        eigvecs = data
    else:
        raise KeyError(
            f"{resolved_path} must be a .npy array or contain 'eigvecs'/'pos_enc' in .npz."
        )

    if eigvecs.shape[0] != n_items:
        raise ValueError(
            f"LPE cache row count mismatch: got {eigvecs.shape[0]}, expected {n_items}."
        )

    requested_k = int(k)
    eigvec_start = int(eigvec_start)
    eigvec_stride = int(eigvec_stride)
    if eigvec_start < 0:
        raise ValueError("eigvec_start must be non-negative.")
    if eigvec_stride <= 0:
        raise ValueError("eigvec_stride must be positive.")

    indices = eigvec_start + np.arange(requested_k) * eigvec_stride
    valid_indices = indices[indices < eigvecs.shape[1]]
    pos_enc = eigvecs[:, valid_indices]
    computed_k = pos_enc.shape[1]
    if computed_k < requested_k:
        pos_enc = np.pad(pos_enc, ((0, 0), (0, requested_k - computed_k)), mode="constant")

    print(
        f"Loaded {computed_k} eigenvector columns from {resolved_path} "
        f"(start={eigvec_start}, stride={eigvec_stride}), padded to k={requested_k}"
    )
    return torch.from_numpy(np.asarray(pos_enc, dtype=np.float32))


def build_laplacian_positional_encoding(
    train_dataset,
    k,
    n_items,
    eigvec_path=None,
    eigvec_start=0,
    eigvec_stride=1,
):
    if eigvec_path is not None:
        return load_laplacian_positional_encoding(
            eigvec_path,
            k,
            n_items,
            eigvec_start=eigvec_start,
            eigvec_stride=eigvec_stride,
        )

    n_items = train_dataset.n_items
    n_users = train_dataset.n_users
    requested_k = min(k, max(1, n_items - 2))
    if requested_k <= 0:
        raise ValueError("Laplacian positional encoding requires at least two items.")

    interactions = lil_matrix((n_users, n_items), dtype="float32")
    for user in train_dataset:
        user_id = user["user_id"]
        for item in user["history"].tolist():
            if item < n_items:
                interactions[user_id, item] = 1.0
    interactions = interactions.tocsr()

    adjacency = (interactions.T @ interactions).tocsr()
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()

    n_components, _ = connected_components(csgraph=adjacency, directed=False)
    print(f"Connected components: {n_components}")

    degree = adjacency.sum(axis=1).A1
    deg_inv_sqrt = np.zeros(n_items, dtype=np.float32)
    nonzero_degree = degree > 0
    deg_inv_sqrt[nonzero_degree] = 1.0 / np.sqrt(degree[nonzero_degree])

    laplacian = eye(n_items, format="csr") - diags(deg_inv_sqrt) @ adjacency @ diags(deg_inv_sqrt)
    eigsh_k = min(requested_k + 1, n_items - 1)
    _, eigvecs = eigsh(laplacian, k=eigsh_k, which="SM")

    pos_enc = torch.as_tensor(eigvecs[:, 1:], dtype=torch.float32)
    if pos_enc.shape[1] < k:
        pad_width = k - pos_enc.shape[1]
        pos_enc = torch.cat(
            [
                pos_enc,
                torch.zeros(n_items, pad_width, dtype=torch.float32),
            ],
            dim=1,
        )
        return pos_enc
    return pos_enc[:, :k]


class SASRecRoPELapRawBackBone(nn.Module):
    def __init__(
        self,
        item_num,
        hidden_units,
        dropout_rate,
        maxlen,
        num_blocks,
        num_heads,
        lap_alpha=0.05,
        learnable_lap_gate=False,
        lap_gate_init=-3.0,
        lap_gate_activation="sigmoid",
        lap_gate_target_ratio=0.05,
        rope_base=10000.0,
        manual_seed=37,
    ):
        super(SASRecRoPELapRawBackBone, self).__init__()
        self.item_num = item_num
        self.pad_token = item_num
        self.lap_alpha = lap_alpha
        self.learnable_lap_gate = learnable_lap_gate
        self.lap_gate_activation = lap_gate_activation
        self.lap_gate_target_ratio = lap_gate_target_ratio
        if self.lap_gate_activation not in {"sigmoid", "raw"}:
            raise ValueError(f"Unknown lap_gate_activation={self.lap_gate_activation}.")

        self.item_emb = nn.Embedding(self.item_num + 1, hidden_units, padding_idx=self.pad_token)
        self.lap_pos_emb = None
        if learnable_lap_gate:
            self.lap_gate = nn.Parameter(torch.tensor(float(lap_gate_init)))
        else:
            self.register_parameter("lap_gate", None)

        self.emb_dropout = nn.Dropout(p=dropout_rate)
        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)

        for _ in range(num_blocks):
            self.attention_layernorms.append(nn.LayerNorm(hidden_units, eps=1e-8))
            self.attention_layers.append(
                RotaryMultiheadAttention(
                    hidden_units,
                    num_heads,
                    dropout=dropout_rate,
                    rope_base=rope_base,
                )
            )
            self.forward_layernorms.append(nn.LayerNorm(hidden_units, eps=1e-8))
            self.forward_layers.append(PointWiseFeedForward(hidden_units, dropout_rate))

        fix_torch_seed(manual_seed)
        self.initialize()

    def initialize(self):
        for _, param in self.named_parameters():
            try:
                torch.nn.init.xavier_uniform_(param.data)
            except Exception:
                pass

    def _lap_weight(self):
        if self.learnable_lap_gate:
            if self.lap_gate_activation == "raw":
                return self.lap_gate
            return torch.sigmoid(self.lap_gate)
        return self.lap_alpha

    @torch.no_grad()
    def initialize_lap_gate_from_norms(self):
        if not self.learnable_lap_gate or self.lap_pos_emb is None:
            return
        if self.lap_gate_activation != "sigmoid":
            print(
                "Initialized learnable Lap gate: "
                f"raw gate={self.lap_gate.detach().cpu().item():.6f}"
            )
            return

        scale = self.item_emb.embedding_dim ** 0.5
        item_repr = self.item_emb.weight[:-1] * scale
        lap_repr = F.normalize(self.lap_pos_emb.weight[:-1], p=2, dim=-1) * scale
        item_norm = item_repr.norm(dim=1).mean()
        lap_norm = lap_repr.norm(dim=1).mean()
        target_weight = self.lap_gate_target_ratio * item_norm / (lap_norm + 1e-8)
        target_weight = target_weight.clamp(min=1e-4, max=1.0 - 1e-4)
        self.lap_gate.data.copy_(torch.logit(target_weight))
        print(
            "Initialized learnable Lap gate: "
            f"sigmoid(gate)={target_weight.item():.6f}, "
            f"target_ratio={self.lap_gate_target_ratio}"
        )

    def log2feats(self, log_seqs):
        if self.lap_pos_emb is None:
            raise RuntimeError("lap_pos_emb is not initialized.")

        device = log_seqs.device
        seqs = self.item_emb(log_seqs)
        scale = self.item_emb.embedding_dim ** 0.5
        seqs *= scale

        lap = self.lap_pos_emb(log_seqs)
        lap = F.normalize(lap, p=2, dim=-1) * scale
        seqs = seqs + self._lap_weight() * lap
        seqs = self.emb_dropout(seqs)

        timeline_mask = log_seqs == self.pad_token
        seqs *= ~timeline_mask.unsqueeze(-1)

        seq_len = seqs.shape[1]
        attention_mask = ~torch.tril(torch.full((seq_len, seq_len), True, device=device))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            seqs = self.forward_layernorms[i](seqs)
            q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](
                q,
                seqs,
                seqs,
                attn_mask=attention_mask,
            )

            seqs = q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)

        return self.last_layernorm(seqs)

    def forward(self, log_seqs, pos_seqs, neg_seqs):
        log_feats = self.log2feats(log_seqs)
        pos_embs = self.item_emb(pos_seqs)
        neg_embs = self.item_emb(neg_seqs)

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats[:, None, :, :] * neg_embs).sum(dim=-1)
        return pos_logits, neg_logits

    def score(self, seq):
        log_feats = self.log2feats(seq)
        final_feat = log_feats[:, -1, :]
        item_embs = self.item_emb.weight
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        return logits


class SASRecRoPELapRaw(SASRecModel):
    def __init__(
        self,
        *args,
        k=None,
        lap_alpha: float = 0.05,
        learnable_lap_gate: bool = False,
        lap_gate_init: float = -3.0,
        lap_gate_activation: str = "sigmoid",
        lap_gate_target_ratio: float = 0.05,
        lap_eigvec_path=None,
        lap_eigvec_start: int = 0,
        lap_eigvec_stride: int = 1,
        rope_base: float = 10000.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.k = k
        if self.k is None:
            self.k = self.hidden_size
        self.lap_alpha = lap_alpha
        self.learnable_lap_gate = learnable_lap_gate
        self.lap_gate_init = lap_gate_init
        self.lap_gate_activation = lap_gate_activation
        self.lap_gate_target_ratio = lap_gate_target_ratio
        self.lap_eigvec_path = lap_eigvec_path
        self.lap_eigvec_start = lap_eigvec_start
        self.lap_eigvec_stride = lap_eigvec_stride
        self.rope_base = rope_base

    def _post_init(self, train_dataset, val_dataset):
        self.n_items = train_dataset.n_items

        self.bucket_size_x = int(2 * (self.batch_size * self.seq_len) ** 0.5)
        self.n_buckets = int(2 * (self.batch_size * self.seq_len) ** 0.5)

        print("Calculated bucket size:", self.bucket_size_x)
        print("Calculated n buckets:", self.n_buckets)

        self.model = SASRecRoPELapRawBackBone(
            self.n_items,
            self.hidden_size,
            self.dropout,
            self.seq_len,
            self.num_blocks,
            self.num_heads,
            lap_alpha=self.lap_alpha,
            learnable_lap_gate=self.learnable_lap_gate,
            lap_gate_init=self.lap_gate_init,
            lap_gate_activation=self.lap_gate_activation,
            lap_gate_target_ratio=self.lap_gate_target_ratio,
            rope_base=self.rope_base,
            manual_seed=37,
        )

        lap_pos = build_laplacian_positional_encoding(
            train_dataset,
            self.k,
            self.n_items,
            eigvec_path=self.lap_eigvec_path,
            eigvec_start=self.lap_eigvec_start,
            eigvec_stride=self.lap_eigvec_stride,
        )
        if lap_pos.shape[1] > self.hidden_size:
            lap_pos = lap_pos[:, :self.hidden_size]
        elif lap_pos.shape[1] < self.hidden_size:
            pad_width = self.hidden_size - lap_pos.shape[1]
            lap_pos = torch.cat(
                [lap_pos, torch.zeros(lap_pos.shape[0], pad_width, dtype=lap_pos.dtype)],
                dim=1,
            )
        pad_vec = torch.zeros(1, self.hidden_size, dtype=lap_pos.dtype)
        lap_pos = torch.cat([lap_pos, pad_vec], dim=0)
        self.model.lap_pos_emb = nn.Embedding.from_pretrained(lap_pos, freeze=True)
        self.model.initialize_lap_gate_from_norms()

        self.model.to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999))
