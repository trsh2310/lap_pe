import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.sasrec import PointWiseFeedForward, SASRecModel, fix_torch_seed
from src.models.sasrec_rope_lap_raw import build_laplacian_positional_encoding


class RotaryQKVAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, rope_base=10000.0):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head_dim.")

        self.dropout = nn.Dropout(dropout)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        inv_freq = 1.0 / (
            rope_base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _shape(self, x):
        seq_len, batch_size, _ = x.shape
        x = x.view(seq_len, batch_size, self.num_heads, self.head_dim)
        return x.permute(1, 2, 0, 3)

    def _apply_rope(self, x):
        seq_len = x.shape[-2]
        positions = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq)
        cos = freqs.cos()[None, None, :, :]
        sin = freqs.sin()[None, None, :, :]

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        x_rotated = torch.stack(
            (x_even * cos - x_odd * sin, x_even * sin + x_odd * cos),
            dim=-1,
        )
        return x_rotated.flatten(-2)

    def forward(self, query_source, key_source, value_source, attn_mask=None):
        seq_len, batch_size, _ = query_source.shape

        q = self._apply_rope(self._shape(self.q_proj(query_source)))
        k = self._apply_rope(self._shape(self.k_proj(key_source)))
        v = self._shape(self.v_proj(value_source))

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask[None, None, :, :], float("-inf"))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous()
        attn_output = attn_output.view(seq_len, batch_size, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights.mean(dim=1)


class SASRecRoPELapProjectionBackBone(nn.Module):
    VALID_MODES = {"v", "kv", "qk"}

    def __init__(
        self,
        item_num,
        hidden_units,
        dropout_rate,
        maxlen,
        num_blocks,
        num_heads,
        lap_projection_mode,
        lap_alpha=0.01,
        learnable_lap_gate=False,
        lap_gate_init=-3.0,
        lap_gate_activation="sigmoid",
        lap_gate_init_from_norms=False,
        rope_base=10000.0,
        manual_seed=37,
    ):
        super(SASRecRoPELapProjectionBackBone, self).__init__()
        if lap_projection_mode not in self.VALID_MODES:
            raise ValueError(f"Unknown lap_projection_mode={lap_projection_mode}.")

        self.item_num = item_num
        self.pad_token = item_num
        self.lap_projection_mode = lap_projection_mode
        self.lap_alpha = lap_alpha
        self.learnable_lap_gate = learnable_lap_gate
        self.lap_gate_activation = lap_gate_activation
        self.lap_gate_init_from_norms = lap_gate_init_from_norms
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
                RotaryQKVAttention(
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
        if (
            not self.learnable_lap_gate
            or not self.lap_gate_init_from_norms
            or self.lap_pos_emb is None
        ):
            return

        item_norm = self.item_emb.weight[:-1].norm(dim=1).mean()
        lap_norm = self.lap_pos_emb.weight[:-1].norm(dim=1).mean()
        scale = item_norm / (lap_norm + 1e-8)
        if self.lap_gate_activation == "raw":
            self.lap_gate.data.copy_(scale)
            print(f"Initialized raw Lap gate from norms: gate={scale.item():.6f}")
        else:
            target_weight = scale.clamp(min=1e-4, max=1.0 - 1e-4)
            self.lap_gate.data.copy_(torch.logit(target_weight))
            print(
                "Initialized sigmoid Lap gate from norms: "
                f"sigmoid(gate)={target_weight.item():.6f}"
            )

    def _qkv_sources(self, base, lap):
        lap_augmented = base + self._lap_weight() * lap
        if self.lap_projection_mode == "v":
            return base, base, lap_augmented
        if self.lap_projection_mode == "kv":
            return base, lap_augmented, lap_augmented
        if self.lap_projection_mode == "qk":
            return lap_augmented, lap_augmented, base
        raise RuntimeError(f"Unexpected lap_projection_mode={self.lap_projection_mode}.")

    def log2feats(self, log_seqs):
        if self.lap_pos_emb is None:
            raise RuntimeError("lap_pos_emb is not initialized.")

        device = log_seqs.device
        seqs = self.item_emb(log_seqs)
        scale = self.item_emb.embedding_dim ** 0.5
        seqs *= scale
        seqs = self.emb_dropout(seqs)

        lap = self.lap_pos_emb(log_seqs)
        lap = F.normalize(lap, p=2, dim=-1) * scale

        timeline_mask = log_seqs == self.pad_token
        seqs *= ~timeline_mask.unsqueeze(-1)
        lap *= ~timeline_mask.unsqueeze(-1)

        seq_len = seqs.shape[1]
        attention_mask = ~torch.tril(torch.full((seq_len, seq_len), True, device=device))

        lap = torch.transpose(lap, 0, 1)
        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            seqs = self.forward_layernorms[i](seqs)
            base = self.attention_layernorms[i](seqs)
            q_source, k_source, v_source = self._qkv_sources(base, lap)
            mha_outputs, _ = self.attention_layers[i](
                q_source,
                k_source,
                v_source,
                attn_mask=attention_mask,
            )

            seqs = base + mha_outputs
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


class SASRecRoPELapProjection(SASRecModel):
    def __init__(
        self,
        *args,
        k=None,
        lap_projection_mode: str = "v",
        lap_alpha: float = 0.01,
        learnable_lap_gate: bool = False,
        lap_gate_init: float = -3.0,
        lap_gate_activation: str = "sigmoid",
        lap_gate_init_from_norms: bool = False,
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
        self.lap_projection_mode = lap_projection_mode
        self.lap_alpha = lap_alpha
        self.learnable_lap_gate = learnable_lap_gate
        self.lap_gate_init = lap_gate_init
        self.lap_gate_activation = lap_gate_activation
        self.lap_gate_init_from_norms = lap_gate_init_from_norms
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

        self.model = SASRecRoPELapProjectionBackBone(
            self.n_items,
            self.hidden_size,
            self.dropout,
            self.seq_len,
            self.num_blocks,
            self.num_heads,
            self.lap_projection_mode,
            lap_alpha=self.lap_alpha,
            learnable_lap_gate=self.learnable_lap_gate,
            lap_gate_init=self.lap_gate_init,
            lap_gate_activation=self.lap_gate_activation,
            lap_gate_init_from_norms=self.lap_gate_init_from_norms,
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


class SASRecRoPELapV(SASRecRoPELapProjection):
    def __init__(self, *args, **kwargs):
        kwargs["lap_projection_mode"] = "v"
        super().__init__(*args, **kwargs)


class SASRecRoPELapKV(SASRecRoPELapProjection):
    def __init__(self, *args, **kwargs):
        kwargs["lap_projection_mode"] = "kv"
        super().__init__(*args, **kwargs)


class SASRecRoPELapQK(SASRecRoPELapProjection):
    def __init__(self, *args, **kwargs):
        kwargs["lap_projection_mode"] = "qk"
        super().__init__(*args, **kwargs)
