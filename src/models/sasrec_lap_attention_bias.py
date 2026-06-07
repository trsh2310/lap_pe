import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.sasrec import PointWiseFeedForward, SASRecModel, fix_torch_seed
from src.models.sasrec_rope_lap_raw import build_laplacian_positional_encoding


class LapBiasMultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        position_encoding="rope",
        lap_bias_mode="cosine",
        lap_bias_dim=32,
        lap_gate_init=-3.0,
        rope_base=10000.0,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")
        if position_encoding not in {"rope", "absolute", "lap"}:
            raise ValueError(f"Unknown position_encoding={position_encoding}.")
        if lap_bias_mode not in {"cosine", "bilinear"}:
            raise ValueError(f"Unknown lap_bias_mode={lap_bias_mode}.")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.position_encoding = position_encoding
        self.lap_bias_mode = lap_bias_mode
        self.lap_bias_dim = lap_bias_dim

        if self.position_encoding == "rope" and self.head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head_dim.")

        self.dropout = nn.Dropout(dropout)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        if self.position_encoding == "rope":
            inv_freq = 1.0 / (
                rope_base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)
        else:
            self.register_buffer("inv_freq", torch.empty(0), persistent=False)

        self.lap_attn_gate = nn.Parameter(torch.full((num_heads,), float(lap_gate_init)))
        if self.lap_bias_mode == "bilinear":
            self.lap_freq_weight = nn.Parameter(torch.ones(num_heads, lap_bias_dim))
        else:
            self.register_parameter("lap_freq_weight", None)

    def _shape(self, x):
        seq_len, batch_size, _ = x.shape
        x = x.view(seq_len, batch_size, self.num_heads, self.head_dim)
        return x.permute(1, 2, 0, 3)

    def _apply_rope(self, x):
        if self.position_encoding != "rope":
            return x

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

    def _lap_bias(self, lap_bias_emb):
        lap = F.normalize(lap_bias_emb, p=2, dim=-1)
        if self.lap_bias_mode == "cosine":
            bias = torch.einsum("btd,bsd->bts", lap, lap)
            bias = bias[:, None, :, :].expand(-1, self.num_heads, -1, -1)
        else:
            bias = torch.einsum("btk,hk,bsk->bhts", lap, self.lap_freq_weight, lap)

        gamma = torch.sigmoid(self.lap_attn_gate).view(1, self.num_heads, 1, 1)
        return gamma * bias

    def forward(self, query, key, value, lap_bias_emb, attn_mask=None):
        seq_len, batch_size, _ = query.shape

        q = self._apply_rope(self._shape(self.q_proj(query)))
        k = self._apply_rope(self._shape(self.k_proj(key)))
        v = self._shape(self.v_proj(value))

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn_scores = attn_scores + self._lap_bias(lap_bias_emb)

        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask[None, None, :, :], float("-inf"))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous()
        attn_output = attn_output.view(seq_len, batch_size, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights.mean(dim=1)


class SASRecLapAttentionBiasBackBone(nn.Module):
    def __init__(
        self,
        item_num,
        hidden_units,
        dropout_rate,
        maxlen,
        num_blocks,
        num_heads,
        position_encoding="rope",
        lap_bias_mode="cosine",
        lap_bias_dim=32,
        lap_gate_init=-3.0,
        rope_base=10000.0,
        manual_seed=37,
    ):
        super(SASRecLapAttentionBiasBackBone, self).__init__()
        if position_encoding not in {"rope", "absolute", "lap"}:
            raise ValueError(f"Unknown position_encoding={position_encoding}.")

        self.item_num = item_num
        self.pad_token = item_num
        self.position_encoding = position_encoding
        self.lap_bias_dim = lap_bias_dim
        self.lap_gate_init = lap_gate_init

        self.item_emb = nn.Embedding(self.item_num + 1, hidden_units, padding_idx=self.pad_token)
        self.pos_emb = (
            nn.Embedding(maxlen, hidden_units) if self.position_encoding == "absolute" else None
        )
        self.lap_bias_emb = None
        self.lap_pe_emb = None
        self.emb_dropout = nn.Dropout(p=dropout_rate)

        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)

        for _ in range(num_blocks):
            self.attention_layernorms.append(nn.LayerNorm(hidden_units, eps=1e-8))
            self.attention_layers.append(
                LapBiasMultiheadAttention(
                    hidden_units,
                    num_heads,
                    dropout=dropout_rate,
                    position_encoding=position_encoding,
                    lap_bias_mode=lap_bias_mode,
                    lap_bias_dim=lap_bias_dim,
                    lap_gate_init=lap_gate_init,
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
        for attention_layer in self.attention_layers:
            attention_layer.lap_attn_gate.data.fill_(self.lap_gate_init)
            if attention_layer.lap_freq_weight is not None:
                attention_layer.lap_freq_weight.data.fill_(1.0)

    def log2feats(self, log_seqs):
        if self.lap_bias_emb is None:
            raise RuntimeError("lap_bias_emb is not initialized.")
        if self.position_encoding == "lap" and self.lap_pe_emb is None:
            raise RuntimeError("lap_pe_emb is not initialized.")

        device = log_seqs.device
        seqs = self.item_emb(log_seqs)
        scale = self.item_emb.embedding_dim ** 0.5
        seqs *= scale

        if self.position_encoding == "absolute":
            positions = np.tile(np.arange(log_seqs.shape[1]), [log_seqs.shape[0], 1])
            seqs += self.pos_emb(torch.LongTensor(positions).to(device))
        elif self.position_encoding == "lap":
            seqs += self.lap_pe_emb(log_seqs) * scale

        seqs = self.emb_dropout(seqs)

        timeline_mask = log_seqs == self.pad_token
        seqs *= ~timeline_mask.unsqueeze(-1)
        lap_bias = self.lap_bias_emb(log_seqs)
        lap_bias *= ~timeline_mask.unsqueeze(-1)

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
                lap_bias,
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


class SASRecLapAttentionBias(SASRecModel):
    def __init__(
        self,
        *args,
        k=None,
        lap_bias_dim: int = 32,
        position_encoding: str = "rope",
        lap_bias_mode: str = "cosine",
        lap_gate_init: float = -3.0,
        lap_eigvec_path=None,
        rope_base: float = 10000.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.k = k
        self.lap_bias_dim = lap_bias_dim
        self.position_encoding = position_encoding
        self.lap_bias_mode = lap_bias_mode
        self.lap_gate_init = lap_gate_init
        self.lap_eigvec_path = lap_eigvec_path
        self.rope_base = rope_base

    @staticmethod
    def _pad_lap(lap_pos, width):
        if lap_pos.shape[1] > width:
            return lap_pos[:, :width]
        if lap_pos.shape[1] < width:
            pad_width = width - lap_pos.shape[1]
            return torch.cat(
                [lap_pos, torch.zeros(lap_pos.shape[0], pad_width, dtype=lap_pos.dtype)],
                dim=1,
            )
        return lap_pos

    def _post_init(self, train_dataset, val_dataset):
        self.n_items = train_dataset.n_items

        self.bucket_size_x = int(2 * (self.batch_size * self.seq_len) ** 0.5)
        self.n_buckets = int(2 * (self.batch_size * self.seq_len) ** 0.5)

        print("Calculated bucket size:", self.bucket_size_x)
        print("Calculated n buckets:", self.n_buckets)

        self.model = SASRecLapAttentionBiasBackBone(
            self.n_items,
            self.hidden_size,
            self.dropout,
            self.seq_len,
            self.num_blocks,
            self.num_heads,
            position_encoding=self.position_encoding,
            lap_bias_mode=self.lap_bias_mode,
            lap_bias_dim=self.lap_bias_dim,
            lap_gate_init=self.lap_gate_init,
            rope_base=self.rope_base,
            manual_seed=37,
        )

        load_k = max(self.hidden_size, self.lap_bias_dim, int(self.k or 0))
        lap_pos = build_laplacian_positional_encoding(
            train_dataset,
            load_k,
            self.n_items,
            eigvec_path=self.lap_eigvec_path,
        )

        lap_bias = self._pad_lap(lap_pos, self.lap_bias_dim)
        lap_bias = torch.cat(
            [lap_bias, torch.zeros(1, self.lap_bias_dim, dtype=lap_bias.dtype)],
            dim=0,
        )
        self.model.lap_bias_emb = nn.Embedding.from_pretrained(lap_bias, freeze=True)

        if self.position_encoding == "lap":
            lap_pe = self._pad_lap(lap_pos, self.hidden_size)
            lap_pe = F.normalize(lap_pe, p=2, dim=-1)
            lap_pe = torch.cat(
                [lap_pe, torch.zeros(1, self.hidden_size, dtype=lap_pe.dtype)],
                dim=0,
            )
            self.model.lap_pe_emb = nn.Embedding.from_pretrained(lap_pe, freeze=True)

        self.model.to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999))
