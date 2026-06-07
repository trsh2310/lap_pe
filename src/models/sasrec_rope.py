import torch
import torch.nn as nn

from src.models.sasrec import PointWiseFeedForward, SASRecModel, fix_torch_seed


class RotaryMultiheadAttention(nn.Module):
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

    def forward(self, query, key, value, attn_mask=None):
        seq_len, batch_size, _ = query.shape

        q = self._apply_rope(self._shape(self.q_proj(query)))
        k = self._apply_rope(self._shape(self.k_proj(key)))
        v = self._shape(self.v_proj(value))

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


class SASRecRoPEBackBone(nn.Module):
    def __init__(
        self,
        item_num,
        hidden_units,
        dropout_rate,
        maxlen,
        num_blocks,
        num_heads,
        rope_base=10000.0,
        manual_seed=37,
    ):
        super(SASRecRoPEBackBone, self).__init__()
        self.item_num = item_num
        self.pad_token = item_num

        self.item_emb = nn.Embedding(self.item_num + 1, hidden_units, padding_idx=self.pad_token)
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

    def log2feats(self, log_seqs):
        device = log_seqs.device
        seqs = self.item_emb(log_seqs)
        seqs *= self.item_emb.embedding_dim ** 0.5
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

        log_feats = self.last_layernorm(seqs)
        return log_feats

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


class SASRecRoPE(SASRecModel):
    def __init__(self, *args, rope_base: float = 10000.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.rope_base = rope_base

    def _post_init(self, train_dataset, val_dataset):
        self.n_items = train_dataset.n_items

        self.bucket_size_x = int(2 * (self.batch_size * self.seq_len) ** 0.5)
        self.n_buckets = int(2 * (self.batch_size * self.seq_len) ** 0.5)

        print("Calculated bucket size:", self.bucket_size_x)
        print("Calculated n buckets:", self.n_buckets)

        self.model = SASRecRoPEBackBone(
            self.n_items,
            self.hidden_size,
            self.dropout,
            self.seq_len,
            self.num_blocks,
            self.num_heads,
            rope_base=self.rope_base,
            manual_seed=37,
        )

        self.model.to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999))
