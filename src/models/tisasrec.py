import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from src.base import BaseModel
from src.metrics import NDCGMetric
from src.models.sasrec import PointWiseFeedForward, fix_torch_seed


def _minimum_positive_interval(timestamps):
    timestamps = np.asarray(timestamps, dtype=np.int64)
    if timestamps.size < 2:
        return 1

    positive_intervals = np.abs(np.diff(timestamps))
    positive_intervals = positive_intervals[positive_intervals > 0]
    return int(positive_intervals.min()) if positive_intervals.size > 0 else 1


def _time_interval_matrix(timestamps, max_time_interval, min_interval=1):
    timestamps = np.asarray(timestamps, dtype=np.int64)
    intervals = np.abs(timestamps[:, None] - timestamps[None, :])
    interval_ids = intervals // min_interval
    interval_ids = np.minimum(interval_ids, max_time_interval)
    return interval_ids.astype(np.int64)


def pad_seqs_with_timestamps(
    user_items,
    user_timestamps,
    maxlen,
    pad_token,
    max_time_interval,
    is_train,
):
    user_items = list(user_items)
    user_timestamps = list(user_timestamps)
    if len(user_items) != len(user_timestamps):
        raise ValueError("Item history and timestamp history must have the same length.")

    # TiSASRec uses one personalized time scale per user. It must not be
    # recalculated independently for the training and inference sequences.
    min_interval = _minimum_positive_interval(user_timestamps)
    first_timestamp = user_timestamps[0] if user_timestamps else 0
    seq = np.full(maxlen, pad_token, dtype=np.int64)
    pos = np.full(maxlen, pad_token, dtype=np.int64)
    seq_times = np.full(maxlen, first_timestamp, dtype=np.int64)
    inference_times = np.full(maxlen, first_timestamp, dtype=np.int64)

    if len(user_items) <= maxlen:
        if len(user_items) > 1:
            seq[-len(user_items) + 1:] = user_items[:-1]
            seq_times[-len(user_items) + 1:] = user_timestamps[:-1]
        if len(user_items) > 0:
            pos[-len(user_items):] = user_items
            inference_times[-len(user_items):] = user_timestamps
    else:
        seq = np.asarray(user_items[-maxlen - 1:-1], dtype=np.int64)
        pos = np.asarray(user_items[-maxlen:], dtype=np.int64)
        seq_times = np.asarray(user_timestamps[-maxlen - 1:-1], dtype=np.int64)
        inference_times = np.asarray(user_timestamps[-maxlen:], dtype=np.int64)

    if is_train:
        time_matrix = _time_interval_matrix(
            seq_times,
            max_time_interval,
            min_interval=min_interval,
        )
        return seq, pos, time_matrix

    inference_time_matrix = _time_interval_matrix(
        inference_times,
        max_time_interval,
        min_interval=min_interval,
    )
    return pos, pos, inference_time_matrix


def collate_fn_tisasrec(x, maxlen, pad_token, max_time_interval, is_train):
    seqs_batch = []
    pos_batch = []
    time_matrix_batch = []
    user_ids = []
    seen_history = []

    for user in x:
        seq, pos, time_matrix = pad_seqs_with_timestamps(
            user["history"].tolist(),
            user["timestamps"].tolist(),
            maxlen,
            pad_token,
            max_time_interval,
            is_train,
        )

        seqs_batch.append(seq)
        pos_batch.append(pos)
        time_matrix_batch.append(time_matrix)
        user_ids.append(user["user_id"])
        if not is_train:
            seen_history.append(user["history"])

    batch = {
        "seq": torch.LongTensor(np.asarray(seqs_batch)),
        "pos": torch.LongTensor(np.asarray(pos_batch)),
        "time_matrix": torch.LongTensor(np.asarray(time_matrix_batch)),
        "user_id": torch.LongTensor(np.asarray(user_ids)),
        "seen_history": seen_history,
    }

    return batch


class TimeIntervalAwareAttention(nn.Module):
    def __init__(self, hidden_units, num_heads, dropout_rate, relation_chunk_size=0):
        super().__init__()
        if hidden_units % num_heads != 0:
            raise ValueError("hidden_units must be divisible by num_heads.")
        if relation_chunk_size < 0:
            raise ValueError("relation_chunk_size must be non-negative.")

        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.head_dim = hidden_units // num_heads
        self.relation_chunk_size = relation_chunk_size

        self.q_proj = nn.Linear(hidden_units, hidden_units)
        self.k_proj = nn.Linear(hidden_units, hidden_units)
        self.v_proj = nn.Linear(hidden_units, hidden_units)
        self.out_proj = nn.Linear(hidden_units, hidden_units)

        self.dropout = nn.Dropout(dropout_rate)

    def _split_heads(self, x):
        seq_len, batch_size, _ = x.shape
        x = x.view(seq_len, batch_size, self.num_heads, self.head_dim)
        return x.permute(1, 2, 0, 3)

    def _split_position_heads(self, x):
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def _split_relation_heads(self, x):
        batch_size, seq_len, _, _ = x.shape
        x = x.view(batch_size, seq_len, seq_len, self.num_heads, self.head_dim)
        return x.permute(0, 3, 1, 2, 4)

    def _checkpoint_if_training(self, fn, *args):
        if torch.is_grad_enabled() and any(
            torch.is_tensor(arg) and arg.requires_grad for arg in args
        ):
            return checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def _time_scores_chunked(
        self,
        q,
        time_matrix,
        time_key_emb,
        relation_dropout,
    ):
        batch_size = q.shape[0]
        score_chunks = []

        for start in range(0, batch_size, self.relation_chunk_size):
            end = min(start + self.relation_chunk_size, batch_size)
            q_chunk = q[start:end]
            time_chunk = time_matrix[start:end]

            def compute_scores(q_part, time_part):
                rel_k = relation_dropout(time_key_emb(time_part))
                rel_k = self._split_relation_heads(rel_k)
                return torch.einsum("bhid,bhijd->bhij", q_part, rel_k)

            score_chunks.append(
                self._checkpoint_if_training(compute_scores, q_chunk, time_chunk)
            )

        return torch.cat(score_chunks, dim=0)

    def _time_output_chunked(
        self,
        attn_weights,
        time_matrix,
        time_value_emb,
        relation_dropout,
    ):
        batch_size = attn_weights.shape[0]
        output_chunks = []

        for start in range(0, batch_size, self.relation_chunk_size):
            end = min(start + self.relation_chunk_size, batch_size)
            attn_chunk = attn_weights[start:end]
            time_chunk = time_matrix[start:end]

            def compute_output(attn_part, time_part):
                rel_v = relation_dropout(time_value_emb(time_part))
                rel_v = self._split_relation_heads(rel_v)
                return torch.einsum("bhij,bhijd->bhid", attn_part, rel_v)

            output_chunks.append(
                self._checkpoint_if_training(compute_output, attn_chunk, time_chunk)
            )

        return torch.cat(output_chunks, dim=0)

    def forward(
        self,
        query,
        key,
        value,
        pos_key,
        pos_value,
        time_key=None,
        time_value=None,
        time_matrix=None,
        time_key_emb=None,
        time_value_emb=None,
        relation_dropout=None,
        attn_mask=None,
        key_padding_mask=None,
    ):
        seq_len, batch_size, _ = query.shape
        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key))
        v = self._split_heads(self.v_proj(value))

        pos_k = self._split_position_heads(pos_key)
        pos_v = self._split_position_heads(pos_value)

        content_scores = torch.matmul(q, k.transpose(-2, -1))
        if self.relation_chunk_size > 0:
            time_scores = self._time_scores_chunked(
                q,
                time_matrix,
                time_key_emb,
                relation_dropout,
            )
        else:
            rel_k = self._split_relation_heads(time_key)
            time_scores = torch.einsum("bhid,bhijd->bhij", q, rel_k)
        position_scores = torch.einsum("bhid,bhjd->bhij", q, pos_k)
        attn_scores = (content_scores + time_scores + position_scores) * (self.head_dim ** -0.5)

        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask[None, None, :, :], float("-inf"))
        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights)
        attn_weights = self.dropout(attn_weights)

        content_output = torch.matmul(attn_weights, v)
        if self.relation_chunk_size > 0:
            time_output = self._time_output_chunked(
                attn_weights,
                time_matrix,
                time_value_emb,
                relation_dropout,
            )
        else:
            rel_v = self._split_relation_heads(time_value)
            time_output = torch.einsum("bhij,bhijd->bhid", attn_weights, rel_v)
        position_output = torch.einsum("bhij,bhjd->bhid", attn_weights, pos_v)
        attn_output = content_output + time_output + position_output
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous()
        attn_output = attn_output.view(seq_len, batch_size, self.hidden_units)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights.mean(dim=1)


class TiSASRecBackBone(nn.Module):
    def __init__(
        self,
        item_num,
        hidden_units,
        dropout_rate,
        maxlen,
        num_blocks,
        num_heads,
        max_time_interval,
        relation_chunk_size=0,
        manual_seed=37,
    ):
        super(TiSASRecBackBone, self).__init__()
        self.item_num = item_num
        self.pad_token = item_num
        self.relation_chunk_size = relation_chunk_size
        self.item_emb = nn.Embedding(self.item_num + 1, hidden_units, padding_idx=self.pad_token)
        self.emb_dropout = nn.Dropout(p=dropout_rate)
        self.relation_dropout = nn.Dropout(p=dropout_rate)

        # As in the original TiSASRec, positional and temporal relation
        # embeddings are shared by all self-attention blocks.
        self.pos_key_emb = nn.Embedding(maxlen, hidden_units)
        self.pos_value_emb = nn.Embedding(maxlen, hidden_units)
        # Interval id 0 is a real relation (same timestamp / diagonal), not padding.
        self.time_key_emb = nn.Embedding(max_time_interval + 1, hidden_units)
        self.time_value_emb = nn.Embedding(max_time_interval + 1, hidden_units)

        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)

        for _ in range(num_blocks):
            self.attention_layernorms.append(nn.LayerNorm(hidden_units, eps=1e-8))
            self.attention_layers.append(
                TimeIntervalAwareAttention(
                    hidden_units,
                    num_heads,
                    dropout_rate,
                    relation_chunk_size=relation_chunk_size,
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
        self.item_emb.weight.data[self.pad_token].zero_()

    def log2feats(self, log_seqs, time_matrix):
        device = log_seqs.device
        seqs = self.item_emb(log_seqs)
        seqs *= self.item_emb.embedding_dim ** 0.5
        seqs = self.emb_dropout(seqs)

        timeline_mask = log_seqs == self.pad_token
        seqs *= ~timeline_mask.unsqueeze(-1)

        seq_len = seqs.shape[1]
        attention_mask = ~torch.tril(torch.full((seq_len, seq_len), True, device=device))

        # These tensors are deliberately created once and reused by every block,
        # matching the reference TiSASRec implementation.
        positions = torch.arange(seq_len, device=device)
        positions = positions.unsqueeze(0).expand(log_seqs.shape[0], -1)
        pos_key = self.relation_dropout(self.pos_key_emb(positions))
        pos_value = self.relation_dropout(self.pos_value_emb(positions))
        use_relation_chunks = self.relation_chunk_size > 0
        if use_relation_chunks:
            time_key = None
            time_value = None
        else:
            time_key = self.relation_dropout(self.time_key_emb(time_matrix))
            time_value = self.relation_dropout(self.time_value_emb(time_matrix))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            seqs = self.forward_layernorms[i](seqs)
            q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](
                q,
                seqs,
                seqs,
                pos_key,
                pos_value,
                time_key,
                time_value,
                time_matrix=time_matrix,
                time_key_emb=self.time_key_emb,
                time_value_emb=self.time_value_emb,
                relation_dropout=self.relation_dropout,
                attn_mask=attention_mask,
                key_padding_mask=timeline_mask,
            )

            seqs = q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)
        return log_feats

    def score(self, seq, time_matrix):
        log_feats = self.log2feats(seq, time_matrix)
        final_feat = log_feats[:, -1, :]
        item_embs = self.item_emb.weight
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        logits[:, self.pad_token] = -1e9
        return logits


class TiSASRec(BaseModel):
    def __init__(
        self,
        name: str = "tisasrec",
        hidden_size: int = 64,
        num_blocks: int = 1,
        num_heads: int = 1,
        dropout: float = 0.1,
        lr: float = 5e-3,
        device: str = "cuda",
        seq_len: int = 50,
        n_epochs: int = 2000,
        batch_size: int = 64,
        seed: int = 52,
        log_step: int = 200,
        patience_per_epoch: int = 1,
        max_patience: int = 50,
        val_top_n: int = 10,
        filter_seen: bool = True,
        max_time_interval: int = 256,
        relation_chunk_size: int = 0,
    ):
        BaseModel.__init__(self, name)
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.lr = lr
        self.device = device
        self.seq_len = seq_len
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.filter_seen = filter_seen
        self.seed = seed
        self.patience_per_epoch = patience_per_epoch
        self.max_patience = max_patience
        self.log_step = log_step
        self.val_top_n = val_top_n
        self.max_time_interval = max_time_interval
        self.relation_chunk_size = relation_chunk_size

    def _post_init(self, train_dataset, val_dataset):
        self.n_items = train_dataset.n_items
        self.model = TiSASRecBackBone(
            self.n_items,
            self.hidden_size,
            self.dropout,
            self.seq_len,
            self.num_blocks,
            self.num_heads,
            self.max_time_interval,
            relation_chunk_size=self.relation_chunk_size,
            manual_seed=self.seed,
        )
        self.model.to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999))

    def _tisasrec_forward(self, log_seqs, pos_seqs, time_matrix, return_hidden=False):
        emb = self.model.log2feats(log_seqs, time_matrix)
        hidden_dim = emb.shape[-1]
        x = emb.view(-1, hidden_dim)
        y = pos_seqs.view(-1)
        mask = y != self.model.pad_token
        x = x[mask]
        y = y[mask]
        logits = x @ self.model.item_emb.weight.T
        logits[:, self.model.pad_token] = -1e9
        loss = F.cross_entropy(logits, y)
        if return_hidden:
            return loss, emb
        return loss

    def suggest_additional_params(self):
        return {"n_epochs": self.n_epochs}

    def fit(self, train_dataset, val_dataset):
        self._post_init(train_dataset, val_dataset)
        train_dataloader = DataLoader(
            train_dataset,
            self.batch_size,
            shuffle=True,
            collate_fn=lambda x: collate_fn_tisasrec(
                x,
                self.seq_len,
                self.n_items,
                self.max_time_interval,
                True,
            ),
        )

        best_metric = -1
        patience_cnt = 0
        best_epoch = -1
        best_state_dict = None

        for i in trange(self.n_epochs):
            self.model.train()
            train_loss = []

            for index, batch in enumerate(tqdm(train_dataloader)):
                loss = self._tisasrec_forward(
                    batch["seq"].to(self.device),
                    batch["pos"].to(self.device),
                    batch["time_matrix"].to(self.device),
                )

                self.opt.zero_grad()
                train_loss.append(loss.item())
                loss.backward()
                self.opt.step()

                if (index + 1) % self.log_step == 0:
                    print(f"Mean train loss: {sum(train_loss[-self.log_step:]) / self.log_step}")

            print(f"Mean train loss: {sum(train_loss) / len(train_loss)}")

            if (i + 1) % self.patience_per_epoch == 0 and val_dataset is not None:
                holdout_users = val_dataset.get_holdout_users()
                predictions = self.predict(val_dataset, self.val_top_n)

                metric = NDCGMetric(self.val_top_n)
                metric_val = metric(
                    predictions[holdout_users, :],
                    val_dataset.get_holdout_array()[holdout_users],
                )
                print(f"Value of {metric.name}: {metric_val}")

                if best_metric < metric_val:
                    best_metric = metric_val
                    patience_cnt = 0
                    best_state_dict = {
                        k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
                    }
                    best_epoch = i + 1
                else:
                    patience_cnt += 1
                    if patience_cnt >= self.max_patience:
                        break

        if best_metric >= 0:
            self.model.load_state_dict(best_state_dict)
            self.n_epochs = best_epoch

    @torch.no_grad()
    def predict(self, dataset, top_n: int, batch_size: int = 128):
        self.model.eval()
        n_users = dataset.n_users

        dataloader = DataLoader(
            dataset,
            batch_size,
            shuffle=False,
            collate_fn=lambda x: collate_fn_tisasrec(
                x,
                self.seq_len,
                self.n_items,
                self.max_time_interval,
                False,
            ),
        )

        recommendations = np.zeros((n_users, top_n))

        for batch in dataloader:
            logits = self.model.score(
                seq=batch["seq"].to(self.device),
                time_matrix=batch["time_matrix"].to(self.device),
            )

            if self.filter_seen:
                for i in range(len(batch["seen_history"])):
                    if batch["seen_history"][i].shape[0] > 0:
                        logits[i][batch["seen_history"][i]] = -10 ** 9

            top_items = torch.topk(logits, k=top_n, dim=1).indices
            for uid, recs in zip(batch["user_id"], top_items):
                recommendations[uid, :] = recs.cpu().numpy()

        return recommendations
