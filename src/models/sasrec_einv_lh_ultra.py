from pathlib import Path

from src.base import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import trange, tqdm

from torch.utils.data import DataLoader
from src.metrics import NDCGMetric

def fix_torch_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def pad_seqs(user_items, maxlen, pad_token):
    seq = np.full(maxlen, pad_token, dtype=np.int32)
    pos = np.full(maxlen, pad_token, dtype=np.int32)
    neg = np.empty((maxlen, 1))

    if len(user_items) <= maxlen:
        if len(user_items) > 1:
            seq[-len(user_items) + 1:] = user_items[:-1]
        if len(user_items) > 0:
            pos[-len(user_items):] = user_items
    else:
        seq = user_items[-maxlen-1:-1]
        pos = user_items[-maxlen:]

    return seq, pos, neg

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

def count_connected_components(train_dataset, n_items):
    rows = []
    cols = []

    for user in train_dataset:
        items = user["history"].tolist()
        for i in range(len(items) - 1):
            u = items[i]
            v = items[i + 1]

            rows.append(u)
            cols.append(v)

            rows.append(v)  # делаем неориентированный граф
            cols.append(u)

    data = np.ones(len(rows))

    A = coo_matrix((data, (rows, cols)), shape=(n_items, n_items))

    n_components, labels = connected_components(csgraph=A, directed=False)

    return n_components, labels

def build_laplacian_positional_encoding(train_dataset, k, n_items):
    n_items = train_dataset.n_items
    n_components, labels = count_connected_components(train_dataset, n_items)
    print(n_components)
    A = np.zeros((n_items, n_items), dtype=np.float32)

    for user in train_dataset:
        items = user["history"].tolist()
        for i in range(len(items) - 1):
            u = items[i]
            v = items[i + 1]
            A[u, v] += 1
            A[v, u] += 1

    D = np.diag(A.sum(axis=1))

    L = D - A 
    eigvals, eigvecs = np.linalg.eigh(L)
    start = n_components

    k_low = k // 2
    k_high = k - k_low
    low = eigvecs[:, start:start + k_low]
    high = eigvecs[:, -k_high:]
    pos_enc = np.concatenate([low, high], axis=1)
    print(f"low: {k_low}, high: {k_high}, total: {pos_enc.shape[1]}")

    return torch.FloatTensor(pos_enc)

def collate_fn_sasrec(x, maxlen, pad_token, is_train):

    seqs_batch = []
    pos_batch = []
    user_ids = []
    seen_history = []

    for user in x:
        seq, pos, neg = pad_seqs(user["history"].tolist(), maxlen, pad_token)

        seqs_batch.append(seq)
        pos_batch.append(pos)
        user_ids.append(user["user_id"])
        if not is_train:
            seen_history.append(user["history"])

    batch = {
        "seq": torch.LongTensor(np.asarray(seqs_batch)),
        "pos": torch.LongTensor(np.asarray(pos_batch)),
        "user_id": torch.LongTensor(np.asarray(user_ids)),
        "seen_history": seen_history
    }

    return batch


class PointWiseFeedForward(torch.nn.Module):

    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.ff = nn.Sequential(
            torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1),
            torch.nn.Dropout(p=dropout_rate),
            torch.nn.ReLU(),
            torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1),
            torch.nn.Dropout(p=dropout_rate)
        )

    def forward(self, inputs):
        outputs = self.ff(
            inputs.transpose(-1,-2)
        )
        outputs = outputs.transpose(-1, -2)
        outputs += inputs
        return outputs

class SASRecBackBone(nn.Module):
    def __init__(
            self,
            item_num,
            hidden_units,
            dropout_rate,
            maxlen,
            num_blocks,
            num_heads,
            manual_seed=37
        ):
        super(SASRecBackBone, self).__init__()
        self.item_num = item_num
        self.pad_token = item_num
        self.lap_pos_emb = None 

        self.item_emb = nn.Embedding(self.item_num+1, hidden_units, padding_idx=self.pad_token)
        self.pos_emb = nn.Embedding(maxlen, hidden_units)
        self.emb_dropout = nn.Dropout(p=dropout_rate)
        self.lpe_weight = nn.Parameter(torch.tensor(1.0))

        self.attention_layernorms = nn.ModuleList() # to be Q for self-attention
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)

        for _ in range(num_blocks):
            new_attn_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)
            new_attn_layer =  nn.MultiheadAttention(
                hidden_units,num_heads,dropout_rate
            )
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(hidden_units, dropout_rate)
            self.forward_layers.append(new_fwd_layer)

        fix_torch_seed(manual_seed)
        self.initialize()

    def initialize(self):
        for _, param in self.named_parameters():
            try:
                torch.nn.init.xavier_uniform_(param.data)
            except:
                pass # just ignore those failed init layers

    def log2feats(self, log_seqs):
        device = log_seqs.device
        seqs = self.item_emb(log_seqs)
        #seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.arange(log_seqs.shape[1]), [log_seqs.shape[0], 1])
        #seqs += self.pos_emb(torch.LongTensor(positions).to(device))
        pos_ids = log_seqs
        lap_pos = self.lap_pos_emb(pos_ids)
        if torch.rand(1).item() < 0.001:
            print("lpe_weight:", self.lpe_weight.item())
            print("item mean norm:", seqs.norm(dim=-1).mean().item()) 
            print("lap mean norm:", lap_pos.norm(dim=-1).mean().item())
        seqs = seqs + self.lpe_weight * lap_pos
        
        seqs = self.emb_dropout(seqs)

        timeline_mask = log_seqs == self.pad_token

        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.full((tl, tl), True, device=device))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            seqs = self.forward_layernorms[i](seqs)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](
                Q, seqs, seqs, attn_mask=attention_mask
            )

            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

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
        final_feat = log_feats[:, -1, :] # only use last QKV classifier

        item_embs = self.item_emb.weight
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        return logits


class SASRecEinv(BaseModel):
    def __init__(
        self,
        name: str = "sasrec_einv",
        hidden_size: int = 64,
        num_blocks: int = 1,
        num_heads: int = 1,
        dropout: float = 0.1,
        lr: float = 5e-3,
        device: str = "cuda",
        seq_len: int = 50,
        n_epochs: int = 1000,
        batch_size: int = 64,
        seed: int = 52,
        log_step: int = 200,
        mix_x: bool = True,
        alpha: float = 2.0,
        beta: float = 1.0,
        bucket_size_y: int = 256,
        patience_per_epoch: int = 1,
        max_patience: int = 50,
        val_top_n: int = 10,
        filter_seen: bool = True,
        k = None,
        gamma: float = 0.1,
        k_neighbors: int = 20
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

        self.alpha = alpha
        self.beta = beta

        self.patience_per_epoch = patience_per_epoch
        self.max_patience = max_patience

        self.mix_x = mix_x

        self.bucket_size_y = bucket_size_y

        self.log_step = log_step
        self.val_top_n = val_top_n
        self.k = k
        if k == None:
            self.k = hidden_size

        self.gamma = gamma
        self.k_neighbors = k_neighbors
        self._cooc_cache_dir = Path("sasrec_cooc_cache")
        
    
    def _post_init(self, train_dataset, val_dataset):
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
            manual_seed=37
        )

        self.model.to(self.device)

        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999))
        lap_pos = build_laplacian_positional_encoding(train_dataset, self.k, self.n_items)
        if self.k != self.hidden_size:
            self.lap_linear = nn.Linear(self.k, self.hidden_size)
            lap_pos = self.lap_linear(lap_pos)
        pad_vec = torch.zeros(1, lap_pos.shape[1])
        lap_pos = torch.cat([lap_pos, pad_vec], dim=0)

        self.model.lap_pos_emb = nn.Embedding.from_pretrained(lap_pos,freeze=True).to(self.device)
        with torch.no_grad():
            item_norm = self.model.item_emb.weight.norm(dim=1).mean()
            lap_norm = self.model.lap_pos_emb.weight.norm(dim=1).mean()

            scale = item_norm / (lap_norm + 1e-8)
            self.model.lpe_weight.data = scale

        coo = train_dataset.get_coo_array()

        self.constraint_matrix, self.neighbor_matrix = self._compute_item_item_topk(
            coo,
            dataset_name=train_dataset.name,
        )
        self.constraint_matrix = self.constraint_matrix.to(self.device)
        self.neighbor_matrix = self.neighbor_matrix.to(self.device)

    def _sasrec_forward(self, log_seqs, pos_seqs):
        emb = self.model.log2feats(log_seqs)
        hd = emb.shape[-1]
        x = emb.view(-1, hd)
        y = pos_seqs.view(-1)
        mask = y != self.model.pad_token
        x = x[mask]
        y = y[mask]
        logits = x @ self.model.item_emb.weight.T
        logits[:, self.model.pad_token] = -1e9
        loss = F.cross_entropy(logits, y)
        return loss, x, y

    def suggest_additional_params(self):
        return {"n_epochs": self.n_epochs}
            

    def fit(self, train_dataset, val_dataset):
        self._post_init(train_dataset, val_dataset)

        train_dataloader = DataLoader(
            train_dataset,
            self.batch_size,
            shuffle=True,
            collate_fn=lambda x: collate_fn_sasrec(
                x, self.seq_len, self.n_items, True
            )
        )

        best_metric = -1
        patience_cnt = 0
        best_epoch = -1
        best_state_dict = None

        for epoch in trange(self.n_epochs):
            self.model.train()
            train_loss = []
            index = 0

            for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}"):

                seq = batch["seq"].to(self.device)
                pos = batch["pos"].to(self.device)

                loss_main, x_masked, y_masked = self._sasrec_forward(seq, pos)

                if hasattr(self, "neighbor_matrix"):
                    neighbor_embeds = self.model.item_emb(
                        self.neighbor_matrix[y_masked]
                    ) 

                    sim_scores = self.constraint_matrix[y_masked] 

                    logits = (x_masked.unsqueeze(1) * neighbor_embeds).sum(dim=-1)

                    L_I = -(sim_scores * F.logsigmoid(logits)).sum()

                    loss = loss_main + self.gamma * L_I
                else:
                    loss = loss_main

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                train_loss.append(loss.item())
                index += 1

                if (index + 1) % self.log_step == 0:
                    print(f"Mean train loss: {sum(train_loss[-self.log_step:]) / self.log_step}")

            print(f"Epoch {epoch+1} mean loss: {sum(train_loss) / len(train_loss)}")

            if (epoch + 1) % self.patience_per_epoch == 0 and val_dataset is not None:

                holdout_users = val_dataset.get_holdout_users()

                predictions = self.predict(
                    val_dataset,
                    self.val_top_n)

                metric = NDCGMetric(self.val_top_n)
                metric_val = metric(
                    predictions[holdout_users, :],
                    val_dataset.get_holdout_array()[holdout_users],)

                print(f"Value of {metric.name}: {metric_val}")

                if best_metric < metric_val:
                    best_metric = metric_val
                    patience_cnt = 0

                    best_state_dict = {
                        k: v.detach().cpu().clone()
                        for k, v in self.model.state_dict().items() }

                    best_epoch = epoch + 1
                else:
                    patience_cnt += 1
                    if patience_cnt >= self.max_patience:
                        print("Early stopping triggered")
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
            shuffle = False,
            collate_fn = lambda x: collate_fn_sasrec(x, self.seq_len, self.n_items, False)
        )
    
        recommendations = np.zeros((n_users, top_n))

        for batch in dataloader:

            logits = self.model.score(
                seq=batch["pos"].to(self.device),
            )

            if self.filter_seen:
                for i in range(len(batch["seen_history"])):
                    if batch["seen_history"][i].shape[0] > 0:
                        logits[i][batch["seen_history"][i]] = -10 ** 9
    
            top_items = torch.topk(
                logits,
                k=top_n,
                dim=1
            ).indices
    
            for uid, recs in zip(batch["user_id"], top_items):
                recommendations[uid, :] = recs.cpu().numpy()
                
        return recommendations
    
    def _compute_item_item_topk(self, coo_matrix, dataset_name: str):
        save_path = self._cooc_cache_dir / dataset_name
        save_path.mkdir(parents=True, exist_ok=True)
        omega_path = save_path / "omega_matrix.pt"
        nbr_path = save_path / "neighbor_matrix.pt"

        if omega_path.exists() and nbr_path.exists():
            omega_mat = torch.load(omega_path)
            nbr_mat = torch.load(nbr_path)
            return omega_mat, nbr_mat

        print("Computing item-item top-K...")

        A = coo_matrix.T.dot(coo_matrix)

        items_D = np.array(A.sum(axis=0)).reshape(-1) if not CUPY_AVAILABLE else cp.asnumpy(A.sum(axis=0)).reshape(-1)
        users_D = np.array(A.sum(axis=1)).reshape(-1) if not CUPY_AVAILABLE else cp.asnumpy(A.sum(axis=1)).reshape(-1)

        beta_uD = np.zeros_like(users_D, dtype=np.float32)
        mask = users_D > 0
        beta_uD[mask] = np.sqrt(users_D[mask] + 1) / users_D[mask]
        beta_iD = 1.0 / np.sqrt(items_D + 1)

        A_csr = csp.csr_matrix(
            (cp.asarray(A.data, dtype=cp.float32),
                cp.asarray(A.indices, dtype=cp.int32),
                cp.asarray(A.indptr, dtype=cp.int32)),
            shape=A.shape
        )
        beta_u_cp = cp.asarray(beta_uD)
        beta_i_cp = cp.asarray(beta_iD)
        K = self.k_neighbors
        n_items = self.n_items
        res_idx = cp.empty((n_items, K), dtype=cp.int32)
        res_val = cp.empty((n_items, K), dtype=cp.float32)

        for i in range(n_items):
            a_row = A_csr.getrow(i).toarray().ravel()
            row = (beta_u_cp[i] * beta_i_cp) * a_row  # weighted row
            if K < row.size:
                idx_part = cp.argpartition(row, -K)[-K:]
            else:
                idx_part = cp.arange(row.size, dtype=cp.int32)
            vals_part = row[idx_part]
            order = cp.argsort(vals_part)[::-1]
            topk_idx = idx_part[order][:K]
            topk_val = row[topk_idx]
            res_idx[i] = topk_idx.astype(cp.int32)
            res_val[i] = topk_val.astype(cp.float32)

        nbr_mat = torch.utils.dlpack.from_dlpack(res_idx.toDlpack()).long()
        omega_mat = torch.utils.dlpack.from_dlpack(res_val.toDlpack()).float()

        torch.save(omega_mat.cpu(), omega_path)
        torch.save(nbr_mat.cpu(), nbr_path)
        print("Saved item-item top-K cache to", save_path)
        return omega_mat, nbr_mat
    
    def cal_loss_I(self, h_t, pos_items):
        neighbor_embeds = self.model.item_emb(
            self.neighbor_matrix[pos_items]
        )  # (M, K, D)

        sim_scores = self.constraint_matrix[pos_items]  # (M, K)

        logits = (h_t.unsqueeze(1) * neighbor_embeds).sum(dim=-1)  # (M, K)

        return -(sim_scores * F.logsigmoid(logits)).sum()

    
