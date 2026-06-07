from scipy.sparse.linalg import eigsh
from src.base import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import trange, tqdm
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from torch.utils.data import DataLoader
from src.metrics import NDCGMetric
from scipy.sparse import lil_matrix, diags, eye

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
    R = lil_matrix((n_users, n_items), dtype=np.float32)
    for user in train_dataset:
        uid = user["user_id"]
        for item in user["history"].tolist():
            if item < n_items:
                R[uid, item] = 1.0
    R = R.tocsr()

    A = (R.T @ R).tocsr()
    A.setdiag(0)
    A.eliminate_zeros()

    n_components, _ = connected_components(csgraph=A, directed=False)
    print(f"Connected components: {n_components}")

    deg = np.array(A.sum(axis=1)).flatten()
    with np.errstate(divide='ignore', invalid='ignore'):
        deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    D_inv_sqrt = diags(deg_inv_sqrt)
    L = eye(n_items, format='csr') - D_inv_sqrt @ A @ D_inv_sqrt
    eigvals, eigvecs = eigsh(L, k=k+1, which='SM')
    pos_enc = eigvecs[:, 1:k+1]
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
            lap_dim,
            manual_seed=37,
        ):
        super(SASRecBackBone, self).__init__()
        self.item_num = item_num
        self.pad_token = item_num
        self.lap_pos_emb = None 

        self.item_emb = nn.Embedding(self.item_num+1, hidden_units, padding_idx=self.pad_token)
        self.emb_dropout = nn.Dropout(p=dropout_rate)
        self.lpe_weight = nn.Parameter(torch.tensor(1.0))

        self.attention_layernorms = nn.ModuleList() # to be Q for self-attention
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
        self.lap_proj = nn.Sequential(
            nn.Linear(lap_dim, hidden_units),
            nn.GELU(),
            nn.Linear(hidden_units, hidden_units),
        )

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
        self.lpe_logs = {
            "lpe_weight": [],
            "item_norm": [],
            "lap_norm": []
        }

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
        pos_ids = log_seqs
        lap_pos = self.lap_proj(self.lap_pos_emb(pos_ids))
        item_norm_val = seqs.norm(dim=-1).mean().item()
        lap_norm_val = lap_pos.norm(dim=-1).mean().item()
        lpe_weight_val = self.lpe_weight.item()

        self.lpe_logs["lpe_weight"].append(lpe_weight_val)
        self.lpe_logs["item_norm"].append(item_norm_val)
        self.lpe_logs["lap_norm"].append(lap_norm_val)
        with open("lpe_logs.csv", "a") as f:
            f.write(f"{lpe_weight_val},{item_norm_val},{lap_norm_val}\n")
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
        name: str = "sasrec_einv_mink_mlp1024",
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
        bucket_size_y: int = 256,
        patience_per_epoch: int = 1,
        max_patience: int = 50,
        val_top_n: int = 10,
        filter_seen: bool = True,
        k: int = 1024,
        lap_eigvec_path = None,
        lap_eigvec_start: int = 0,
        lap_eigvec_stride: int = 1
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

        self.patience_per_epoch = patience_per_epoch
        self.max_patience = max_patience

        self.bucket_size_y = bucket_size_y

        self.log_step = log_step
        self.val_top_n = val_top_n
        self.k = k
        if k == None:
            self.k = 1024
        self.lap_eigvec_path = lap_eigvec_path
        self.lap_eigvec_start = lap_eigvec_start
        self.lap_eigvec_stride = lap_eigvec_stride
        
    
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
            self.k,
            manual_seed=37
        )

        lap_pos = build_laplacian_positional_encoding(
            train_dataset,
            self.k,
            self.n_items,
            eigvec_path=self.lap_eigvec_path,
            eigvec_start=self.lap_eigvec_start,
            eigvec_stride=self.lap_eigvec_stride,
        )
        pad_vec = torch.zeros(1, lap_pos.shape[1])
        lap_pos = torch.cat([lap_pos, pad_vec], dim=0)

        self.model.lap_pos_emb = nn.Embedding.from_pretrained(lap_pos,freeze=True)
        self.model.to(self.device)

        with torch.no_grad():
            item_norm = self.model.item_emb.weight.norm(dim=1).mean()
            lap_norm = self.model.lap_proj(self.model.lap_pos_emb.weight).norm(dim=1).mean()

            scale = item_norm / (lap_norm + 1e-8)
            self.model.lpe_weight.data = scale

        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999))

        
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
        return loss
    
    def suggest_additional_params(self):
        return {"n_epochs": self.n_epochs}
            
    
    def fit(self, train_dataset, val_dataset):
        """
        Fit the model to the dataset.
        """

        self._post_init(
            train_dataset,
            val_dataset
        )

        train_dataloader = DataLoader(
            train_dataset,
            self.batch_size,
            shuffle = True,
            collate_fn = lambda x: collate_fn_sasrec(x, self.seq_len, self.n_items, True)
        )

        best_metric = -1
        patience_cnt = 0
        best_epoch = -1
        best_state_dict = None

        for i in trange(self.n_epochs):
            self.model.train()

            train_loss = []

            index = 0

            for batch in tqdm(train_dataloader):

                loss = self._sasrec_forward(
                    batch["seq"].to(self.device),
                    batch["pos"].to(self.device),
                )

                self.opt.zero_grad()

                train_loss.append(loss.item())

                loss.backward()
                
                self.opt.step() 

                index += 1

                if (index + 1) % self.log_step == 0:
                    print(f"Mean train loss: {sum(train_loss[-self.log_step:]) / self.log_step}")    
            
            
            print(f"Mean train loss: {sum(train_loss) / len(train_loss)}")

            if (i + 1) % self.patience_per_epoch == 0 and val_dataset is not None:

                holdout_users = val_dataset.get_holdout_users()
        
                predictions = self.predict(
                    val_dataset,
                    self.val_top_n
                )

                

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
    
