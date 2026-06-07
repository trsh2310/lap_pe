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

def build_activation(name):
    name = name.lower()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    if name == "tanh":
        return nn.Tanh()
    if name in {"identity", "none"}:
        return nn.Identity()
    raise ValueError(f"Unknown activation: {name}")


def normalize_lap_proj_mode(mode):
    mode = str(mode).lower()
    if mode in {"mlp", "linear"}:
        return "mlp"
    if mode in {"identity", "none", "no_mlp"}:
        return "identity"
    raise ValueError(f"Unknown lap_proj_mode: {mode}")

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

def pad_inference_seq(user_items, maxlen, pad_token):
    seq = np.full(maxlen, pad_token, dtype=np.int32)
    if len(user_items) <= maxlen:
        if len(user_items) > 0:
            seq[-len(user_items):] = user_items
    else:
        seq = user_items[-maxlen:]
    return seq

import numpy as np

def load_laplacian_positional_encoding(path, k, n_items):
    path = str(path)
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.lib.npyio.NpzFile) and "eigvecs" in data:
        eigvecs = data["eigvecs"]
    elif isinstance(data, np.lib.npyio.NpzFile) and "pos_enc" in data:
        eigvecs = data["pos_enc"]
    elif isinstance(data, np.ndarray):
        eigvecs = data
    else:
        raise KeyError(f"{path} must be a .npy array or contain 'eigvecs'/'pos_enc' in .npz.")

    if eigvecs.shape[0] != n_items:
        raise ValueError(
            f"LPE cache row count mismatch: got {eigvecs.shape[0]}, expected {n_items}."
        )

    requested_k = int(k)
    pos_enc = eigvecs[:, :requested_k]
    computed_k = pos_enc.shape[1]
    if computed_k < requested_k:
        pos_enc = np.pad(pos_enc, ((0, 0), (0, requested_k - computed_k)), mode="constant")

    print(f"Loaded eigenvectors columns [0:{computed_k}] from {path}, padded to k={requested_k}")
    return torch.from_numpy(np.asarray(pos_enc, dtype=np.float32))


def build_laplacian_positional_encoding(train_dataset, k, n_items, eigvec_path=None):
    if eigvec_path is not None:
        return load_laplacian_positional_encoding(eigvec_path, k, n_items)

    from scipy.sparse import lil_matrix, diags, eye
    from scipy.sparse.linalg import eigsh

    requested_k = int(k)
    n_items = train_dataset.n_items
    n_users = train_dataset.n_users
    if requested_k <= 0:
        raise ValueError("k must be positive.")
    if n_items <= 1:
        raise ValueError("Laplacian positional encoding requires at least two items.")

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

    deg = np.array(A.sum(axis=1)).flatten()
    with np.errstate(divide='ignore'):
        deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)

    D_inv_sqrt = diags(deg_inv_sqrt)
    L = eye(n_items, format='csr') - D_inv_sqrt @ A @ D_inv_sqrt
    eigsh_k = min(requested_k, n_items - 1)
    eigenvalues, eigvecs = eigsh(L, k=eigsh_k, which='SM')

    idx = np.argsort(eigenvalues)
    eigvecs = eigvecs[:, idx]
    pos_enc = eigvecs[:, :requested_k]
    computed_k = pos_enc.shape[1]
    if pos_enc.shape[1] < requested_k:
        pad_width = requested_k - pos_enc.shape[1]
        pos_enc = np.pad(pos_enc, ((0, 0), (0, pad_width)), mode="constant")

    print(f"Using eigenvectors columns [0:{computed_k}], padded to k={requested_k}")

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
    if not is_train:
        batch["inference_seq"] = torch.LongTensor(np.asarray([
            pad_inference_seq(user["history"].tolist(), maxlen, pad_token)
            for user in x
        ]))

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
            k,
            lap_proj_hidden_mult=2.0,
            lap_proj_num_layers=2,
            lap_proj_dropout=None,
            lap_proj_activation="gelu",
            lap_proj_layer_norm=False,
            lap_proj_mode="mlp",
            use_input_lpe=True,
            use_output_lpe=True,
            use_absolute_pos_emb=False,
            use_item_emb_scale=True,
            learn_lpe_gates=True,
            lap_project_chunk_size=1024,
            item_chunk_size=2048,
            debug_lpe=False,
            manual_seed=37,
        ):
        super(SASRecBackBone, self).__init__()
        self.item_num = item_num
        self.pad_token = item_num
        self.lap_pos_emb = None 
        self.use_input_lpe = use_input_lpe
        self.use_output_lpe = use_output_lpe
        self.use_absolute_pos_emb = use_absolute_pos_emb
        self.use_item_emb_scale = use_item_emb_scale
        self.lap_project_chunk_size = int(lap_project_chunk_size)
        self.item_chunk_size = int(item_chunk_size)
        self.debug_lpe = debug_lpe
        self.lap_proj_mode = normalize_lap_proj_mode(lap_proj_mode)
        if self.lap_proj_mode == "identity" and int(k) != int(hidden_units):
            raise ValueError(
                "lap_proj_mode='identity' requires k == hidden_units; "
                f"got k={k}, hidden_units={hidden_units}."
            )

        self.item_emb = nn.Embedding(self.item_num+1, hidden_units, padding_idx=self.pad_token)
        self.pos_emb = nn.Embedding(maxlen, hidden_units)
        self.emb_dropout = nn.Dropout(p=dropout_rate)
        self.lpe_weight = nn.Parameter(torch.tensor(1.0), requires_grad=learn_lpe_gates)
        self.out_lpe_weight = nn.Parameter(torch.tensor(1.0), requires_grad=learn_lpe_gates)
        self.hidden_units = hidden_units

        self.attention_layernorms = nn.ModuleList() # to be Q for self-attention
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)

        if self.lap_proj_mode == "identity":
            self.lap_proj = nn.Identity()
        else:
            lap_hidden = int(hidden_units * lap_proj_hidden_mult)
            if lap_hidden <= 0:
                raise ValueError("lap_proj_hidden_mult must produce a positive hidden size.")
            if lap_proj_num_layers < 1:
                raise ValueError("lap_proj_num_layers must be at least 1.")
            lap_dropout = dropout_rate if lap_proj_dropout is None else lap_proj_dropout
            lap_layers = []
            if lap_proj_num_layers == 1:
                lap_layers.append(nn.Linear(k, hidden_units))
            else:
                lap_layers.append(nn.Linear(k, lap_hidden))
                lap_layers.append(build_activation(lap_proj_activation))
                if lap_proj_layer_norm:
                    lap_layers.append(nn.LayerNorm(lap_hidden))
                if lap_dropout > 0:
                    lap_layers.append(nn.Dropout(p=lap_dropout))

                for _ in range(lap_proj_num_layers - 2):
                    lap_layers.append(nn.Linear(lap_hidden, lap_hidden))
                    lap_layers.append(build_activation(lap_proj_activation))
                    if lap_proj_layer_norm:
                        lap_layers.append(nn.LayerNorm(lap_hidden))
                    if lap_dropout > 0:
                        lap_layers.append(nn.Dropout(p=lap_dropout))

                lap_layers.append(nn.Linear(lap_hidden, hidden_units))
            if lap_proj_layer_norm:
                lap_layers.append(nn.LayerNorm(hidden_units))
            self.lap_proj = nn.Sequential(*lap_layers)

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
        with torch.no_grad():
            self.item_emb.weight[self.pad_token].fill_(0.0)

    def zero_pad_item(self):
        with torch.no_grad():
            self.item_emb.weight[self.pad_token].fill_(0.0)

    def set_lap_pos_emb(self, lap_pos):
        # Keep the full eigenvector table on CPU; GPU projection is done in chunks.
        self.lap_pos_emb = lap_pos.detach().cpu().contiguous()
        self.lap_pos_emb.requires_grad_(False)

    def _project_lap_values(self, lap_pos, deterministic=False):
        if deterministic:
            was_training = self.lap_proj.training
            self.lap_proj.eval()
            try:
                return self.lap_proj(lap_pos)
            finally:
                self.lap_proj.train(was_training)
        return self.lap_proj(lap_pos)

    def project_lap(self, ids, deterministic=False):
        if self.lap_pos_emb is None:
            raise RuntimeError("lap_pos_emb is not initialized.")

        original_shape = ids.shape
        flat_ids = ids.reshape(-1)
        out = torch.empty(
            flat_ids.shape[0],
            self.hidden_units,
            device=ids.device,
            dtype=self.item_emb.weight.dtype,
        )
        chunk_size = max(1, self.lap_project_chunk_size)
        for start in range(0, flat_ids.shape[0], chunk_size):
            end = min(start + chunk_size, flat_ids.shape[0])
            chunk_ids = flat_ids[start:end]
            lap_pos = self.lap_pos_emb[chunk_ids.detach().cpu()].to(ids.device)
            projected = self._project_lap_values(lap_pos, deterministic=deterministic)
            projected = projected.masked_fill((chunk_ids == self.pad_token).unsqueeze(-1), 0.0)
            out[start:end] = projected
        return out.view(*original_shape, self.hidden_units)

    def get_item_representations(self, ids=None, deterministic_lap=True):
        if ids is None:
            ids = torch.arange(self.item_num + 1, device=self.item_emb.weight.device)
        item_repr = self.item_emb(ids)
        if self.use_output_lpe:
            lap_pos = self.project_lap(ids, deterministic=deterministic_lap)
            item_repr = item_repr + self.out_lpe_weight * lap_pos
        item_repr = item_repr.clone()
        item_repr = item_repr.masked_fill((ids == self.pad_token).unsqueeze(-1), 0.0)
        return item_repr

    def full_softmax_loss(self, x, y):
        pos_logits = torch.empty(x.shape[0], device=x.device, dtype=x.dtype)
        logsumexp = None
        chunk_size = max(1, self.item_chunk_size)
        for start in range(0, self.item_num + 1, chunk_size):
            end = min(start + chunk_size, self.item_num + 1)
            item_ids = torch.arange(start, end, device=x.device)
            item_repr = self.get_item_representations(item_ids, deterministic_lap=True)
            logits = x @ item_repr.T
            if start <= self.pad_token < end:
                logits[:, self.pad_token - start] = -1e9

            chunk_lse = torch.logsumexp(logits, dim=1)
            logsumexp = chunk_lse if logsumexp is None else torch.logaddexp(logsumexp, chunk_lse)

            mask = (y >= start) & (y < end)
            if mask.any():
                pos_logits[mask] = logits[mask, y[mask] - start]

        return (logsumexp - pos_logits).mean()

    def log2feats(self, log_seqs):
        device = log_seqs.device
        seqs = self.item_emb(log_seqs)
        if self.use_item_emb_scale:
            seqs *= self.item_emb.embedding_dim ** 0.5
        if self.use_absolute_pos_emb:
            positions = np.tile(np.arange(log_seqs.shape[1]), [log_seqs.shape[0], 1])
            seqs += self.pos_emb(torch.LongTensor(positions).to(device))
        if self.use_input_lpe:
            lap_pos = self.project_lap(log_seqs)
            if self.debug_lpe and torch.rand(1).item() < 0.001:
                print("lpe_weight:", self.lpe_weight.item())
                print("out_lpe_weight:", self.out_lpe_weight.item())
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

        logits = torch.empty(
            final_feat.shape[0],
            self.item_num + 1,
            device=final_feat.device,
            dtype=final_feat.dtype,
        )
        chunk_size = max(1, self.item_chunk_size)
        for start in range(0, self.item_num + 1, chunk_size):
            end = min(start + chunk_size, self.item_num + 1)
            item_ids = torch.arange(start, end, device=final_feat.device)
            item_embs = self.get_item_representations(item_ids, deterministic_lap=True)
            logits[:, start:end] = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        logits[:, self.pad_token] = -1e9
        return logits


class SASRecMLP(BaseModel):
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
        k = 1024,
        lap_proj_hidden_mult: float = 2.0,
        lap_proj_num_layers: int = 2,
        lap_proj_dropout = None,
        lap_proj_activation: str = "gelu",
        lap_proj_layer_norm: bool = False,
        lap_proj_mode: str = "mlp",
        use_input_lpe: bool = True,
        use_output_lpe: bool = True,
        use_absolute_pos_emb: bool = False,
        use_item_emb_scale: bool = True,
        input_lpe_gate = None,
        output_lpe_gate = None,
        learn_lpe_gates: bool = True,
        lpe_init_scale = None,
        lap_eigvec_path = None,
        lap_project_chunk_size: int = 1024,
        item_chunk_size: int = 2048,
        debug_lpe: bool = False,
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
        self.lap_proj_mode = normalize_lap_proj_mode(lap_proj_mode)
        self.k = k
        if k == None or self.lap_proj_mode == "identity":
            self.k = hidden_size
        self.lap_proj_hidden_mult = lap_proj_hidden_mult
        self.lap_proj_num_layers = lap_proj_num_layers
        self.lap_proj_dropout = lap_proj_dropout
        self.lap_proj_activation = lap_proj_activation
        self.lap_proj_layer_norm = lap_proj_layer_norm
        self.use_input_lpe = use_input_lpe
        self.use_output_lpe = use_output_lpe
        self.use_absolute_pos_emb = use_absolute_pos_emb
        self.use_item_emb_scale = use_item_emb_scale
        self.input_lpe_gate = input_lpe_gate
        self.output_lpe_gate = output_lpe_gate
        self.learn_lpe_gates = learn_lpe_gates
        self.lpe_init_scale = lpe_init_scale
        self.lap_eigvec_path = lap_eigvec_path
        self.lap_project_chunk_size = lap_project_chunk_size
        self.item_chunk_size = item_chunk_size
        self.debug_lpe = debug_lpe
        
    
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
            k = self.k,
            lap_proj_hidden_mult=self.lap_proj_hidden_mult,
            lap_proj_num_layers=self.lap_proj_num_layers,
            lap_proj_dropout=self.lap_proj_dropout,
            lap_proj_activation=self.lap_proj_activation,
            lap_proj_layer_norm=self.lap_proj_layer_norm,
            lap_proj_mode=self.lap_proj_mode,
            use_input_lpe=self.use_input_lpe,
            use_output_lpe=self.use_output_lpe,
            use_absolute_pos_emb=self.use_absolute_pos_emb,
            use_item_emb_scale=self.use_item_emb_scale,
            learn_lpe_gates=self.learn_lpe_gates,
            lap_project_chunk_size=self.lap_project_chunk_size,
            item_chunk_size=self.item_chunk_size,
            debug_lpe=self.debug_lpe,
            manual_seed=37
        )

        lap_pos = build_laplacian_positional_encoding(
            train_dataset,
            self.k,
            self.n_items,
            eigvec_path=self.lap_eigvec_path,
        )
        pad_vec = torch.zeros(1, lap_pos.shape[1])
        lap_pos = torch.cat([lap_pos, pad_vec], dim=0)

        self.model.set_lap_pos_emb(lap_pos)
        self.model.to(self.device)
        with torch.no_grad():
            ids = torch.arange(self.n_items, device=self.device)
            item_norm = self.model.item_emb.weight[:-1].norm(dim=1).mean()
            lap_norm = self.model.project_lap(ids, deterministic=True).norm(dim=1).mean()

            if self.lpe_init_scale is None:
                base_scale = item_norm / torch.clamp(lap_norm, min=1e-8)
            else:
                base_scale = torch.tensor(float(self.lpe_init_scale), device=self.device)

            input_gate = 1.0 if self.input_lpe_gate is None else float(self.input_lpe_gate)
            output_gate = 1.0 if self.output_lpe_gate is None else float(self.output_lpe_gate)
            self.model.lpe_weight.copy_(base_scale * input_gate)
            self.model.out_lpe_weight.copy_(base_scale * output_gate)
            self.model.zero_pad_item()

        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999))

        
    def _sasrec_forward(self, log_seqs, pos_seqs):

        emb = self.model.log2feats(log_seqs)


        hd = emb.shape[-1]

        x = emb.view(-1, hd)
        y = pos_seqs.view(-1)
        mask = y != self.model.pad_token
        x = x[mask]
        y = y[mask]
        return self.model.full_softmax_loss(x, y)
    
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
                self.model.zero_pad_item()

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
        if top_n > self.n_items:
            raise ValueError(f"top_n={top_n} is larger than the number of real items: {self.n_items}.")

        n_users = dataset.n_users
        holdout_array = None
        has_holdout = None
        try:
            holdout_array = dataset.get_holdout_array()
            has_holdout = np.zeros(n_users, dtype=bool)
            has_holdout[dataset.get_holdout_users()] = True
        except AssertionError:
            pass
        
        dataloader = DataLoader(
            dataset,
            batch_size,
            shuffle = False,
            collate_fn = lambda x: collate_fn_sasrec(x, self.seq_len, self.n_items, False)
        )
    
        recommendations = np.full((n_users, top_n), self.model.pad_token, dtype=np.int64)

        for batch in dataloader:
            inference_seq = batch["inference_seq"].to(self.device)
            if holdout_array is not None:
                user_ids_np = batch["user_id"].numpy()
                valid_mask_np = has_holdout[user_ids_np]
                if valid_mask_np.any():
                    valid_mask = torch.as_tensor(valid_mask_np, device=self.device)
                    holdouts = torch.as_tensor(
                        holdout_array[user_ids_np[valid_mask_np]],
                        device=self.device,
                    )
                    if (inference_seq[valid_mask] == holdouts[:, None]).any().item():
                        raise ValueError("Holdout item leakage detected in inference sequence.")
                    for row_idx, holdout in zip(np.where(valid_mask_np)[0], holdouts.cpu().numpy()):
                        if (batch["seen_history"][row_idx].numpy() == holdout).any():
                            raise ValueError("Holdout item leakage detected in seen history.")

            logits = self.model.score(
                seq=inference_seq,
            )

            logits[:, self.model.pad_token] = -10 ** 9

            if self.filter_seen:
                for i in range(len(batch["seen_history"])):
                    if batch["seen_history"][i].shape[0] > 0:
                        seen = batch["seen_history"][i]
                        seen = seen[(seen >= 0) & (seen < self.n_items)].to(self.device)
                        logits[i, seen] = -10 ** 9
    
            top_items = torch.topk(
                logits,
                k=top_n,
                dim=1
            ).indices
    
            for uid, recs in zip(batch["user_id"], top_items):
                if (recs == self.model.pad_token).any().item():
                    raise ValueError("Pad item appeared in recommendations.")
                recommendations[uid, :] = recs.cpu().numpy()
                
        return recommendations


SASRecEinv = SASRecMLP
    
