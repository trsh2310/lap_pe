# Enriching Sequential Recommendation with Graph Laplacian Positional Embeddings

This repository contains `SASRec-Einv`, a SASRec model with graph-derived positional embeddings, together with the baselines and experimental pipeline used for its evaluation.

## Method

`SASRec-Einv` builds an item co-occurrence graph from the training interactions and computes eigenvectors of its symmetric normalized Laplacian. These frozen graph-derived embeddings replace the standard ordinal positional embeddings in SASRec. The backbone and training objective remain unchanged.

The repository contains the following principal variants:

- `SASRec`: the standard sequential recommendation baseline with learnable absolute positional embeddings;
- `SASRec-Einv`: SASRec with frozen Laplacian eigenvectors;
- `SASRec-RoPE`: SASRec with rotary positional embeddings;
- `TiSASRec`: a time-interval-aware sequential recommendation baseline.

Evaluation includes NDCG, Recall, and Coverage at cutoffs 10 and 100.

## Running an experiment

Run `run_model.py` from the project root and specify a configuration from `configs/` without the `.yaml` extension:

```bash
python3 run_model.py -cn <your_config>
```

Ready-to-use dataset configurations contain the best SASRec hyperparameters found during tuning and apply them to the corresponding model variants:

| Dataset | SASRec | SASRec-Einv | SASRec-RoPE | TiSASRec |
|---|---|---|---|---|
| Amazon Beauty | `sasrec_beauty` | `sasrec_einv_beauty` | `sasrec_rope_beauty` | `tisasrec_beauty` |
| Amazon Clothing, Shoes and Jewelry | `sasrec_clothing` | `sasrec_einv_clothing` | `sasrec_rope_clothing` | `tisasrec_clothing` |
| Amazon Sports and Outdoors | `sasrec_sports` | `sasrec_einv_sports` | `sasrec_rope_sports` | `tisasrec_sports` |
| Yelp | `sasrec_yelp` | `sasrec_einv_yelp` | `sasrec_rope_yelp` | `tisasrec_yelp` |

Training uses validation-based early stopping. Results are stored below `outputs/`.

## Hyperparameter optimization

Use `run_optuna.py` to tune model hyperparameters:

```bash
python3 run_optuna.py \
  --config_name <config_name> \
  --dataset <dataset_name> \
  --optuna_params <optuna_config> \
  --experiment_name <experiment_name>
```

The search space is defined in `configs/optuna_params/`. Optimization uses validation NDCG@10, and results are stored below `optuna_outputs/<experiment_name>/`.

## Datasets

The experiments use four public sequential-recommendation benchmarks:

- Amazon 2014 Beauty;
- Amazon 2014 Clothing, Shoes and Jewelry;
- Amazon 2014 Sports and Outdoors;
- [Yelp 2018](https://deephypergraph.readthedocs.io/en/latest/generated/dhg.data.Yelp2018.html), provided through the DeepHypergraph dataset collection.

Dataset descriptors are stored in `configs/dataset/`. Download all configured datasets with:

```bash
python3 offline_download.py
```

Processed datasets are stored as follows:

```text
data/<dataset-name>/
├── train.csv
├── validation.csv
├── test.csv
├── holdout_validation.csv
├── holdout_test_0.csv
├── ...
├── holdout_test_9.csv
└── info.json
```

The preprocessing pipeline performs a global chronological split and creates validation and test holdouts.

To prepare a custom interaction dataset, run:

```bash
python3 scripts/dataset_pipeline.py \
  --filename <path-to-csv> \
  --user_col <user-column> \
  --item_col <item-column> \
  --time_col <timestamp-column> \
  --rating_col <rating-column>
```

Then add a descriptor in `configs/dataset/`:

```yaml
_target_: src.datasets.RecSysDataset
name: my-dataset
url: ""
```

Set `url` to a public download location, or leave it empty for local data.

## Efficiency benchmark

Runtime and memory measurements use the same Hydra configurations as ordinary model training:

```bash
python3 scripts/benchmark_efficiency.py -cn sasrec_einv_beauty
```

Each run creates a JSON record below `timing_results/results/`. Laplacian preprocessing can be measured separately from training.

Create comparison plots from the saved runs with:

```bash
python3 scripts/plot_efficiency.py --baseline sasrec
```

Figures and aggregated summaries are written to `reports/efficiency/`.

## Project structure

```text
lap_pe/
├── configs/
│   ├── dataset/                 # Dataset descriptors
│   ├── metrics/                 # Evaluation metric groups
│   ├── model/                   # Reusable model configurations
│   ├── optuna_params/           # Hyperparameter search spaces
│   ├── sasrec_*_*.yaml          # Dataset-specific SASRec variants
│   └── tisasrec_*.yaml          # Dataset-specific TiSASRec runs
├── src/
│   ├── datasets/                # Loading and dataset abstractions
│   ├── metrics/                 # NDCG, Recall, and Coverage
│   ├── models/
│   │   ├── sasrec.py            # SASRec baseline
│   │   ├── sasrec_MLP.py        # SASRec MLP variant
│   │   ├── sasrec_einv.py       # Laplacian positional embeddings
│   │   ├── sasrec_rope.py       # Rotary positional embeddings
│   │   └── tisasrec.py          # Time-interval-aware baseline
│   └── utils/                   # Shared helpers
├── scripts/
│   ├── dataset_pipeline.py      # Dataset preprocessing
│   ├── precompute_laplacian_eigvecs.py
│   ├── benchmark_efficiency.py
│   ├── plot_efficiency.py
│   └── run_tisasrec_time_interval_grid.py
├── run_model.py                 # Train and evaluate one configuration
├── run_optuna.py                # Hyperparameter optimization
├── offline_download.py          # Retrieve configured datasets
└── requirements.txt
```

Generated artifacts are kept outside the source tree: regular runs in `outputs/`, hyperparameter studies in `optuna_outputs/`, timing samples in `timing_results/`, and plots or summaries in `reports/`.
