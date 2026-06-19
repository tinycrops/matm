# Preprocessing: Building the Shared MATM Index

This directory contains the scripts that turn raw benchmark trajectories into the
**shared LanceDB trajectory index** that MATM consumer agents retrieve from, plus
the canonical train/test split definitions used by the framework.

We **do not** redistribute the raw ALFWorld / WebArena task data (it is publicly
available from the upstream benchmarks) or the large machine-generated split
permutations. Instead we ship the **generation scripts** and a small set of
**canonical split files** so you can rebuild everything yourself.

## Directory layout

```
preprocess/
├── alfworld/                 # ALFWorld staging, indexing & split scripts
│   ├── trajectory_entry.py           # Index record schema (used by the runtime)
│   ├── create_data_splits_train_only.py
│   ├── create_seq_to_seq_indices_train_only.py
│   ├── create_lancedb_index_train_only.py
│   ├── sample_train_subset.py        # builds the LTR sampling subset
│   ├── sample_official_test_subset.py# builds the final test set
│   └── verify_no_leakage_train_only.py
├── webarena/                 # WebArena staging, indexing & split scripts
│   ├── trajectory_entry.py           # Index record schema (used by the runtime)
│   ├── staged_trajectory_schema.py
│   ├── stage_gold_trajectories_train_only.py
│   ├── stage_successful_trajectories.py
│   ├── create_data_splits_train_only.py
│   ├── create_lancedb_index_train_only.py
│   ├── create_lancedb_indices_train_only.py
│   ├── sample_test_subset.py
│   └── verify_no_leakage_train_only.py
├── new_splits/               # Canonical split definitions (see below)
└── no_retrieval_runs/        # Cached no-retrieval baseline JSON (gitignored; used by simulate strategies)
```

## Canonical splits (`new_splits/`)

Only the minimal, canonical split files needed to run the framework are committed:

| File | Purpose |
|---|---|
| `alfworld/official_test/final_test_set.json` | ALFWorld evaluation set (default in `run_evaluation_config.yaml`) |
| `alfworld/train/index_source_train_all.json` | ALFWorld trajectories used to populate the index |
| `alfworld/train/samples_ltr_datasets.json` | ALFWorld subset used to generate LTR training data |
| `webarena/final_test_set.json` | WebArena evaluation set |
| `webarena/train/index_source_train_augmented.json` | WebArena trajectories used to populate the index |
| `webarena/train/samples_ltr_datasets.json` | WebArena subset used to generate LTR training data |

Larger, fully reproducible artifacts (per-consumer-model task allocations,
pseudo-simulation splits, intermediate reports) are **not** committed; regenerate
them with the scripts below when needed.

## Rebuilding from scratch

1. **Fetch ALFWorld data** (from the repo root; WebArena setup is separate — see the [WebArena guide](https://github.com/web-arena-x/webarena)):

   ```bash
   python setup_environments.py --worlds alfworld
   ```

2. **Build splits** (writes into `new_splits/`):

   ```bash
   python traj_retrieval/preprocess/alfworld/create_data_splits_train_only.py
   python traj_retrieval/preprocess/webarena/create_data_splits_train_only.py
   ```

3. **Build the LanceDB index** (writes to `environments/train_only_lancedb/<env>/lancedb_indices`):

   ```bash
   python traj_retrieval/preprocess/alfworld/create_seq_to_seq_indices_train_only.py
   python traj_retrieval/preprocess/alfworld/create_lancedb_index_train_only.py

   python traj_retrieval/preprocess/webarena/stage_gold_trajectories_train_only.py
   python traj_retrieval/preprocess/webarena/create_lancedb_indices_train_only.py
   python traj_retrieval/preprocess/webarena/create_lancedb_index_train_only.py
   ```

Trajectories are chunked with window size `W=5` (key = task description + last 5
action–observation steps; value = next 5 steps) and embedded for dense retrieval.
At inference, consumer agents pull from this index, and (optionally) producer
agents push their own successful runtime trajectories back into it — see the
"Shared trajectory index" section of the top-level `README.md`.
