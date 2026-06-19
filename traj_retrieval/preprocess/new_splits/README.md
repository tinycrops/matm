# Canonical Splits

This directory holds the minimal, canonical train/test split definitions used by
the MATM framework. Only the files needed to run evaluation, populate the index,
and generate LTR data are committed here:

| File | Purpose |
|---|---|
| `alfworld/official_test/final_test_set.json` | ALFWorld evaluation set |
| `alfworld/train/index_source_train_all.json` | ALFWorld trajectories used to populate the index |
| `alfworld/train/samples_ltr_datasets.json` | ALFWorld subset for LTR data generation |
| `webarena/final_test_set.json` | WebArena evaluation set |
| `webarena/train/index_source_train_augmented.json` | WebArena trajectories used to populate the index |
| `webarena/train/samples_ltr_datasets.json` | WebArena subset for LTR data generation |

Larger machine-generated splits (per-consumer-model allocations, pseudo-simulation
splits, intermediate reports) are reproducible and not committed. See
`../README.md` for how to regenerate splits and rebuild the index.
