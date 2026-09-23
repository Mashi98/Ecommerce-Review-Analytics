# Running the DistilBERT Fine-Tune on Colab

`notebooks/02_distilbert_colab.ipynb` fine-tunes `distilbert-base-uncased` as the
improved model for Milestone 2. It needs a GPU and it needs the train/val/test
splits, which are **not** in the repo — `data/processed/*` is gitignored because
the dataset is not ours to redistribute.

Roughly 20 minutes on a T4, against ~4.5 hours on CPU.

## One-time setup

**1. Upload the splits to Drive.**

Create the folder `capstone/processed` in the Drive account you will open Colab
with, and upload all three files from `data/processed/`:

| File | Size | SHA-256 (first 12) |
|---|---|---|
| `train.csv` | 8,643,310 bytes | `c82c3f828d3c` |
| `val.csv` | 1,880,376 bytes | `93bdaac4cc2b` |
| `test.csv` | 1,860,211 bytes | `efd47d8954e7` |

The full path the notebook expects is
`/content/drive/MyDrive/capstone/processed/`. If you put them elsewhere, change
`DRIVE_SPLITS` in the "Get the data" cell.

Wait for the Drive upload to finish before running the notebook — a partial
sync is the most likely thing to go wrong here, which is why the notebook
checksums all three files before it touches the GPU.

**2. Set the runtime.** Runtime → Change runtime type → **T4 GPU**. The first
cell asserts on this rather than silently falling back to CPU.

## Running it

Open the notebook in Colab from GitHub (`Mashi98/Ecommerce-Review-Analytics`,
`notebooks/02_distilbert_colab.ipynb`) and run the cells in order. The notebook
clones the repo itself and pulls on re-runs, so it always trains against the
committed `src/`, not a stale copy.

Two gates worth knowing about:

- **The verify cell** re-hashes the three CSVs against the table above and stops
  with a `ValueError` if anything differs. If it fires, the upload is the
  problem, not the code.
- **`RUN_TEST_EVAL`** stays `False`. The test split has been untouched through
  all model selection; flip it to `True` exactly once, for the final number that
  goes in the report.

## After the run

The fine-tuned model is saved to
`/content/drive/MyDrive/capstone/distilbert-sentiment/` (weights are gitignored —
they live in Drive, not the repo).

Record the output of the last cell for the reproducibility checklist: GPU name,
library versions, hyperparameters and seed.

## Pinned versions

The install cell pins the HuggingFace stack to the versions the local env and
the unit tests run on, so the Colab result matches what the checklist claims:

```
transformers==5.17.0   datasets==5.0.1   accelerate==1.15.0   evaluate==0.4.6
```

`torch` is left as whatever Colab ships, since the pinned stack is compatible
with it and forcing a torch version on Colab tends to break CUDA.
