# Running the DistilBERT Fine-Tune on Colab

`notebooks/02_distilbert_colab.ipynb` fine-tunes `distilbert-base-uncased` as the
improved model for Milestone 2. It needs a GPU and it needs the train/val/test
splits, which are **not** in the repo — `data/processed/*` is gitignored because
the dataset is not ours to redistribute.

Roughly 20 minutes on a T4, against ~4.5 hours on CPU.

## Before you start

**Set the runtime.** Runtime → Change runtime type → **T4 GPU**. The first cell
asserts on this rather than silently falling back to CPU.

Open the notebook in Colab from GitHub (`Mashi98/Ecommerce-Review-Analytics`,
`notebooks/02_distilbert_colab.ipynb`). It clones the repo itself and pulls on
re-runs, so it always trains against the committed `src/`, not a stale copy.

## Getting the splits in

**Option B, direct upload, is the default.** Run the upload cell and pick all
three files from `data/processed/`. No Drive access needed; the cost is
re-uploading ~12 MB every session.

| File | Size | SHA-256 (first 12) |
|---|---|---|
| `train.csv` | 8,643,310 bytes | `c82c3f828d3c` |
| `val.csv` | 1,880,376 bytes | `93bdaac4cc2b` |
| `test.csv` | 1,860,211 bytes | `efd47d8954e7` |

`train.csv` is the large one and the one the Colab uploader handles worst — let
all three finish before moving on.

**Option A, Google Drive**, is there if you would rather not re-upload each
session: put the three files in `MyDrive/capstone/processed`, uncomment that
cell, and skip the upload cell. Change `DRIVE_SPLITS` if you use another folder.

Either way the **verify cell** re-hashes all three against the table above and
stops with a `ValueError` if anything differs. If it fires, the upload is the
problem, not the code.

## Running it

Run the cells in order. One gate worth knowing about: **`RUN_TEST_EVAL` stays
`False`.** The test split has been untouched through all model selection; flip
it to `True` exactly once, for the final number that goes in the report.

## Saving the model

The save cell mounts Drive **for the save only**, independently of how the
splits arrived, and writes to `MyDrive/capstone/distilbert-sentiment/`.

This matters more than it looks. The Colab VM is ephemeral, and with Drive
unmounted `/content/drive/MyDrive/...` is just an ordinary directory on that
VM's disk — `trainer.save_model()` would create it, print `saved to
/content/drive/...`, and the weights would be gone at disconnect, with nothing
in the output suggesting anything went wrong. The cell therefore refuses to save
unless the mount is actually live.

Weights are gitignored; they live in Drive, not the repo.

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
