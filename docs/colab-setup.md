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

**Option A, Google Drive, is the default** (`USE_DRIVE = True`), because the run
has to mount Drive anyway to save the model. Put the three files in
`MyDrive/capstone/processed` once; change `DRIVE_SPLITS` if you use another folder.

**Option B, direct upload**, is there if you set `USE_DRIVE = False`: run the
upload cell and pick all three files from `data/processed/`. `train.csv` is the
large one and the one the Colab uploader handles worst; let all three finish.

| File | Size | SHA-256 (first 12) |
|---|---|---|
| `train.csv` | 8,643,310 bytes | `c82c3f828d3c` |
| `val.csv` | 1,880,376 bytes | `93bdaac4cc2b` |
| `test.csv` | 1,860,211 bytes | `efd47d8954e7` |

Either way the **verify cell** re-hashes all three against the table above and
stops with a `ValueError` if anything differs. If it fires, the upload is the
problem, not the code.

## Saving the model (section 4b, straight after training)

The Milestone 2 run's weights were lost: saving was the last cell, and the
session reset before it ran. The save now happens immediately after training,
before validation or test scoring, into one Drive folder for the whole run:

```
MyDrive/capstone/final-run/
  model/                            weights + tokenizer
  run_info.json                     date, git commit, GPU, versions, best checkpoint
  val_preds_distilbert_final.csv    written in section 5
  test_preds_distilbert.csv         written in section 6
```

The cell stops if Drive is not actually mounted (an unmounted
`/content/drive/...` is an ordinary folder on the VM that vanishes at disconnect,
with nothing in the output to say so) and if `final-run/model/` already exists,
so a later session cannot silently overwrite the final model.

## The final test evaluation (section 6)

The test split has been untouched through all model selection. The procedure,
done once:

1. Run sections 1 to 5 as normal. Check the validation macro-F1 is in the
   expected range (the Milestone 2 runs gave 0.676 and 0.682).
2. Set `RUN_TEST_EVAL = True` in section 6 and run that cell. It scores
   DistilBERT on the test set, writes `test_preds_distilbert.csv` to `final-run/`,
   prints the test metrics, and downloads both prediction CSVs. It refuses to run
   if that file already exists.
3. Put the two downloaded CSVs in `reports/` in the repo, then run locally:

   ```
   python -m scripts.test_evaluation
   ```

   This checks every prediction row lines up with the local val and test splits
   (by a digest of the review text), retrains the baseline and checks it still
   gives its reported 0.6245 validation macro-F1, scores the baseline on the test
   set, and writes `reports/test_results.md` with both models' test results and
   a paired bootstrap interval for the difference.

The baseline is scored locally rather than in Colab so its test number comes from
the pinned environment and can be reproduced on any machine.

Record the output of the last cell for the reproducibility checklist; the same
details are in `final-run/run_info.json`.

## Pinned versions

The install cell pins the HuggingFace stack to the versions the local env and
the unit tests run on, so the Colab result matches what the checklist claims:

```
transformers==5.17.0   datasets==5.0.1   accelerate==1.15.0   evaluate==0.4.6
```

`torch` is left as whatever Colab ships, since the pinned stack is compatible
with it and forcing a torch version on Colab tends to break CUDA.
