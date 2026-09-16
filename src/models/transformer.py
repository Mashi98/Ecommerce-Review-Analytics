"""Fine-tuned DistilBERT for review sentiment — the improved model.

Trained against the same 3-class sentiment target, the same train/val/test
splits and the same row set as the TF-IDF baseline in `src.models.baseline`,
so the comparison is like-for-like.

One deliberate difference: this model is fed the **raw** `Review Text`, not
the lemmatized/stopword-stripped `Review Text Clean` the baseline uses.
DistilBERT has its own subword tokenizer and draws signal from negations,
function words and punctuation that `preprocess()` strips out — feeding it
the preprocessed text would handicap it for no benefit. The row set stays
identical, so only the representation differs, not the examples.

Designed to run on Colab (T4). On CPU it works but is impractically slow.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.models.baseline import (
    LABEL_COL,
    RANDOM_SEED,
    SENTIMENT_LABELS,
    load_split,
)

MODEL_NAME = "distilbert-base-uncased"
TEXT_COL_RAW = "Review Text"
MAX_LENGTH = 256

LEARNING_RATE = 2e-5
BATCH_SIZE = 16
EPOCHS = 3
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 500

LABEL2ID = {label: i for i, label in enumerate(SENTIMENT_LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}

ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "models" / "distilbert-sentiment"


def encode_labels(labels) -> list[int]:
    """Map string sentiment labels onto the integer ids the model expects."""
    unknown = set(labels) - set(LABEL2ID)
    if unknown:
        raise ValueError(f"Unknown sentiment labels: {sorted(unknown)}")
    return [LABEL2ID[label] for label in labels]


def decode_labels(ids) -> list[str]:
    """Map model output ids back onto string sentiment labels."""
    return [ID2LABEL[int(i)] for i in ids]


def prepare_frame(df: pd.DataFrame, text_col: str = TEXT_COL_RAW) -> pd.DataFrame:
    """Reduce a split to the two columns the trainer needs: `text` and `label`.

    Rows whose raw text is blank are dropped — `load_split()` already removes
    rows with no usable review text, this guards the raw column specifically.
    """
    out = df.copy()
    out[text_col] = out[text_col].fillna("").astype(str)
    out = out[out[text_col].str.strip() != ""]
    return pd.DataFrame(
        {
            "text": out[text_col].to_list(),
            "label": encode_labels(out[LABEL_COL].to_list()),
        }
    )


def build_dataset(df: pd.DataFrame, tokenizer, text_col: str = TEXT_COL_RAW, max_length: int = MAX_LENGTH):
    """Tokenized HuggingFace Dataset built from a split DataFrame."""
    from datasets import Dataset

    dataset = Dataset.from_pandas(prepare_frame(df, text_col=text_col), preserve_index=False)
    return dataset.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=max_length),
        batched=True,
        remove_columns=["text"],
    )


def compute_metrics(eval_pred) -> dict[str, float]:
    """Accuracy + macro-F1 for the Trainer, matching the baseline's metrics."""
    from src.evaluation.metrics import classification_metrics

    logits, label_ids = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return classification_metrics(
        decode_labels(label_ids), decode_labels(predictions), SENTIMENT_LABELS
    )


def build_model(num_labels: int = len(SENTIMENT_LABELS), model_name: str = MODEL_NAME):
    """DistilBERT with a fresh classification head sized to the label set."""
    from transformers import AutoModelForSequenceClassification

    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )


def build_training_args(
    output_dir: Path = ARTIFACT_DIR,
    learning_rate: float = LEARNING_RATE,
    batch_size: int = BATCH_SIZE,
    epochs: int = EPOCHS,
    weight_decay: float = WEIGHT_DECAY,
    warmup_steps: int = WARMUP_STEPS,
    seed: int = RANDOM_SEED,
    fp16: bool = False,
):
    """Training configuration.

    Selects the best checkpoint on macro-F1 rather than loss or accuracy —
    the same metric the baseline is judged on, and the one that actually
    reflects minority-class performance. Set `fp16=True` on a GPU.
    """
    from transformers import TrainingArguments

    return TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=100,
        seed=seed,
        fp16=fp16,
        report_to=[],
    )


def class_weights(labels, num_labels: int = len(SENTIMENT_LABELS)) -> list[float]:
    """Inverse-frequency class weights, normalized to mean 1.

    Mirrors sklearn's `class_weight="balanced"`, which is what made the
    baseline's macro-F1 competitive — the transformer gets the same
    treatment so the comparison isolates the model, not the imbalance
    strategy.
    """
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=num_labels)
    if (counts == 0).any():
        raise ValueError(f"Some classes are absent from the labels: {counts.tolist()}")
    weights = len(labels) / (num_labels * counts)
    return (weights / weights.mean()).tolist()


def build_weighted_loss(weights):
    """A `compute_loss_func` for Trainer that applies per-class weights."""
    import torch
    from torch.nn import CrossEntropyLoss

    def compute_loss(outputs, labels, num_items_in_batch=None):
        weight_tensor = torch.tensor(weights, dtype=outputs.logits.dtype, device=outputs.logits.device)
        return CrossEntropyLoss(weight=weight_tensor)(outputs.logits, labels)

    return compute_loss


def train_transformer(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_name: str = MODEL_NAME,
    use_class_weights: bool = True,
    output_dir: Path = ARTIFACT_DIR,
    **training_kwargs,
):
    """Fine-tune DistilBERT on the training split, selecting on val macro-F1.

    Returns the fitted `Trainer` — call `.evaluate()` on it, or use
    `predict_labels()` to get string predictions for a held-out split.
    """
    from transformers import AutoTokenizer, DataCollatorWithPadding, Trainer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_ds = build_dataset(train_df, tokenizer)
    val_ds = build_dataset(val_df, tokenizer)

    loss_func = build_weighted_loss(class_weights(train_ds["label"])) if use_class_weights else None

    trainer = Trainer(
        model=build_model(model_name=model_name),
        args=build_training_args(output_dir=output_dir, **training_kwargs),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        compute_loss_func=loss_func,
    )
    trainer.train()
    return trainer


def predict_labels(trainer, df: pd.DataFrame) -> list[str]:
    """String sentiment predictions for a split, using a fitted Trainer."""
    dataset = build_dataset(df, trainer.processing_class)
    logits = trainer.predict(dataset).predictions
    return decode_labels(np.argmax(logits, axis=-1))


if __name__ == "__main__":
    from src.evaluation.metrics import evaluate

    train_split, val_split = load_split("train"), load_split("val")
    print(f"train: {len(train_split)} rows | val: {len(val_split)} rows")

    fitted = train_transformer(train_split, val_split)
    predictions = predict_labels(fitted, val_split)
    result = evaluate(
        type("Fixed", (), {"predict": staticmethod(lambda _: predictions)})(),
        val_split[TEXT_COL_RAW],
        prepare_frame(val_split)["label"].map(ID2LABEL),
        SENTIMENT_LABELS,
    )
    print(f"\nValidation summary: {result['summary']}")
    print(f"\nPer-class metrics:\n{result['per_class'].to_string()}")
    print(f"\nConfusion matrix:\n{result['confusion_matrix'].to_string()}")
