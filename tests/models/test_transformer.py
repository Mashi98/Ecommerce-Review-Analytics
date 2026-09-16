"""Unit tests for the DistilBERT module.

These cover the pure data/label/config logic only — no model weights are
downloaded and no training runs, so the suite stays fast and works offline.
Actual fine-tuning is exercised on Colab via notebooks/02_distilbert_colab.ipynb.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.baseline import SENTIMENT_LABELS
from src.models.transformer import (
    ID2LABEL,
    LABEL2ID,
    build_training_args,
    build_weighted_loss,
    class_weights,
    compute_metrics,
    decode_labels,
    encode_labels,
    prepare_frame,
)


@pytest.fixture
def split_df():
    return pd.DataFrame(
        {
            "Review Text": ["love this dress", "it was fine", "poor quality", None, "   "],
            "Review Text Clean": ["love dress", "fine", "poor quality", "", ""],
            "Sentiment": ["positive", "neutral", "negative", "positive", "negative"],
            "Rating": [5, 3, 1, 5, 2],
        }
    )


def test_label_maps_are_inverses():
    assert set(LABEL2ID) == set(SENTIMENT_LABELS)
    assert all(ID2LABEL[i] == label for label, i in LABEL2ID.items())


def test_encode_decode_roundtrip():
    labels = ["positive", "negative", "neutral", "positive"]
    assert decode_labels(encode_labels(labels)) == labels


def test_encode_labels_rejects_unknown_label():
    with pytest.raises(ValueError, match="Unknown sentiment labels"):
        encode_labels(["positive", "mildly_annoyed"])


def test_prepare_frame_drops_blank_raw_text(split_df):
    result = prepare_frame(split_df)
    assert list(result.columns) == ["text", "label"]
    assert len(result) == 3  # None and whitespace-only rows dropped
    assert "   " not in result["text"].to_list()


def test_prepare_frame_encodes_labels_as_ints(split_df):
    result = prepare_frame(split_df)
    assert result["label"].to_list() == [
        LABEL2ID["positive"],
        LABEL2ID["neutral"],
        LABEL2ID["negative"],
    ]


def test_class_weights_are_inverse_frequency():
    weights = class_weights([0] * 100 + [1] * 50 + [2] * 850)
    # the rarest class must carry the largest weight
    assert weights[1] > weights[0] > weights[2]
    assert np.isclose(np.mean(weights), 1.0)


def test_class_weights_uniform_for_balanced_labels():
    weights = class_weights([0, 1, 2] * 10)
    assert np.allclose(weights, [1.0, 1.0, 1.0])


def test_class_weights_rejects_absent_class():
    with pytest.raises(ValueError, match="absent"):
        class_weights([0, 0, 1, 1])  # no examples of class 2


def test_compute_metrics_matches_baseline_metric_names():
    logits = np.array([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]])
    label_ids = np.array([0, 1, 2])
    metrics = compute_metrics((logits, label_ids))
    assert set(metrics) == {"accuracy", "macro_f1", "weighted_f1"}
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0


def test_compute_metrics_penalises_majority_collapse():
    """All-positive predictions: high accuracy, poor macro-F1."""
    logits = np.tile([0.0, 0.0, 5.0], (10, 1))
    label_ids = np.array([2] * 8 + [0, 1])
    metrics = compute_metrics((logits, label_ids))
    assert metrics["accuracy"] == pytest.approx(0.8)
    assert metrics["macro_f1"] < 0.4


def test_training_args_select_best_on_macro_f1(tmp_path):
    args = build_training_args(output_dir=tmp_path)
    assert args.metric_for_best_model == "macro_f1"
    assert args.greater_is_better is True
    assert args.load_best_model_at_end is True
    assert args.eval_strategy == args.save_strategy  # required for load_best_model_at_end


def test_training_args_respect_overrides(tmp_path):
    args = build_training_args(output_dir=tmp_path, epochs=5, batch_size=8, learning_rate=3e-5)
    assert args.num_train_epochs == 5
    assert args.per_device_train_batch_size == 8
    assert args.learning_rate == pytest.approx(3e-5)


def test_weighted_loss_penalises_minority_error_more():
    """The same logits should cost more when the true label is a rare class."""
    import torch

    loss_fn = build_weighted_loss([5.0, 1.0, 0.2])
    outputs = type("Out", (), {"logits": torch.tensor([[0.0, 0.0, 10.0]])})()

    rare_loss = loss_fn(outputs, torch.tensor([0]))
    common_loss = loss_fn(outputs, torch.tensor([2]))
    assert rare_loss.item() > common_loss.item()
