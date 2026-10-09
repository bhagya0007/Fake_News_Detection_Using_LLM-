"""Baseline Inference Service.

Provides reusable inference using the fitted TF-IDF + Logistic Regression pipeline:
- Loads serialized Pipeline artifact from disk.
- Validates and sanitizes input statement text via clean_text.
- Computes predicted label and class probability estimates.
- Strictly validates that 'unverifiable' is never output by the classical model.

Note on Terminology:
The values returned in 'probabilities' represent model probability estimates
(softmax probabilities from Logistic Regression), not guaranteed real-world truth
confidence, as calibration has not been performed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
from sklearn.pipeline import Pipeline

from backend.app.services.dataset_loader import ALLOWED_PROJECT_LABELS
from backend.app.services.text_cleaner import clean_text

# Default model artifact path
DEFAULT_MODEL_PATH: Path = (
    Path(__file__).resolve().parents[3] / "models" / "baseline_initial.joblib"
)


def load_baseline_pipeline(model_path: Optional[Path] = None) -> Any:
    """Load a serialized scikit-learn Pipeline artifact from disk.

    Args:
        model_path: Optional custom path to the joblib artifact.
            Defaults to models/baseline_initial.joblib.

    Returns:
        The deserialized Pipeline object.

    Raises:
        FileNotFoundError: If the model artifact does not exist.
        TypeError: If the loaded object is not a fitted Pipeline with required methods.
    """
    path = DEFAULT_MODEL_PATH if model_path is None else Path(model_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Baseline model artifact not found at: {path.as_posix()}"
        )

    pipeline = joblib.load(path)

    if not hasattr(pipeline, "predict"):
        raise TypeError(
            f"Loaded object from {path.name} does not implement predict."
        )

    if not hasattr(pipeline, "predict_proba") and not hasattr(pipeline, "decision_function"):
        raise TypeError(
            f"Loaded object from {path.name} does not implement predict_proba or decision_function."
        )

    if not hasattr(pipeline, "classes_"):
        raise TypeError(
            f"Loaded pipeline from {path.name} does not expose classes_ attribute."
        )

    return pipeline


def predict_baseline(pipeline: Any, text: str) -> dict[str, Any]:
    """Run baseline prediction and compute model probability or decision scores for a single text.

    Args:
        pipeline: Fitted scikit-learn Pipeline (LogisticRegression or LinearSVC).
        text: Raw statement string to evaluate.

    Returns:
        Dictionary containing:
        - 'statement': Cleaned input statement.
        - 'predicted_label': One of 'likely_reliable', 'misleading', 'likely_fake'.
        - 'probabilities': Dict mapping class names to model probability estimates (if available).
        - 'decision_scores': Dict mapping class names to decision scores (if predict_proba not available).

    Raises:
        ValueError: If text is empty or invalid, or if an unexpected label is predicted.
        RuntimeError: If 'unverifiable' is illegally emitted by the classical baseline.
    """
    cleaned = clean_text(text)

    # Scikit-learn expects an iterable of strings
    predicted_label = str(pipeline.predict([cleaned])[0])

    # Guard: classical baseline must never output 'unverifiable'
    if predicted_label == "unverifiable":
        raise RuntimeError(
            "Classical baseline illegally emitted 'unverifiable'. "
            "Abstention is reserved for downstream fusion logic."
        )

    if predicted_label not in ALLOWED_PROJECT_LABELS:
        raise ValueError(
            f"Model predicted unexpected label '{predicted_label}'. "
            f"Allowed labels: {ALLOWED_PROJECT_LABELS}"
        )

    result: dict[str, Any] = {
        "statement": cleaned,
        "predicted_label": predicted_label,
    }

    # Safe mapping using pipeline's actual classes_ order
    if hasattr(pipeline, "predict_proba"):
        prob_array = pipeline.predict_proba([cleaned])[0]
        result["probabilities"] = {
            str(cls): float(prob) for cls, prob in zip(pipeline.classes_, prob_array)
        }
    elif hasattr(pipeline, "decision_function"):
        scores = pipeline.decision_function([cleaned])[0]
        if hasattr(scores, "__len__") and len(scores) == len(pipeline.classes_):
            result["decision_scores"] = {
                str(cls): float(sc) for cls, sc in zip(pipeline.classes_, scores)
            }
        else:
            result["decision_scores"] = float(scores)
        result["probabilities"] = None

    return result


def predict_baseline_batch(
    pipeline: Any, texts: list[str]
) -> list[dict[str, Any]]:
    """Run baseline prediction on a batch of text statements.

    Args:
        pipeline: Fitted scikit-learn Pipeline.
        texts: List of raw statement strings.

    Returns:
        List of prediction result dictionaries.
    """
    return [predict_baseline(pipeline, text) for text in texts]
