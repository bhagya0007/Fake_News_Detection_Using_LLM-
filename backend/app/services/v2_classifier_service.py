"""Production V2 Text Classifier Service.

Provides modular inference for V2 trained text classifiers:
- Model A: V2 TF-IDF + Logistic Regression
- Model B: DistilBERT Sequence Classifier
- Model C: Frozen all-MiniLM-L6-v2 Embeddings + Logistic Regression

STRICT CONTRACT GUARANTEES:
- Strictly 3-class semantic label space:
    * likely_reliable
    * misleading
    * likely_fake
- Zero predictions of 'unverifiable' (strictly enforced by runtime guard).
- Epistemic abstention ('unverifiable') is reserved for downstream retrieval/fusion.
- Full compatibility with HybridAnalysisEngine via MLAssessment schema.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

from backend.app.config import MODELS_DIR
from backend.app.schemas import (
    ClassProbabilities,
    MLAssessment,
    MLLabel,
    adapter_from_classical_prediction,
)
from backend.app.services.text_cleaner import clean_text

V2_MODELS_DIR = MODELS_DIR / "v2"

# Canonical 3-class mapping
LABEL_ENCODING: dict[str, int] = {
    "likely_reliable": 0,
    "misleading": 1,
    "likely_fake": 2,
}
ID2LABEL: dict[int, str] = {0: "likely_reliable", 1: "misleading", 2: "likely_fake"}
ALLOWED_LABELS: set[str] = {"likely_reliable", "misleading", "likely_fake"}


class V2ClassifierService:
    """Production service for V2 trained text classifiers."""

    def __init__(
        self,
        model_type: str = "tfidf",  # 'tfidf', 'minilm', or 'distilbert'
        custom_artifact_path: Optional[Path] = None,
    ) -> None:
        self.model_type = model_type.lower().strip()
        self.custom_artifact_path = custom_artifact_path
        self._model: Any = None
        self._embedder: Any = None
        self._tokenizer: Any = None
        self._metadata: Optional[Dict[str, Any]] = None

    def _load_metadata(self, meta_path: Path) -> Dict[str, Any]:
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _init_model(self) -> None:
        if self._model is not None:
            return

        if self.model_type in ["tfidf", "v2_tfidf_logreg"]:
            path = self.custom_artifact_path or (V2_MODELS_DIR / "tfidf_logreg" / "model.joblib")
            if not path.exists():
                raise FileNotFoundError(f"V2 TF-IDF model not found at {path.as_posix()}")
            self._model = joblib.load(path)
            self._metadata = self._load_metadata(path.parent / "metadata.json")

        elif self.model_type in ["minilm", "v2_minilm_logreg"]:
            path = self.custom_artifact_path or (V2_MODELS_DIR / "minilm_logreg" / "model.joblib")
            if not path.exists():
                raise FileNotFoundError(f"V2 MiniLM classifier not found at {path.as_posix()}")
            self._model = joblib.load(path)
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            self._metadata = self._load_metadata(path.parent / "metadata.json")

        elif self.model_type in ["distilbert", "v2_distilbert"]:
            dir_path = self.custom_artifact_path or (V2_MODELS_DIR / "distilbert")
            if not dir_path.exists():
                raise FileNotFoundError(f"V2 DistilBERT directory not found at {dir_path.as_posix()}")
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._tokenizer = AutoTokenizer.from_pretrained(str(dir_path))
            self._model = AutoModelForSequenceClassification.from_pretrained(str(dir_path))
            self._model.to(self._device)
            self._model.eval()
            self._metadata = self._load_metadata(dir_path / "metadata.json")

        else:
            raise ValueError(f"Unknown V2 classifier type: {self.model_type}")

    def predict_dict(self, text: str) -> dict[str, Any]:
        """Predict class label and probability estimates for input text.

        Returns:
            dict containing:
            - 'statement': cleaned text
            - 'predicted_label': one of 'likely_reliable', 'misleading', 'likely_fake'
            - 'probabilities': dict mapping the 3 classes to estimated probabilities
        """
        self._init_model()
        cleaned = clean_text(text)

        if self.model_type in ["tfidf", "v2_tfidf_logreg"]:
            pred_idx = int(self._model.predict([cleaned])[0])
            prob_arr = self._model.predict_proba([cleaned])[0]
            predicted_label = ID2LABEL[pred_idx]
            probabilities = {
                ID2LABEL[i]: float(prob_arr[i]) for i in range(len(prob_arr))
            }

        elif self.model_type in ["minilm", "v2_minilm_logreg"]:
            emb = self._embedder.encode([cleaned], normalize_embeddings=True)
            pred_idx = int(self._model.predict(emb)[0])
            prob_arr = self._model.predict_proba(emb)[0]
            predicted_label = ID2LABEL[pred_idx]
            probabilities = {
                ID2LABEL[i]: float(prob_arr[i]) for i in range(len(prob_arr))
            }

        elif self.model_type in ["distilbert", "v2_distilbert"]:
            import torch
            with torch.no_grad():
                inputs = self._tokenizer(
                    [cleaned],
                    truncation=True,
                    padding=True,
                    max_length=128,
                    return_tensors="pt"
                ).to(self._device)
                outputs = self._model(**inputs)
                prob_tensor = torch.softmax(outputs.logits, dim=-1)[0]
                prob_arr = prob_tensor.cpu().numpy()
                pred_idx = int(torch.argmax(prob_tensor).item())
                predicted_label = ID2LABEL[pred_idx]
                probabilities = {
                    ID2LABEL[i]: float(prob_arr[i]) for i in range(len(prob_arr))
                }

        # Runtime guard: ML classifier must NEVER emit unverifiable
        if predicted_label == "unverifiable":
            raise RuntimeError(
                "V2 ML classifier illegally emitted 'unverifiable'. "
                "Abstention is strictly reserved for downstream retrieval/evidence consensus."
            )

        if predicted_label not in ALLOWED_LABELS:
            raise ValueError(f"Unexpected prediction: {predicted_label}")

        return {
            "statement": cleaned,
            "predicted_label": predicted_label,
            "probabilities": probabilities,
        }

    def predict(self, text: str) -> MLAssessment:
        """Production method compliant with HybridAnalysisEngine."""
        pred_dict = self.predict_dict(text)
        model_display_name = f"V2 {self.model_type.upper()} Classifier"
        return adapter_from_classical_prediction(
            pred_dict,
            model_name=model_display_name,
            model_version="v2.5_frozen",
        )
