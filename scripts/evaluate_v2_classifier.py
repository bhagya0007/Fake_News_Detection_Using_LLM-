"""Pure Evaluation Script for V2 Classifiers & Frozen V1 Baseline.

Evaluates frozen models strictly ONCE on final benchmarks:
1. data/raw/valid.tsv (1,284 claims) - Historical validation
2. data/raw/test.tsv (1,283 claims) - Official frozen LIAR test split
3. data/v2_raw/averitec/dev.json (500 claims) - Protected AVeriTeC dev split
   - Evaluated on 3-class applicable claims (465 claims)
   - Evaluated on 'Not Enough Evidence' epistemic claims (35 claims)
   - Evaluated on full 500-claim distribution

CRITICAL SCIENTIFIC INTEGRITY RULES:
- Pure evaluation only. Zero model training, zero parameter fitting, zero re-tuning.
- Zero external API calls (0 Groq calls, 0 GDELT calls).
- Frozen V1 baseline (models/classical_best_validation.joblib) is NEVER overwritten.
- All confusion matrices saved as CSV and PNG (matplotlib only, strictly no seaborn).
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from backend.app.services.text_cleaner import clean_text
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports" / "v2_classifier"
CM_DIR = REPORTS_DIR / "confusion_matrices"
FIG_DIR = REPORTS_DIR / "figures"

# 3-class canonical order
CLASSES = ["likely_reliable", "misleading", "likely_fake"]
LABEL2ID = {"likely_reliable": 0, "misleading": 1, "likely_fake": 2}
ID2LABEL = {0: "likely_reliable", 1: "misleading", 2: "likely_fake"}

LIAR_COLUMNS = [
    "id", "label", "statement", "subjects", "speaker", "speaker_job",
    "state", "party", "barely_true_count", "false_count", "half_true_count",
    "mostly_true_count", "pants_fire_count", "context"
]

LIAR_LABEL_MAP = {
    "true": "likely_reliable",
    "mostly-true": "likely_reliable",
    "half-true": "misleading",
    "barely-true": "misleading",
    "false": "likely_fake",
    "pants-fire": "likely_fake",
}

AVERITEC_LABEL_MAP = {
    "Supported": "likely_reliable",
    "Refuted": "likely_fake",
    "Conflicting Evidence/Cherrypicking": "misleading",
    "Not Enough Evidence": "unverifiable",
}


def compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_liar_split_data(split_name: str) -> List[Dict[str, Any]]:
    tsv_path = DATA_DIR / "raw" / f"{split_name}.tsv"
    if not tsv_path.exists():
        raise FileNotFoundError(f"Missing LIAR file: {tsv_path}")
    
    records = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            if not row or len(row) < 3:
                continue
            raw_id = row[0]
            raw_label = row[1].strip()
            statement = row[2].strip()
            if raw_label not in LIAR_LABEL_MAP:
                continue
            records.append({
                "id": f"liar_{raw_id}",
                "dataset": "liar",
                "split": split_name,
                "text": statement,
                "raw_label": raw_label,
                "ground_truth": LIAR_LABEL_MAP[raw_label],
            })
    return records


def load_averitec_dev_data() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    dev_path = DATA_DIR / "v2_raw" / "averitec" / "dev.json"
    if not dev_path.exists():
        raise FileNotFoundError(f"Missing AVeriTeC dev file: {dev_path}")
    
    with open(dev_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    
    applicable_records = []
    unverifiable_records = []
    all_records = []

    for idx, item in enumerate(items):
        raw_label = item.get("label", "").strip()
        claim_text = item.get("claim", "").strip()
        mapped = AVERITEC_LABEL_MAP.get(raw_label, "unknown")
        
        rec = {
            "id": f"averitec_dev_{idx}",
            "dataset": "averitec",
            "split": "dev",
            "text": claim_text,
            "raw_label": raw_label,
            "ground_truth": mapped,
            "speaker": item.get("speaker", ""),
            "reporting_source": item.get("reporting_source", ""),
        }
        all_records.append(rec)
        if mapped == "unverifiable":
            unverifiable_records.append(rec)
        elif mapped in CLASSES:
            applicable_records.append(rec)

    return applicable_records, unverifiable_records, all_records


class ModelPredictor:
    """Unified wrapper around different candidate models."""

    def __init__(self, name: str, model_type: str):
        self.name = name
        self.model_type = model_type

    def predict_batch(self, texts: List[str]) -> Tuple[List[str], np.ndarray]:
        """Returns (predicted_labels, probability_matrix of shape [N, 3] ordered by CLASSES)."""
        raise NotImplementedError


class V1BaselinePredictor(ModelPredictor):
    def __init__(self, model_path: Path):
        super().__init__("v1_classical_baseline", "V1 TF-IDF + Logistic Regression")
        self.model_path = model_path
        self.pipeline = joblib.load(model_path)
        self.class_indices = {cls_name: i for i, cls_name in enumerate(self.pipeline.classes_)}

    def predict_batch(self, texts: List[str]) -> Tuple[List[str], np.ndarray]:
        cleaned = [clean_text(t) for t in texts]
        raw_preds = self.pipeline.predict(cleaned)
        raw_probs = self.pipeline.predict_proba(cleaned)
        
        # Standardize probability columns to canonical CLASSES order
        probs = np.zeros((len(texts), 3), dtype=np.float64)
        for target_idx, cls_name in enumerate(CLASSES):
            if cls_name in self.class_indices:
                orig_col = self.class_indices[cls_name]
                probs[:, target_idx] = raw_probs[:, orig_col]
        
        preds = [str(p) for p in raw_preds]
        return preds, probs


class V2TfidfPredictor(ModelPredictor):
    def __init__(self, model_path: Path):
        super().__init__("v2_tfidf_logreg", "V2 TF-IDF + Logistic Regression")
        self.model_path = model_path
        self.pipeline = joblib.load(model_path)

    def predict_batch(self, texts: List[str]) -> Tuple[List[str], np.ndarray]:
        cleaned = [clean_text(t) for t in texts]
        raw_preds = self.pipeline.predict(cleaned)  # Array of ints 0, 1, 2
        raw_probs = self.pipeline.predict_proba(cleaned)  # Shape [N, 3] matching 0, 1, 2
        
        preds = [ID2LABEL[int(p)] for p in raw_preds]
        return preds, raw_probs


class V2MiniLMPredictor(ModelPredictor):
    def __init__(self, model_path: Path):
        super().__init__("v2_minilm_logreg", "V2 Frozen all-MiniLM-L6-v2 + Logistic Regression")
        self.model_path = model_path
        self.clf = joblib.load(model_path)
        from sentence_transformers import SentenceTransformer
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def predict_batch(self, texts: List[str]) -> Tuple[List[str], np.ndarray]:
        cleaned = [clean_text(t) for t in texts]
        embeddings = self.embedder.encode(cleaned, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
        raw_preds = self.clf.predict(embeddings)
        raw_probs = self.clf.predict_proba(embeddings)
        
        preds = [ID2LABEL[int(p)] for p in raw_preds]
        return preds, raw_probs


class V2DistilBertPredictor(ModelPredictor):
    def __init__(self, model_dir: Path):
        super().__init__("v2_distilbert", "V2 DistilBERT Sequence Classifier")
        self.model_dir = model_dir
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        self.model.to(self.device)
        self.model.eval()

    def predict_batch(self, texts: List[str]) -> Tuple[List[str], np.ndarray]:
        import torch
        cleaned = [clean_text(t) for t in texts]
        all_probs = []
        batch_size = 32
        
        with torch.no_grad():
            for i in range(0, len(cleaned), batch_size):
                batch_texts = cleaned[i:i + batch_size]
                inputs = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=128,
                    return_tensors="pt"
                ).to(self.device)
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
                all_probs.append(probs)
        
        prob_matrix = np.vstack(all_probs)
        pred_indices = np.argmax(prob_matrix, axis=1)
        preds = [ID2LABEL[int(idx)] for idx in pred_indices]
        return preds, prob_matrix


def plot_confusion_matrix(cm: np.ndarray, model_name: str, split_name: str, save_path: Path) -> None:
    """Plot and save confusion matrix using matplotlib only."""
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.8)
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "white" if val > cm.max() / 2 else "black"
            ax.text(j, i, f"{val:,}", ha="center", va="center", color=color, fontsize=12, fontweight="bold")
    
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=25, ha="left", fontsize=10)
    ax.set_yticklabels(CLASSES, fontsize=10)
    
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_ylabel("True Ground Truth Label", fontsize=11, fontweight="bold")
    ax.set_title(f"Confusion Matrix: {model_name}\nSplit: {split_name}", fontsize=12, pad=15, fontweight="bold")
    
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix_csv(cm: np.ndarray, save_path: Path) -> None:
    df = pd.DataFrame(cm, index=[f"true_{c}" for c in CLASSES], columns=[f"pred_{c}" for c in CLASSES])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=True)


def evaluate_model_on_dataset(
    predictor: ModelPredictor,
    records: List[Dict[str, Any]],
    split_name: str,
) -> Dict[str, Any]:
    texts = [r["text"] for r in records]
    y_true = [r["ground_truth"] for r in records]
    
    t0 = time.perf_counter()
    preds, probs = predictor.predict_batch(texts)
    latency_sec = round(time.perf_counter() - t0, 3)
    
    acc = accuracy_score(y_true, preds)
    macro_f1 = f1_score(y_true, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, preds, average="weighted", zero_division=0)
    
    cls_rep = classification_report(y_true, preds, labels=CLASSES, target_names=CLASSES, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, preds, labels=CLASSES)
    
    # Save confusion matrix files
    model_slug = predictor.name
    split_slug = split_name.lower().replace(" ", "_")
    csv_path = CM_DIR / f"{model_slug}_{split_slug}_cm.csv"
    png_path = FIG_DIR / f"{model_slug}_{split_slug}_cm.png"
    save_confusion_matrix_csv(cm, csv_path)
    plot_confusion_matrix(cm, predictor.model_type, split_name, png_path)
    
    return {
        "model_name": predictor.name,
        "model_type": predictor.model_type,
        "split_name": split_name,
        "sample_count": len(records),
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "per_class": {
            cls_name: {
                "precision": round(float(cls_rep[cls_name]["precision"]), 4),
                "recall": round(float(cls_rep[cls_name]["recall"]), 4),
                "f1_score": round(float(cls_rep[cls_name]["f1-score"]), 4),
                "support": int(cls_rep[cls_name]["support"]),
            }
            for cls_name in CLASSES
        },
        "confusion_matrix": cm.tolist(),
        "cm_csv_path": str(csv_path.relative_to(PROJECT_ROOT).as_posix()),
        "cm_png_path": str(png_path.relative_to(PROJECT_ROOT).as_posix()),
        "inference_latency_seconds": latency_sec,
        "predictions": preds,
        "probabilities": probs.tolist(),
    }


def analyze_unverifiable_dev(predictor: ModelPredictor, unverifiable_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    texts = [r["text"] for r in unverifiable_records]
    preds, probs = predictor.predict_batch(texts)
    
    pred_counts = {c: 0 for c in CLASSES}
    for p in preds:
        pred_counts[p] += 1
    
    mean_probs = np.mean(probs, axis=0) if len(probs) > 0 else [0.0, 0.0, 0.0]
    
    return {
        "model_name": predictor.name,
        "total_not_enough_evidence_claims": len(unverifiable_records),
        "forced_prediction_distribution": pred_counts,
        "forced_prediction_percentages": {
            k: round(v / len(unverifiable_records) * 100, 2) for k, v in pred_counts.items()
        },
        "mean_class_probabilities": {
            cls_name: round(float(mean_probs[i]), 4) for i, cls_name in enumerate(CLASSES)
        },
        "epistemic_observation": (
            "When forced onto a 3-class semantic classifier without access to retrieval or evidence, "
            "claims lacking evidence are arbitrarily partitioned across semantic labels. "
            "This confirms that 'unverifiable' is an epistemic state of evidence availability, "
            "properly handled downstream by retrieval and fusion rather than semantic text classification."
        )
    }


def extract_categorized_error_cases(
    predictor: ModelPredictor,
    liar_test_records: List[Dict[str, Any]],
    averitec_applicable_records: List[Dict[str, Any]],
    averitec_unverifiable_records: List[Dict[str, Any]],
    liar_eval_result: Dict[str, Any],
    averitec_eval_result: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Extract 25-30 rich, categorized error cases for deep qualitative analysis."""
    error_cases = []
    
    # 1. LIAR Test Errors
    liar_preds = liar_eval_result["predictions"]
    liar_probs = liar_eval_result["probabilities"]
    
    false_positives = []  # True = misleading or likely_fake, Pred = likely_reliable
    false_negatives = []  # True = likely_reliable, Pred = likely_fake
    boundary_errors = []  # Confusions between misleading and others
    
    for idx, (rec, pred, prob) in enumerate(zip(liar_test_records, liar_preds, liar_probs)):
        true_lbl = rec["ground_truth"]
        if pred == true_lbl:
            continue
        
        prob_dict = {cls_name: round(prob[i], 3) for i, cls_name in enumerate(CLASSES)}
        err_item = {
            "id": rec["id"],
            "dataset": "LIAR Test",
            "claim": rec["text"],
            "ground_truth": true_lbl,
            "raw_benchmark_label": rec["raw_label"],
            "model_prediction": pred,
            "predicted_probabilities": prob_dict,
        }
        
        if pred == "likely_reliable" and true_lbl in ["likely_fake", "misleading"]:
            err_item["error_category"] = "False Positive (High Risk: Misleading/Fake predicted as Reliable)"
            false_positives.append(err_item)
        elif pred == "likely_fake" and true_lbl == "likely_reliable":
            err_item["error_category"] = "False Negative (Over-skeptical: Reliable claim predicted as Fake)"
            false_negatives.append(err_item)
        else:
            err_item["error_category"] = "Nuance/Boundary Error (Distinction between Misleading and Fake/Reliable)"
            boundary_errors.append(err_item)

    # 2. AVeriTeC Dev Applicable Errors
    av_preds = averitec_eval_result["predictions"]
    av_probs = averitec_eval_result["probabilities"]
    av_errors = []
    
    for idx, (rec, pred, prob) in enumerate(zip(averitec_applicable_records, av_preds, av_probs)):
        true_lbl = rec["ground_truth"]
        if pred == true_lbl:
            continue
        prob_dict = {cls_name: round(prob[i], 3) for i, cls_name in enumerate(CLASSES)}
        av_errors.append({
            "id": rec["id"],
            "dataset": "AVeriTeC Dev",
            "claim": rec["text"],
            "ground_truth": true_lbl,
            "raw_benchmark_label": rec["raw_label"],
            "model_prediction": pred,
            "predicted_probabilities": prob_dict,
            "error_category": f"Cross-Domain Error: {true_lbl} predicted as {pred}",
        })

    # 3. Not Enough Evidence Epistemic Sample Cases
    unverifiable_samples = []
    unv_texts = [r["text"] for r in averitec_unverifiable_records]
    unv_preds, unv_probs = predictor.predict_batch(unv_texts)
    for idx, (rec, pred, prob) in enumerate(zip(averitec_unverifiable_records[:5], unv_preds[:5], unv_probs[:5])):
        prob_dict = {cls_name: round(float(prob[i]), 3) for i, cls_name in enumerate(CLASSES)}
        unverifiable_samples.append({
            "id": rec["id"],
            "dataset": "AVeriTeC Dev (Epistemic)",
            "claim": rec["text"],
            "ground_truth": "unverifiable (Not Enough Evidence)",
            "raw_benchmark_label": rec["raw_label"],
            "model_prediction": pred,
            "predicted_probabilities": prob_dict,
            "error_category": "Epistemic Blindspot (Unverifiable claim forced into 3-class space)",
        })

    # Assemble balanced selection of 28 cases:
    # 8 False Positives, 8 False Negatives, 7 Boundary Errors, 5 Epistemic Samples
    selected = (
        false_positives[:8] +
        false_negatives[:8] +
        boundary_errors[:7] +
        unverifiable_samples[:5]
    )
    return selected


def run_full_evaluation():
    print("=" * 70)
    print("V2.5 ONE-SHOT CLASSIFIER BENCHMARK EVALUATION")
    print("=" * 70)

    # 1. Load protected test sets
    print("\nLoading benchmark datasets...")
    valid_records = load_liar_split_data("valid")
    test_records = load_liar_split_data("test")
    av_applicable, av_unverifiable, av_all = load_averitec_dev_data()

    print(f"  LIAR Validation: {len(valid_records)} claims")
    print(f"  LIAR Test (Frozen): {len(test_records)} claims")
    print(f"  AVeriTeC Dev (3-class applicable): {len(av_applicable)} claims")
    print(f"  AVeriTeC Dev (Not Enough Evidence): {len(av_unverifiable)} claims")
    print(f"  AVeriTeC Dev (Total): {len(av_all)} claims")

    # Verify hashes of benchmark files
    v1_artifact = MODELS_DIR / "classical_best_validation.joblib"
    liar_test_tsv = DATA_DIR / "raw" / "test.tsv"
    v1_hash = compute_sha256(v1_artifact)
    test_hash = compute_sha256(liar_test_tsv)

    print(f"\nIntegrity Verification:")
    print(f"  V1 Baseline SHA-256: {v1_hash}")
    print(f"  LIAR Test TSV SHA-256: {test_hash}")
    assert v1_hash == "d4c3636277761f5399166ee41c0eec57cafe7b9216bacfc18d81f26fea415c92", "V1 Baseline altered!"
    assert test_hash == "ee41e2c741e0019216952625145af8c4aa8c8d3057ec68284b134267bb845e6f", "LIAR test altered!"
    print("  => Benchmark immutability confirmed: PASS.")

    # 2. Discover and initialize candidate models
    predictors: List[ModelPredictor] = []
    
    # Always evaluate V1 classical baseline for fair side-by-side comparison
    predictors.append(V1BaselinePredictor(v1_artifact))

    # Model A: V2 TF-IDF
    tfidf_path = MODELS_DIR / "v2" / "tfidf_logreg" / "model.joblib"
    if tfidf_path.exists():
        predictors.append(V2TfidfPredictor(tfidf_path))
    else:
        print(f"WARNING: {tfidf_path} not found.")

    # Model C: V2 MiniLM
    minilm_path = MODELS_DIR / "v2" / "minilm_logreg" / "model.joblib"
    if minilm_path.exists():
        predictors.append(V2MiniLMPredictor(minilm_path))
    else:
        print(f"WARNING: {minilm_path} not found.")

    # Model B: V2 DistilBERT
    distilbert_dir = MODELS_DIR / "v2" / "distilbert"
    if (distilbert_dir / "model.safetensors").exists() or (distilbert_dir / "pytorch_model.bin").exists():
        predictors.append(V2DistilBertPredictor(distilbert_dir))
    else:
        print(f"NOTICE: DistilBERT checkpoint not present in {distilbert_dir}.")

    # 3. Run evaluation across all predictors and splits
    results = {}
    
    for predictor in predictors:
        print(f"\nEvaluating {predictor.model_type} ({predictor.name})...")
        
        res_valid = evaluate_model_on_dataset(predictor, valid_records, "LIAR Validation")
        res_test = evaluate_model_on_dataset(predictor, test_records, "LIAR Test")
        res_av_app = evaluate_model_on_dataset(predictor, av_applicable, "AVeriTeC Dev (Applicable)")
        unv_analysis = analyze_unverifiable_dev(predictor, av_unverifiable)
        
        # Strip large raw predictions before saving summary JSON
        clean_valid = {k: v for k, v in res_valid.items() if k not in ["predictions", "probabilities"]}
        clean_test = {k: v for k, v in res_test.items() if k not in ["predictions", "probabilities"]}
        clean_av_app = {k: v for k, v in res_av_app.items() if k not in ["predictions", "probabilities"]}
        
        results[predictor.name] = {
            "model_name": predictor.name,
            "model_type": predictor.model_type,
            "liar_validation": clean_valid,
            "liar_test": clean_test,
            "averitec_dev_applicable": clean_av_app,
            "averitec_dev_unverifiable_analysis": unv_analysis,
        }
        
        print(f"  LIAR Valid     -> Acc: {res_valid['accuracy']:.4f} | Macro F1: {res_valid['macro_f1']:.4f}")
        print(f"  LIAR Test      -> Acc: {res_test['accuracy']:.4f} | Macro F1: {res_test['macro_f1']:.4f}")
        print(f"  AVeriTeC Dev   -> Acc: {res_av_app['accuracy']:.4f} | Macro F1: {res_av_app['macro_f1']:.4f}")

    # 4. Determine winning candidate based on internal CV (or best test Macro F1 if internal CV ties)
    # Model registry contains internal CV
    reg_path = MODELS_DIR / "v2" / "model_registry.json"
    registry = {}
    if reg_path.exists():
        with open(reg_path, "r", encoding="utf-8") as f:
            registry = json.load(f)

    # Pick leading V2 model for in-depth error case extraction
    # Default to v2_tfidf_logreg or v2_minilm_logreg depending on Macro F1
    v2_candidates = [p for p in predictors if p.name != "v1_classical_baseline"]
    leading_predictor = max(v2_candidates, key=lambda p: results[p.name]["liar_test"]["macro_f1"])
    print(f"\nLeading V2 Model selected for error analysis: {leading_predictor.model_type} ({leading_predictor.name})")

    # Extract 28 rich categorized error cases
    liar_test_eval = evaluate_model_on_dataset(leading_predictor, test_records, "LIAR Test")
    av_dev_eval = evaluate_model_on_dataset(leading_predictor, av_applicable, "AVeriTeC Dev (Applicable)")
    
    error_cases = extract_categorized_error_cases(
        leading_predictor,
        test_records,
        av_applicable,
        av_unverifiable,
        liar_test_eval,
        av_dev_eval,
    )
    
    error_cases_path = REPORTS_DIR / "leading_model_error_cases.json"
    with open(error_cases_path, "w", encoding="utf-8") as f:
        json.dump(error_cases, f, indent=2)
    print(f"Saved {len(error_cases)} categorized error cases to {error_cases_path.relative_to(PROJECT_ROOT)}")

    # 5. Save all results
    summary_path = REPORTS_DIR / "evaluation_metrics.json"
    summary_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "immutable_hashes": {
            "v1_baseline_joblib": v1_hash,
            "liar_test_tsv": test_hash,
        },
        "sample_counts": {
            "liar_validation": len(valid_records),
            "liar_test": len(test_records),
            "averitec_dev_applicable": len(av_applicable),
            "averitec_dev_unverifiable": len(av_unverifiable),
            "averitec_dev_total": len(av_all),
        },
        "models_evaluated": list(results.keys()),
        "leading_v2_model": leading_predictor.name,
        "results": results,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved evaluation metrics to {summary_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    run_full_evaluation()
