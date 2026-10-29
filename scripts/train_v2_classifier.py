"""V2 Text Classifier Training Script (Step V2.5).

Supports training and model selection across:
- Model A: TF-IDF + Logistic Regression (V2 classical baseline)
- Model C: Frozen all-MiniLM-L6-v2 embeddings + Logistic Regression (semantic baseline)
- Model B: distilbert-base-uncased sequence classification (transformer baseline)

Key Constraints:
- Label space: EXACTLY 3 classes ('likely_reliable', 'misleading', 'likely_fake')
- Excludes 'unverifiable' (epistemic/retrieval state, never predicted by classifier)
- Training pool: data/v2_processed/classifier_train_pool.jsonl
- Stratified 5-Fold CV on (source_dataset + harmonized_label)
- ZERO Groq calls, ZERO GDELT calls
- Frozen baseline models/classical_best_validation.joblib remains 100% UNTOUCHED
"""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "v2_processed" / "classifier_train_pool.jsonl"
MODELS_V2_DIR = PROJECT_ROOT / "models" / "v2"
REGISTRY_PATH = MODELS_V2_DIR / "model_registry.json"

# Strict 3-class label mapping
LABEL_ENCODING = {
    "likely_reliable": 0,
    "misleading": 1,
    "likely_fake": 2,
}
ID2LABEL = {v: k for k, v in LABEL_ENCODING.items()}
ALLOWED_CLASSES = set(LABEL_ENCODING.keys())

RANDOM_SEED = 42


def set_seed(seed: int = RANDOM_SEED) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def compute_file_sha256(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_classifier_training_data() -> Tuple[List[Dict[str, Any]], str]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Training pool not found at: {DATA_PATH}")

    pool_hash = compute_file_sha256(DATA_PATH)
    records = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            label = item.get("harmonized_label")
            if label not in ALLOWED_CLASSES:
                raise ValueError(
                    f"Illegal label '{label}' found in classifier pool. "
                    f"Allowed: {ALLOWED_CLASSES}"
                )
            records.append(item)

    return records, pool_hash


def get_stratified_folds(
    records: List[Dict[str, Any]], n_splits: int = 5
) -> List[Tuple[np.ndarray, np.ndarray, Dict[str, Any]]]:
    strata = [f"{r['source_dataset']}_{r['harmonized_label']}" for r in records]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)

    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(records, strata), 1):
        val_records = [records[i] for i in val_idx]
        stats = {
            "fold": fold_idx,
            "train_samples": len(train_idx),
            "val_samples": len(val_idx),
            "liar_val": sum(1 for r in val_records if r["source_dataset"] == "liar"),
            "averitec_val": sum(1 for r in val_records if r["source_dataset"] == "averitec"),
            "class_counts_val": dict(Counter(r["harmonized_label"] for r in val_records)),
        }
        folds.append((train_idx, val_idx, stats))
    return folds


def train_tfidf_candidates(
    records: List[Dict[str, Any]], folds: List[Tuple[np.ndarray, np.ndarray, Dict[str, Any]]]
) -> Dict[str, Any]:
    print("\n--- Training Model A: V2 TF-IDF + Logistic Regression Candidates ---")
    texts = np.array([r["text"] for r in records])
    labels = np.array([LABEL_ENCODING[r["harmonized_label"]] for r in records])

    candidate_grid = [
        {"ngram_range": (1, 1), "C": 1.0, "class_weight": None, "sublinear_tf": True},
        {"ngram_range": (1, 2), "C": 1.0, "class_weight": None, "sublinear_tf": True},
        {"ngram_range": (1, 2), "C": 0.5, "class_weight": None, "sublinear_tf": True},
        {"ngram_range": (1, 2), "C": 2.0, "class_weight": None, "sublinear_tf": True},
        {"ngram_range": (1, 2), "C": 1.0, "class_weight": "balanced", "sublinear_tf": True},
    ]

    best_macro_f1 = -1.0
    best_config = None
    best_cv_metrics = None

    for config in candidate_grid:
        fold_f1s = []
        fold_accs = []
        t0 = time.perf_counter()

        for fold_num, (train_idx, val_idx, _) in enumerate(folds, 1):
            pipe = Pipeline([
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=config["ngram_range"],
                        sublinear_tf=config["sublinear_tf"],
                        max_features=25000,
                        lowercase=True,
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        C=config["C"],
                        class_weight=config["class_weight"],
                        random_state=RANDOM_SEED,
                        max_iter=1000,
                    ),
                ),
            ])

            pipe.fit(texts[train_idx], labels[train_idx])
            preds = pipe.predict(texts[val_idx])

            f1 = f1_score(labels[val_idx], preds, average="macro")
            acc = accuracy_score(labels[val_idx], preds)
            fold_f1s.append(f1)
            fold_accs.append(acc)

        mean_f1 = float(np.mean(fold_f1s))
        std_f1 = float(np.std(fold_f1s))
        mean_acc = float(np.mean(fold_accs))
        elapsed = round(time.perf_counter() - t0, 2)

        print(
            f"Config {config}: Mean CV Macro F1 = {mean_f1:.4f} (+/- {std_f1:.4f}), "
            f"Acc = {mean_acc:.4f} ({elapsed}s)"
        )

        if mean_f1 > best_macro_f1:
            best_macro_f1 = mean_f1
            best_config = config
            best_cv_metrics = {
                "mean_macro_f1": round(mean_f1, 4),
                "std_macro_f1": round(std_f1, 4),
                "mean_accuracy": round(mean_acc, 4),
                "fold_f1s": [round(x, 4) for x in fold_f1s],
                "fold_accuracies": [round(x, 4) for x in fold_accs],
            }

    print(f"\nBest TF-IDF Config Selected: {best_config} with CV Macro F1 = {best_macro_f1:.4f}")

    # Retrain selected best config on full training pool
    print("Fitting selected best TF-IDF model on full classifier pool (11,926 samples)...")
    final_pipe = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=best_config["ngram_range"],
                sublinear_tf=best_config["sublinear_tf"],
                max_features=25000,
                lowercase=True,
            ),
        ),
        (
            "clf",
            LogisticRegression(
                C=best_config["C"],
                class_weight=best_config["class_weight"],
                random_state=RANDOM_SEED,
                max_iter=1000,
            ),
        ),
    ])

    t_train_start = time.perf_counter()
    final_pipe.fit(texts, labels)
    train_duration = round(time.perf_counter() - t_train_start, 2)

    # Save artifact
    out_dir = MODELS_V2_DIR / "tfidf_logreg"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = out_dir / "model.joblib"
    joblib.dump(final_pipe, artifact_path)
    artifact_hash = compute_file_sha256(artifact_path)

    metadata = {
        "model_name": "v2_tfidf_logistic_regression",
        "model_type": "TF-IDF + Logistic Regression",
        "artifact_path": str(artifact_path.as_posix()),
        "artifact_sha256": artifact_hash,
        "label_encoding": LABEL_ENCODING,
        "num_classes": 3,
        "classes": list(LABEL_ENCODING.keys()),
        "hyperparameters": best_config,
        "cv_strategy": "Stratified 5-Fold CV on (source_dataset + harmonized_label)",
        "cv_metrics": best_cv_metrics,
        "training_samples": len(records),
        "training_duration_seconds": train_duration,
        "random_seed": RANDOM_SEED,
    }

    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def train_minilm_candidates(
    records: List[Dict[str, Any]], folds: List[Tuple[np.ndarray, np.ndarray, Dict[str, Any]]]
) -> Dict[str, Any]:
    print("\n--- Training Model C: MiniLM Frozen Embeddings + Logistic Regression ---")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("SentenceTransformers not installed; skipping MiniLM candidate.")
        return {}

    texts = [r["text"] for r in records]
    labels = np.array([LABEL_ENCODING[r["harmonized_label"]] for r in records])

    print("Encoding 11,926 claim texts using sentence-transformers/all-MiniLM-L6-v2...")
    t0_enc = time.perf_counter()
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = embedder.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    enc_duration = round(time.perf_counter() - t0_enc, 2)
    print(f"MiniLM embedding completed in {enc_duration}s. Shape: {embeddings.shape}")

    configs = [
        {"C": 0.5, "class_weight": None},
        {"C": 1.0, "class_weight": None},
        {"C": 2.0, "class_weight": None},
        {"C": 1.0, "class_weight": "balanced"},
    ]

    best_macro_f1 = -1.0
    best_config = None
    best_cv_metrics = None

    for config in configs:
        fold_f1s = []
        fold_accs = []
        for fold_num, (train_idx, val_idx, _) in enumerate(folds, 1):
            clf = LogisticRegression(
                C=config["C"],
                class_weight=config["class_weight"],
                random_state=RANDOM_SEED,
                max_iter=1000,
            )
            clf.fit(embeddings[train_idx], labels[train_idx])
            preds = clf.predict(embeddings[val_idx])
            fold_f1s.append(f1_score(labels[val_idx], preds, average="macro"))
            fold_accs.append(accuracy_score(labels[val_idx], preds))

        mean_f1 = float(np.mean(fold_f1s))
        std_f1 = float(np.std(fold_f1s))
        mean_acc = float(np.mean(fold_accs))

        print(
            f"Config {config}: Mean CV Macro F1 = {mean_f1:.4f} (+/- {std_f1:.4f}), "
            f"Acc = {mean_acc:.4f}"
        )

        if mean_f1 > best_macro_f1:
            best_macro_f1 = mean_f1
            best_config = config
            best_cv_metrics = {
                "mean_macro_f1": round(mean_f1, 4),
                "std_macro_f1": round(std_f1, 4),
                "mean_accuracy": round(mean_acc, 4),
                "fold_f1s": [round(x, 4) for x in fold_f1s],
                "fold_accuracies": [round(x, 4) for x in fold_accs],
            }

    print(f"\nBest MiniLM Config: {best_config} with CV Macro F1 = {best_macro_f1:.4f}")

    # Retrain on full pool
    final_clf = LogisticRegression(
        C=best_config["C"],
        class_weight=best_config["class_weight"],
        random_state=RANDOM_SEED,
        max_iter=1000,
    )
    t_train_start = time.perf_counter()
    final_clf.fit(embeddings, labels)
    train_duration = round(time.perf_counter() - t_train_start, 2)

    # Save artifact
    out_dir = MODELS_V2_DIR / "minilm_logreg"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = out_dir / "model.joblib"
    joblib.dump(final_clf, artifact_path)
    artifact_hash = compute_file_sha256(artifact_path)

    metadata = {
        "model_name": "v2_minilm_logistic_regression",
        "model_type": "all-MiniLM-L6-v2 Frozen Embeddings + Logistic Regression",
        "artifact_path": str(artifact_path.as_posix()),
        "artifact_sha256": artifact_hash,
        "label_encoding": LABEL_ENCODING,
        "num_classes": 3,
        "classes": list(LABEL_ENCODING.keys()),
        "hyperparameters": best_config,
        "cv_strategy": "Stratified 5-Fold CV on (source_dataset + harmonized_label)",
        "cv_metrics": best_cv_metrics,
        "training_samples": len(records),
        "embedding_duration_seconds": enc_duration,
        "training_duration_seconds": train_duration,
        "random_seed": RANDOM_SEED,
    }

    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def train_distilbert_candidate(
    records: List[Dict[str, Any]], folds: List[Tuple[np.ndarray, np.ndarray, Dict[str, Any]]]
) -> Dict[str, Any]:
    print("\n--- Training Model B: DistilBERT Sequence Classification ---")
    try:
        import torch
        from torch.optim import AdamW
        from torch.utils.data import DataLoader, Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            get_linear_schedule_with_warmup,
        )
    except Exception as e:
        print(f"DistilBERT dependencies not available: {e}. Status: TRAINING NOT COMPLETED.")
        return {"status": "TRAINING NOT COMPLETED", "reason": str(e)}

    # Hardware check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target compute device: {device}")

    # Use internal Fold 1 as validation (2,386 samples) and Folds 2-5 as training (9,540 samples)
    train_idx, val_idx, val_stats = folds[0][0], folds[0][1], folds[0][2]
    # Note: folds[0][1] is fold 1 validation; folds[0][0] is folds 2-5 training
    print(
        f"Internal Train Partition: {len(train_idx)} samples | "
        f"Internal Validation Partition (Fold 1): {len(val_idx)} samples"
    )

    texts = [r["text"] for r in records]
    labels = [LABEL_ENCODING[r["harmonized_label"]] for r in records]

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    MAX_LENGTH = 128  # Verified 0.0% truncation across all 11,926 claims

    class TextClassificationDataset(Dataset):
        def __init__(self, txts: List[str], lbls: List[int]):
            self.encodings = tokenizer(
                txts,
                truncation=True,
                padding=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            self.labels = torch.tensor(lbls, dtype=torch.long)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            item = {key: val[idx] for key, val in self.encodings.items()}
            item["labels"] = self.labels[idx]
            return item

    train_dataset = TextClassificationDataset(
        [texts[i] for i in train_idx], [labels[i] for i in train_idx]
    )
    val_dataset = TextClassificationDataset(
        [texts[i] for i in val_idx], [labels[i] for i in val_idx]
    )

    BATCH_SIZE = 16
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL_ENCODING,
    )
    model.to(device)

    EPOCHS = 2
    LEARNING_RATE = 3e-5
    WEIGHT_DECAY = 0.01

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps
    )

    print(
        f"Training config: epochs={EPOCHS}, batch_size={BATCH_SIZE}, lr={LEARNING_RATE}, "
        f"max_length={MAX_LENGTH}, total_steps={total_steps}"
    )

    epoch_logs = []
    best_val_macro_f1 = -1.0
    t_start = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_train_loss = 0.0
        t_epoch_start = time.perf_counter()

        for step, batch in enumerate(train_loader, 1):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=batch_labels
            )
            loss = outputs.loss
            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            if step % 150 == 0 or step == len(train_loader):
                print(
                    f"Epoch {epoch}/{EPOCHS} [Step {step}/{len(train_loader)}] "
                    f"Train Loss: {loss.item():.4f}"
                )

        avg_train_loss = total_train_loss / len(train_loader)

        # Evaluate on validation partition
        model.eval()
        val_preds = []
        val_targets = []
        total_val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                batch_labels = batch["labels"].to(device)

                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=batch_labels
                )
                total_val_loss += outputs.loss.item()
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(batch_labels.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = accuracy_score(val_targets, val_preds)
        val_macro_f1 = f1_score(val_targets, val_preds, average="macro")
        cls_report = classification_report(
            val_targets, val_preds, target_names=[ID2LABEL[i] for i in range(3)], output_dict=True
        )
        epoch_sec = round(time.perf_counter() - t_epoch_start, 2)

        log = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "val_accuracy": round(val_acc, 4),
            "val_macro_f1": round(val_macro_f1, 4),
            "per_class_f1": {
                name: round(cls_report[name]["f1-score"], 4)
                for name in ["likely_reliable", "misleading", "likely_fake"]
            },
            "epoch_duration_seconds": epoch_sec,
        }
        epoch_logs.append(log)
        print(
            f"Epoch {epoch} Result: Val Macro F1 = {val_macro_f1:.4f}, Val Acc = {val_acc:.4f}, "
            f"Val Loss = {avg_val_loss:.4f} ({epoch_sec}s)"
        )

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1

    total_duration = round(time.perf_counter() - t_start, 2)

    # Save artifact
    out_dir = MODELS_V2_DIR / "distilbert"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    # Checksum of pytorch_model.bin / model.safetensors
    weights_path = out_dir / "model.safetensors"
    if not weights_path.exists():
        weights_path = out_dir / "pytorch_model.bin"
    artifact_hash = compute_file_sha256(weights_path) if weights_path.exists() else "unknown"

    metadata = {
        "model_name": "v2_distilbert_sequence_classifier",
        "model_type": "distilbert-base-uncased Sequence Classification",
        "artifact_dir": str(out_dir.as_posix()),
        "weights_sha256": artifact_hash,
        "label_encoding": LABEL_ENCODING,
        "num_classes": 3,
        "classes": list(LABEL_ENCODING.keys()),
        "hyperparameters": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "max_length": MAX_LENGTH,
            "optimizer": "AdamW",
            "device": str(device),
        },
        "validation_strategy": "Internal Fold 1 Validation Split (Source-Aware)",
        "internal_validation_macro_f1": best_val_macro_f1,
        "epoch_logs": epoch_logs,
        "training_samples": len(train_idx),
        "validation_samples": len(val_idx),
        "total_training_duration_seconds": total_duration,
        "random_seed": RANDOM_SEED,
    }

    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Train V2 Text Classifiers")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all", "tfidf", "minilm-logreg", "distilbert"],
        help="Model candidate to train",
    )
    args = parser.parse_args()

    set_seed(RANDOM_SEED)
    records, pool_hash = load_classifier_training_data()
    print(f"Loaded {len(records)} records from classifier pool.")
    print(f"Training Pool SHA-256: {pool_hash}")

    folds = get_stratified_folds(records, n_splits=5)
    print(f"Formed {len(folds)} stratified folds.")
    for f in folds:
        print(f"  Fold {f[2]['fold']}: Val {f[2]['val_samples']} (LIAR {f[2]['liar_val']}, AVeriTeC {f[2]['averitec_val']})")

    MODELS_V2_DIR.mkdir(parents=True, exist_ok=True)
    registry: Dict[str, Any] = {}
    if REGISTRY_PATH.exists():
        try:
            with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
                registry = json.load(f)
        except Exception:
            registry = {}

    registry["training_pool_sha256"] = pool_hash
    registry["training_pool_samples"] = len(records)
    registry["random_seed"] = RANDOM_SEED
    registry["label_encoding"] = LABEL_ENCODING

    if args.model in ("all", "tfidf"):
        tfidf_meta = train_tfidf_candidates(records, folds)
        registry["v2_tfidf_logistic_regression"] = tfidf_meta

    if args.model in ("all", "minilm-logreg"):
        minilm_meta = train_minilm_candidates(records, folds)
        registry["v2_minilm_logistic_regression"] = minilm_meta

    if args.model in ("all", "distilbert"):
        distilbert_meta = train_distilbert_candidate(records, folds)
        registry["v2_distilbert_sequence_classifier"] = distilbert_meta

    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)

    print(f"\nTraining complete. Model registry updated at: {REGISTRY_PATH}")


if __name__ == "__main__":
    main()
