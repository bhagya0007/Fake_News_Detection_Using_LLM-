"""Build classifier-specific training pool excluding unverifiable claims.

Step V2.5:
Source: data/v2_processed/combined_eligible_train_pool.jsonl
Target: data/v2_processed/classifier_train_pool.jsonl

Includes only:
- likely_reliable
- misleading
- likely_fake

Excludes:
- unverifiable (266 claims from AVeriTeC Not Enough Evidence / unverifiable)
"""

from collections import Counter
import hashlib
import json
from pathlib import Path


def build_classifier_pool() -> dict:
    src_path = Path("data/v2_processed/combined_eligible_train_pool.jsonl")
    dst_path = Path("data/v2_processed/classifier_train_pool.jsonl")

    allowed_classes = {"likely_reliable", "misleading", "likely_fake"}

    total_read = 0
    excluded_unverifiable = 0
    records = []
    class_counts: Counter = Counter()
    dataset_counts: Counter = Counter()
    dataset_class_counts: Counter = Counter()

    with open(src_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_read += 1
            item = json.loads(line)
            label = item.get("harmonized_label")
            if label == "unverifiable":
                excluded_unverifiable += 1
                continue
            if label in allowed_classes:
                records.append(item)
                class_counts[label] += 1
                dataset_counts[item.get("source_dataset")] += 1
                dataset_class_counts[(item.get("source_dataset"), label)] += 1
            else:
                raise ValueError(f"Unexpected label found: {label}")

    with open(dst_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(dst_path, "rb") as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()

    stats = {
        "total_read": total_read,
        "excluded_unverifiable": excluded_unverifiable,
        "classifier_pool_size": len(records),
        "class_counts": dict(class_counts),
        "class_percentages": {
            k: round(v / len(records) * 100, 2) for k, v in class_counts.items()
        },
        "dataset_counts": dict(dataset_counts),
        "dataset_class_counts": {
            f"{k[0]}_{k[1]}": v for k, v in dataset_class_counts.items()
        },
        "sha256": sha256,
        "path": str(dst_path.as_posix()),
    }

    print(json.dumps(stats, indent=2))
    return stats


if __name__ == "__main__":
    build_classifier_pool()
