"""Analyze token length distribution for DistilBERT tokenizer on classifier_train_pool.jsonl.

Measures:
- median
- p90
- p95
- p99
- max
- truncation percentage at lengths 64, 128, 256, 512.
"""

import json
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer


def analyze_token_lengths():
    data_path = Path("data/v2_processed/classifier_train_pool.jsonl")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    lengths = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            tokens = tokenizer.encode(item["text"], add_special_tokens=True)
            lengths.append(len(tokens))

    arr = np.array(lengths)
    median_len = float(np.median(arr))
    p90 = float(np.percentile(arr, 90))
    p95 = float(np.percentile(arr, 95))
    p99 = float(np.percentile(arr, 99))
    max_len = int(np.max(arr))
    min_len = int(np.min(arr))
    mean_len = float(np.mean(arr))

    trunc_64 = float(np.mean(arr > 64) * 100)
    trunc_128 = float(np.mean(arr > 128) * 100)
    trunc_256 = float(np.mean(arr > 256) * 100)
    trunc_512 = float(np.mean(arr > 512) * 100)

    res = {
        "sample_count": len(lengths),
        "min": min_len,
        "mean": round(mean_len, 2),
        "median": median_len,
        "p90": p90,
        "p95": p95,
        "p99": p99,
        "max": max_len,
        "truncation_at_64_pct": round(trunc_64, 3),
        "truncation_at_128_pct": round(trunc_128, 3),
        "truncation_at_256_pct": round(trunc_256, 3),
        "truncation_at_512_pct": round(trunc_512, 3),
        "recommended_max_length": 128 if trunc_128 < 1.0 else 256,
    }

    print(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    analyze_token_lengths()
