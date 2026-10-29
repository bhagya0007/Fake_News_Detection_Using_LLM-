"""Master Version 2 Final Scientific & Safety Evaluation Runner (V2.6).

Single unified entrypoint to execute the complete evaluation suite of Version 2:
- Track C: Controlled Claim-Evidence Stance Benchmark (35 cases)
- Track D: Controlled End-to-End Hybrid Pipeline Benchmark (48 cases)
- Track F: Latency Microbenchmarking & Resource Profiling
- Track G: Resilience & Failure-Mode Auditing (9 edge cases)
- Integrity: SHA-256 Cryptographic Hash Verification of Frozen Artifacts
- Optional Track E: Real-World Live Pipeline Demonstration (--live)

Default execution mode is 100% offline (0 Groq calls, 0 GDELT calls).

Usage:
    python scripts/run_v2_final_evaluation.py           # Default offline mode
    python scripts/run_v2_final_evaluation.py --offline # Explicit offline mode
    python scripts/run_v2_final_evaluation.py --live    # Includes Track E live demo
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure Windows console encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PYTHON_EXE = sys.executable


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def verify_frozen_integrity() -> bool:
    print("\n" + "=" * 75)
    print("STEP 0: CRYPTOGRAPHIC INTEGRITY VERIFICATION OF FROZEN BASELINES")
    print("=" * 75)
    manifest_path = PROJECT_ROOT / "reports" / "v2_final" / "artifact_manifest.json"
    if not manifest_path.exists():
        print(f"Artifact manifest not found at: {manifest_path}")
        return False

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    all_intact = True
    for entry in manifest.get("frozen_artifacts", []):
        rel_path = entry["relative_path"]
        expected_sha = entry["sha256"]
        p = PROJECT_ROOT / rel_path
        if not p.exists():
            print(f"[MISSING] {rel_path}")
            all_intact = False
            continue
        actual_sha = compute_sha256(p)
        if actual_sha == expected_sha:
            print(f"[OK] {rel_path} -> {actual_sha[:16]}... (INTACT)")
        else:
            print(f"[FAILED] {rel_path} SHA-256 mismatch!")
            print(f"  Expected: {expected_sha}")
            print(f"  Actual:   {actual_sha}")
            all_intact = False

    return all_intact


def run_command_stream(cmd: list[str], description: str) -> bool:
    print("\n" + "=" * 75)
    print(f"RUNNING: {description}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 75)
    t0 = time.perf_counter()
    ret = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    elapsed = round(time.perf_counter() - t0, 2)
    if ret.returncode == 0:
        print(f"[{description} PASSED] in {elapsed}s")
        return True
    else:
        print(f"[{description} FAILED] exit code {ret.returncode} in {elapsed}s")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="V2.6 Master Evaluation Runner")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Include Track E live real-world demonstration (contacts Groq and GDELT).",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        default=True,
        help="Run strict offline evaluation (default: 0 Groq calls, 0 GDELT calls).",
    )
    args = parser.parse_args()

    start_all = time.perf_counter()
    print("=" * 75)
    print("VERSION 2 FINAL SCIENTIFIC & SAFETY EVALUATION SUITE (V2.6)")
    print(f"Timestamp: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"Execution Mode: {'LIVE DEMO ENABLED' if args.live else 'OFFLINE STRICT (0 Groq, 0 GDELT)'}")
    print("=" * 75)

    # Step 0: Integrity Check
    integrity_ok = verify_frozen_integrity()
    if not integrity_ok:
        print("[ABORT] Cryptographic integrity verification failed! Halting evaluation.")
        sys.exit(1)

    # Step 1: Track C (Controlled Stance Benchmark)
    track_c_ok = run_command_stream(
        [PYTHON_EXE, str(PROJECT_ROOT / "scripts" / "evaluate_v2_stance.py")],
        "Track C: Controlled Stance Benchmark (35 cases)",
    )

    # Step 2: Track D (Controlled End-to-End Hybrid Benchmark)
    track_d_ok = run_command_stream(
        [PYTHON_EXE, str(PROJECT_ROOT / "scripts" / "evaluate_v2_end_to_end.py")],
        "Track D: Controlled End-to-End Hybrid Benchmark (48 cases)",
    )

    # Step 3: Track F & Track G (Latency Profiling & Resilience Audit)
    track_fg_ok = run_command_stream(
        [PYTHON_EXE, str(PROJECT_ROOT / "scripts" / "evaluate_v2_latency_resilience.py")],
        "Track F & G: Latency Profiling & Resilience Failure-Mode Auditing",
    )

    # Step 4: Optional Track E (Live Demonstration)
    track_e_ok = True
    if args.live:
        track_e_ok = run_command_stream(
            [PYTHON_EXE, str(PROJECT_ROOT / "scripts" / "run_v2_live_demo.py")],
            "Track E: Real-World Live Pipeline Demonstration",
        )

    # Summary
    total_elapsed = round(time.perf_counter() - start_all, 2)
    print("\n" + "=" * 75)
    print("EVALUATION SUITE EXECUTION SUMMARY")
    print("=" * 75)
    print(f"  Step 0: Cryptographic Artifact Integrity: {'PASSED' if integrity_ok else 'FAILED'}")
    print(f"  Track C: Stance Benchmark (35 cases):     {'PASSED' if track_c_ok else 'FAILED'}")
    print(f"  Track D: End-to-End Hybrid (48 cases):    {'PASSED' if track_d_ok else 'FAILED'}")
    print(f"  Track F & G: Latency & Resilience:        {'PASSED' if track_fg_ok else 'FAILED'}")
    if args.live:
        print(f"  Track E: Real-World Live Demonstration:   {'PASSED' if track_e_ok else 'FAILED'}")
    else:
        print(f"  Track E: Real-World Live Demonstration:   SKIPPED (offline mode)")
    print(f"  Total Suite Runtime:                      {total_elapsed} s")
    print("=" * 75)

    all_passed = integrity_ok and track_c_ok and track_d_ok and track_fg_ok and track_e_ok
    if all_passed:
        print("ALL VERIFIED EVALUATION TRACKS COMPLETED SUCCESSFULLY.")
    else:
        print("ONE OR MORE EVALUATION TRACKS FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
