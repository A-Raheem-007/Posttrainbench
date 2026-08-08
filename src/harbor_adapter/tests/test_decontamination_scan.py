"""End-to-end test of the deterministic decontamination gate.

Uses the REAL provisioned humaneval reference set and the REAL scanner, so a
pass here means the gate actually detects contamination rather than that a
mock agreed with itself.

The central case is chat-format training data. contamination_check.py only
looks for text in top-level string fields, so `{"messages": [...]}` -- the most
common SFT shape -- is silently dropped and scores clean unless
prepare_scan_input.py flattens it first. A gate that passes unconditionally on
the most likely input format is worse than no gate, because it reads as
assurance.
"""
import gzip
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TESTS = ROOT / "template" / "tests"
REFERENCE = ROOT.parent / "eval" / "tasks" / "humaneval" / "test_data.json"

if not REFERENCE.is_file():
    print(f"SKIP: reference data not provisioned at {REFERENCE}")
    print("      run: python src/harbor_adapter/tools/download_test_data.py")
    sys.exit(0)

ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
print(f"  reference: {len(ref)} humaneval items\n")

# Build genuinely contaminated rows from real benchmark items.
#
# BOTH halves, deliberately. decon scores question and answer jointly against a
# 0.8 combined threshold, so a row carrying only the prompt scores 0 and is NOT
# flagged -- correctly, since a question without its answer teaches nothing.
# Measured on this exact reference set: question-only 0 hits, answer-only
# 0 hits, question+answer 1 hit at score 1.000. Contaminated training data
# contains both, which is what this reproduces.
def ref_text(item):
    parts = [item.get(key) for key in ("question", "answer", "prompt", "text")]
    return "\n".join(p for p in parts if isinstance(p, str) and p.strip())

CONTAMINATED = [ref_text(item) for item in ref[:8]]
CLEAN = [
    "Write a function that reverses a linked list in place using three pointers.",
    "Explain the difference between a process and a thread in an operating system.",
    "Implement binary search over a sorted array of integers and return the index.",
    "Describe how a hash map resolves collisions with open addressing.",
    "Write a SQL query that returns the second highest salary from an employees table.",
] * 3

tmp = Path(tempfile.mkdtemp())
results = []


def run_scan(rows, label, gz=False):
    """Flatten then scan, exactly as test.sh does. Returns hit count."""
    src = tmp / (f"{label}.jsonl.gz" if gz else f"{label}.jsonl")
    payload = "".join(json.dumps(r) + "\n" for r in rows).encode()
    src.write_bytes(gzip.compress(payload) if gz else payload)

    flat = tmp / f"{label}.flat.jsonl"
    prep = subprocess.run(
        [sys.executable, str(TESTS / "prepare_scan_input.py"),
         "--input", str(src), "--output", str(flat)],
        capture_output=True, text=True)
    if prep.returncode != 0:
        return None, f"prepare failed: {prep.stderr.strip()}"
    written = int(prep.stdout.strip())
    if written == 0:
        return 0, "0 rows flattened"

    matches = tmp / f"{label}.matches.jsonl"
    with matches.open("w") as out:
        subprocess.run(
            [sys.executable, str(TESTS / "contamination_check.py"),
             "--reference", str(REFERENCE), "--input", str(flat)],
            stdout=out, stderr=subprocess.DEVNULL)
    hits = sum(1 for line in matches.read_text().splitlines() if line.strip())
    return hits, f"{written} rows flattened"


def check(label, ok, detail=""):
    results.append((label, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{'  -- ' + detail if detail else ''}")


# 1. Clean data must not trip the gate.
hits, info = run_scan([{"text": t} for t in CLEAN], "clean")
check("clean corpus scores 0 hits", hits == 0, f"{hits} hits, {info}")

# 2. Plain-text contamination must be caught.
hits, info = run_scan([{"text": t} for t in CONTAMINATED], "dirty_flat")
check("8 verbatim test items detected", hits is not None and hits > 3,
      f"{hits} hits (tolerance 3)")

# 3. THE CRITICAL CASE: chat format. Without flattening this scores 0.
chat_rows = [{"messages": [{"role": "user", "content": t},
                           {"role": "assistant", "content": "..."}]}
             for t in CONTAMINATED]
hits, info = run_scan(chat_rows, "dirty_chat")
check("chat-format contamination detected (nested strings)",
      hits is not None and hits > 3, f"{hits} hits, {info}")

# 3b. Prove the flattening is what saves it: same rows, scanner alone.
raw = tmp / "chat_raw.jsonl"
raw.write_text("".join(json.dumps(r) + "\n" for r in chat_rows))
with (tmp / "chat_raw.matches").open("w") as out:
    subprocess.run(
        [sys.executable, str(TESTS / "contamination_check.py"),
         "--reference", str(REFERENCE), "--input", str(raw)],
        stdout=out, stderr=subprocess.DEVNULL)
unflattened = sum(1 for l in (tmp / "chat_raw.matches").read_text().splitlines() if l.strip())
check("...and WOULD have been missed without flattening",
      unflattened == 0, f"scanner alone found {unflattened} hits on the same data")

# 4. Gzip form must work (the oracle submits .gz).
hits, info = run_scan([{"text": t} for t in CONTAMINATED], "dirty_gz", gz=True)
check("gzipped training data scanned", hits is not None and hits > 3, f"{hits} hits")

# 5. Empty file = no-training baseline = genuine pass.
hits, info = run_scan([], "empty")
check("empty training data yields 0 rows", hits == 0, info)

# 6. Tolerance boundary: 3 contaminated rows must still pass.
hits, info = run_scan([{"text": t} for t in CONTAMINATED[:3]], "boundary")
check("3 contaminated rows is within tolerance", hits is not None and hits <= 3,
      f"{hits} hits (tolerance 3)")

print()
ok = all(v for _, v in results)
print("ALL PASS" if ok else "FAILURES PRESENT")
sys.exit(0 if ok else 1)
