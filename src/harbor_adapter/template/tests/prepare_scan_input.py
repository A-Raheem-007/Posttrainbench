#!/usr/bin/env python3
"""Flatten the agent's training data into scannable text, one row per line.

WHY THIS IS NOT OPTIONAL
------------------------
contamination_check.py pulls a document's text with:

    text = obj.get(text_field)                       # default "text"
    if text is None:  # try "text","content","input","prompt","question"
    if text is None:  text = " ".join(v for v in obj.values() if isinstance(v, str))
    if not text:      continue                       # <-- row silently dropped

Every one of those paths only looks at TOP-LEVEL strings. The most common SFT
format is `{"messages": [{"role": "user", "content": "..."}]}`, where the only
strings live inside a list of dicts. For such a file the scanner finds no text
in any row, drops all of them, and reports zero contamination -- a gate that
passes unconditionally on exactly the data format most agents will submit.

So we flatten first: walk each row recursively, collect every string anywhere
in it, and emit `{"text": ...}` that the scanner cannot miss. Over-inclusive on
purpose. A false positive costs a judge one extra look; a false negative means
the contamination gate was decoration.

Also handles the .gz form of the training file, which the scanner cannot read.

Prints the number of rows written to stdout so the caller can distinguish
"no training data" from "flattening failed".
"""

import argparse
import gzip
import json
import sys
from pathlib import Path

# Guards against a pathological row exploding memory. Far above any real
# training example; a row longer than this is truncated, not dropped, so it is
# still scanned.
MAX_CHARS_PER_ROW = 200_000


def collect_strings(value, out: list[str]) -> None:
    """Every string anywhere in the structure, depth-first."""
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, dict):
        for item in value.values():
            collect_strings(item, out)
    elif isinstance(value, (list, tuple)):
        for item in value:
            collect_strings(item, out)
    # numbers/bools/None carry no n-grams worth scanning


def read_rows(path: Path):
    raw = path.read_bytes()
    if path.suffix == ".gz":
        raw = gzip.decompress(raw)
    for number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            yield number, json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"[prepare-scan] {path.name} line {number}: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True,
                        help="audit/training_data.jsonl or .jsonl.gz")
    parser.add_argument("--output", required=True,
                        help="flattened JSONL for contamination_check.py")
    args = parser.parse_args()

    source = Path(args.input)
    if not source.is_file():
        raise SystemExit(f"[prepare-scan] no such file: {source}")

    written = 0
    with Path(args.output).open("w", encoding="utf-8") as handle:
        for number, row in read_rows(source):
            parts: list[str] = []
            collect_strings(row, parts)
            text = " ".join(p for p in parts if p.strip())
            if not text:
                continue
            handle.write(json.dumps(
                {"text": text[:MAX_CHARS_PER_ROW], "source": f"row:{number}"}
            ) + "\n")
            written += 1

    print(written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
