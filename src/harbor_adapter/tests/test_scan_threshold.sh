#!/bin/bash
# Exercise the decontamination threshold logic EXTRACTED FROM test.sh.
#
# WHY THIS EXISTS
# The previous counting expression was:
#     SCAN_HITS=$(grep -c . "$f" || echo 0)
# which produces the two-line string "0\n0" on a clean scan, because grep -c
# prints "0" AND exits 1 when nothing matched, so the `|| echo 0` also runs.
# The integer test then fails and a clean scan is reported CONTAMINATED.
#
# It failed three real runs (eval_153121 at 0/10008, eval_152814 at 0/5000,
# eval_152815 at 0/6000) before anyone noticed, because the only case never
# tested was the ordinary one: no contamination at all.
#
# This runs the real logic against 0 / 3 / 4 / missing, so the clean case can
# never regress silently again.
set -uo pipefail

TEST_SH="$(cd "$(dirname "$0")/.." && pwd)/template/tests/test.sh"
SCAN_TOLERANCE=3
pass=0; fail=0

check() {  # check <label> <expected SCAN_OK> <actual SCAN_OK> <detail>
    if [ "$2" = "$3" ]; then
        echo "  [PASS] $1  ($4)"; pass=$((pass + 1))
    else
        echo "  [FAIL] $1  expected SCAN_OK=$2 got $3  ($4)"; fail=$((fail + 1))
    fi
}

# The counting block, lifted verbatim in behaviour from test.sh.
count_and_decide() {   # $1 = matches file
    SCAN_MATCHES="$1"
    if [ -s "$SCAN_MATCHES" ]; then
        SCAN_HITS=$(grep -c '' "$SCAN_MATCHES")
    else
        SCAN_HITS=0
    fi
    case "$SCAN_HITS" in
        ''|*[!0-9]*) SCAN_HITS=-1 ;;
    esac
    if [ "$SCAN_HITS" -ge 0 ] && [ "$SCAN_HITS" -le "$SCAN_TOLERANCE" ]; then
        echo "1 $SCAN_HITS"
    else
        echo "0 $SCAN_HITS"
    fi
}

TMP=$(mktemp -d)

# 0 matches: the clean case, and the one that regressed.
: > "$TMP/none.jsonl"
read -r ok hits <<< "$(count_and_decide "$TMP/none.jsonl")"
check "clean scan passes" 1 "$ok" "hits=$hits"

# Exactly at tolerance.
for i in 1 2 3; do echo "{\"source\":\"row:$i\"}" >> "$TMP/three.jsonl"; done
read -r ok hits <<< "$(count_and_decide "$TMP/three.jsonl")"
check "3 hits is within tolerance" 1 "$ok" "hits=$hits"

# One over.
cp "$TMP/three.jsonl" "$TMP/four.jsonl"; echo '{"source":"row:4"}' >> "$TMP/four.jsonl"
read -r ok hits <<< "$(count_and_decide "$TMP/four.jsonl")"
check "4 hits exceeds tolerance" 0 "$ok" "hits=$hits"

# Missing file behaves like the empty one, not like an error.
read -r ok hits <<< "$(count_and_decide "$TMP/does_not_exist.jsonl")"
check "missing matches file counts as 0" 1 "$ok" "hits=$hits"

# Guard against the shipped test.sh drifting back to the broken form.
if grep -q 'grep -c \. "\$SCAN_MATCHES" 2>/dev/null || echo 0' "$TEST_SH"; then
    echo "  [FAIL] test.sh still contains the broken 'grep -c . || echo 0' form"
    fail=$((fail + 1))
else
    echo "  [PASS] test.sh does not contain the broken counting form"
    pass=$((pass + 1))
fi

rm -rf "$TMP"
echo
if [ "$fail" -eq 0 ]; then echo "ALL PASS ($pass)"; exit 0; else echo "FAILURES: $fail"; exit 1; fi
