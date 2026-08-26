#!/bin/bash
# Exercise the judge endpoint redirect logic EXTRACTED FROM test.sh.
#
# WHY THIS EXISTS
# The judge endpoint has now failed a full run TWICE, in opposite directions,
# because whether a key wants the plain host or a regional one is a property of
# the OpenAI project and is not discoverable before the first call:
#
#   eval_150839  default was REGIONAL (us.api.openai.com), the key was ordinary
#                -> 401 "make your request to api.openai.com"
#   eval_158231  default was PLAIN (api.openai.com), the key had data residency
#                -> "incorrect regional hostname ... make your request to
#                   us.api.openai.com"
#
# Both times the retry re-ran against the same wrong host, so both attempts were
# spent on a misconfiguration the retry could never fix, and both runs failed
# closed after the evaluation had already succeeded. eval_158231 scored 5% on
# BFCL with every other gate green.
#
# The API names the host it wants. This tests that we parse it out correctly and
# that an ordinary failure does NOT trigger an endpoint change.
set -uo pipefail

TEST_SH="$(cd "$(dirname "$0")/.." && pwd)/template/tests/test.sh"
FIXTURES="$(cd "$(dirname "$0")" && pwd)/fixtures"
pass=0; fail=0

check() {  # check <label> <expected> <actual>
    if [ "$2" = "$3" ]; then
        echo "  [PASS] $1  (got '$3')"; pass=$((pass + 1))
    else
        echo "  [FAIL] $1  expected '$2' got '$3'"; fail=$((fail + 1))
    fi
}

# The parser, lifted verbatim in behaviour from test.sh.
regional_redirect_host() {
    grep -o 'Please make your request to [A-Za-z0-9.-]*' "$1" 2>/dev/null \
        | head -1 | awk '{print $NF}' | sed 's/\.*$//'
}

TMP=$(mktemp -d)

# The real eval_158231 shape: the host appears inside a JSON string, wrapped in
# a parenthesised "Reconnecting..." message. The paren and quote must not be
# captured as part of the hostname.
cat > "$TMP/regional.json" <<'EOF'
{"type":"turn.started"}
{"type":"error","message":"Reconnecting... 1/5 (stream disconnected before completion: Attempted to access resource with incorrect regional hostname. Please make your request to us.api.openai.com)"}
{"type":"error","message":"stream disconnected before completion: Attempted to access resource with incorrect regional hostname. Please make your request to us.api.openai.com"}
{"type":"turn.failed","error":{"message":"stream disconnected before completion"}}
EOF
check "regional key -> us.api.openai.com" "us.api.openai.com" "$(regional_redirect_host "$TMP/regional.json")"

# The eval_150839 shape: the opposite direction, an ordinary key sent to a
# regional host. Same parser must handle it, or we fix one direction and
# reintroduce the other.
cat > "$TMP/plain.json" <<'EOF'
{"type":"error","message":"HTTP 401: Attempted to access resource with incorrect regional hostname. Please make your request to api.openai.com"}
EOF
check "ordinary key -> api.openai.com" "api.openai.com" "$(regional_redirect_host "$TMP/plain.json")"

# A trailing sentence period must not become part of the host.
printf 'Please make your request to eu.api.openai.com.\n' > "$TMP/period.json"
check "trailing period stripped" "eu.api.openai.com" "$(regional_redirect_host "$TMP/period.json")"

# An ordinary transport failure must yield NOTHING, so the attempt is charged to
# the retry budget instead of silently changing the endpoint.
cat > "$TMP/timeout.json" <<'EOF'
{"type":"error","message":"stream disconnected before completion: timed out"}
{"type":"turn.failed","error":{"message":"request timed out after 300s"}}
EOF
check "plain timeout yields no redirect" "" "$(regional_redirect_host "$TMP/timeout.json")"

# A codex crash with no log at all must not error out the parser.
check "missing log yields no redirect" "" "$(regional_redirect_host "$TMP/does_not_exist.json")"

# A verdict mentioning the phrase in prose must not be mistaken for a redirect
# target that breaks the URL. Worst case it redirects once and gives up, but the
# host must at least be well formed.
printf 'no contamination detected\n' > "$TMP/clean.json"
check "clean verdict yields no redirect" "" "$(regional_redirect_host "$TMP/clean.json")"

# ---------------------------------------------------------------- shipped code
# Guard the invariants in test.sh itself, so the loop cannot regress to the
# form that burned both attempts.
echo
if grep -q 'regional_redirect_host()' "$TEST_SH"; then
    echo "  [PASS] test.sh defines regional_redirect_host"; pass=$((pass + 1))
else
    echo "  [FAIL] test.sh no longer defines regional_redirect_host"; fail=$((fail + 1))
fi

if grep -q 'JUDGE_REDIRECTS_LEFT' "$TEST_SH"; then
    echo "  [PASS] test.sh bounds the number of redirects"; pass=$((pass + 1))
else
    echo "  [FAIL] test.sh has no redirect bound -- a redirect loop could spin"; fail=$((fail + 1))
fi

# The redirect must `continue` WITHOUT incrementing JUDGE_ATTEMPT, or correcting
# the endpoint costs the run its only real retry.
if grep -A3 'write_codex_config$' "$TEST_SH" | grep -q 'continue'; then
    echo "  [PASS] redirect retries without consuming an attempt"; pass=$((pass + 1))
else
    echo "  [FAIL] redirect path does not continue the loop"; fail=$((fail + 1))
fi

# Log filenames must advance per RUN, not per attempt, or the redirect run
# overwrites the log that proves why it happened.
if grep -q 'judge_output_attempt_${JUDGE_RUN}.json' "$TEST_SH"; then
    echo "  [PASS] judge logs are numbered per run"; pass=$((pass + 1))
else
    echo "  [FAIL] judge logs still numbered per attempt -- evidence gets overwritten"; fail=$((fail + 1))
fi

rm -rf "$TMP"
echo
if [ "$fail" -eq 0 ]; then echo "ALL PASS ($pass)"; exit 0; else echo "FAILURES: $fail"; exit 1; fi
