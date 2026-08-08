"""Offline tests for the verifier's mechanical gates. No network, no GPU."""
import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path

TEMPLATE = Path(__file__).resolve().parent.parent / "template"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ident = load("ident", TEMPLATE / "tests" / "model_identity_check.py")
evid = load("evid", TEMPLATE / "tests" / "validate_eval_evidence.py")

results = []


def check(label, fn, want_status=None, want_error_fragment=None):
    try:
        out = fn()
    except Exception as exc:  # noqa: BLE001
        if want_error_fragment:
            results.append((label, want_error_fragment in str(exc)))
        else:
            results.append((f"{label} -- unexpected {type(exc).__name__}: {exc}", False))
        return
    if want_error_fragment:
        results.append((f"{label} -- expected rejection, got {out}", False))
    elif want_status:
        results.append((label, out.get("status") == want_status))
    else:
        results.append((label, True))


tmp = Path(tempfile.mkdtemp())

ARCH = {
    "model_type": "qwen3", "hidden_size": 2048, "intermediate_size": 6144,
    "num_hidden_layers": 28, "num_attention_heads": 16, "num_key_value_heads": 8,
    "vocab_size": 151936, "head_dim": 128, "tie_word_embeddings": True,
}


def make_model(name, arch, weight_bytes):
    d = tmp / name
    d.mkdir(parents=True, exist_ok=True)
    cfg = dict(arch)
    cfg["architectures"] = ["Qwen3ForCausalLM"]
    (d / "config.json").write_text(json.dumps(cfg))
    (d / "model.safetensors").write_bytes(weight_bytes)
    return d


base_bytes = b"\xaa" * 2048
instruct_bytes = b"\xbb" * 2048
trained_bytes = b"\xcc" * 2048

base_dir = make_model("base", ARCH, base_bytes)
instruct_dir = make_model("instruct", ARCH, instruct_bytes)
trained_dir = make_model("trained", ARCH, trained_bytes)
wrong_arch = make_model("wrongarch", {**ARCH, "num_hidden_layers": 36}, trained_bytes)

IDENTITY = {
    "assigned_model_id": "Qwen/Qwen3-1.7B-Base",
    "assigned_revision": "ea980cb0",
    "architectures": ["Qwen3ForCausalLM"],
    "architecture": ARCH,
    "base_weight_sha256": [hashlib.sha256(base_bytes).hexdigest()],
    "prohibited_model_id": "Qwen/Qwen3-1.7B",
    "prohibited_weight_sha256": [hashlib.sha256(instruct_bytes).hexdigest()],
}

# --- identity gate ---
check("untouched base -> clean", lambda: ident.check(base_dir, IDENTITY), want_status="clean")
check("honest fine-tune -> derived", lambda: ident.check(trained_dir, IDENTITY), want_status="derived")
check("verbatim instruct model -> violation",
      lambda: ident.check(instruct_dir, IDENTITY), want_status="violation")
check("wrong architecture -> violation",
      lambda: ident.check(wrong_arch, IDENTITY), want_status="violation")
check("missing fingerprint fails closed",
      lambda: ident.check(trained_dir, {**IDENTITY, "architecture": {}}),
      want_error_fragment="no architecture fingerprint")

# Regression: a field OMITTED from the saved config is not a mismatch.
#
# transformers' save_pretrained uses to_diff_dict(), which drops any value
# equal to the config class default. tie_word_embeddings defaults to True, so
# an honest fine-tune whose value IS True has the key removed on save even
# though the base model's published config lists it. eval_152211 was failed on
# exactly this, with every other field matching, and "violation" reads as an
# accusation of submitting the wrong model.
omitted = tmp / "omitted_default"
omitted.mkdir()
cfg_omitted = {k: v for k, v in ARCH.items() if k != "tie_word_embeddings"}
cfg_omitted["architectures"] = ["Qwen3ForCausalLM"]
(omitted / "config.json").write_text(json.dumps(cfg_omitted))
(omitted / "model.safetensors").write_bytes(trained_bytes)
check("field omitted as a class default is not a violation",
      lambda: ident.check(omitted, IDENTITY), want_status="derived")

# ...but a field PRESENT with the wrong value still is.
wrong_value = tmp / "wrong_value"
wrong_value.mkdir()
(wrong_value / "config.json").write_text(json.dumps({
    **ARCH, "tie_word_embeddings": False, "architectures": ["Qwen3ForCausalLM"],
}))
(wrong_value / "model.safetensors").write_bytes(trained_bytes)
check("field present with a different value is still a violation",
      lambda: ident.check(wrong_value, IDENTITY), want_status="violation")

# And omission must not become a way to smuggle the prohibited checkpoint.
omitted_instruct = tmp / "omitted_instruct"
omitted_instruct.mkdir()
(omitted_instruct / "config.json").write_text(json.dumps(cfg_omitted))
(omitted_instruct / "model.safetensors").write_bytes(instruct_bytes)
check("prohibited weights still caught when fields are omitted",
      lambda: ident.check(omitted_instruct, IDENTITY), want_status="violation")

# Regression: Gemma-3 text-only extraction must NOT be flagged.
# Agents legitimately drop the vision tower of a multimodal base model for a
# text-only benchmark, which changes model_type gemma3 -> gemma3_text and
# architectures Gemma3ForConditionalGeneration -> Gemma3ForCausalLM. Upstream
# hit this and fixed it in new_judge_v2 (491bcff); an exact-equality check
# rejects an honest submission.
GEMMA_ARCH = {
    "model_type": "gemma3", "hidden_size": 2560, "intermediate_size": 10240,
    "num_hidden_layers": 34, "num_attention_heads": 8, "num_key_value_heads": 4,
    "vocab_size": 262208, "head_dim": 256, "tie_word_embeddings": True,
}
GEMMA_IDENTITY = {
    "assigned_model_id": "google/gemma-3-4b-pt",
    "assigned_revision": "abc123",
    "architectures": ["Gemma3ForConditionalGeneration"],
    "architecture": GEMMA_ARCH,
    "base_weight_sha256": [hashlib.sha256(base_bytes).hexdigest()],
    "prohibited_model_id": "google/gemma-3-4b-it",
    "prohibited_weight_sha256": [hashlib.sha256(instruct_bytes).hexdigest()],
}

gemma_text = tmp / "gemma_text"
gemma_text.mkdir()
(gemma_text / "config.json").write_text(json.dumps({
    **GEMMA_ARCH,
    "model_type": "gemma3_text",
    "architectures": ["Gemma3ForCausalLM"],
}))
(gemma_text / "model.safetensors").write_bytes(trained_bytes)
check("gemma3 text-only extraction -> derived (not flagged)",
      lambda: ident.check(gemma_text, GEMMA_IDENTITY), want_status="derived")

# ...but a genuinely different family must still be rejected, so the
# equivalence above is not a blanket escape hatch.
not_gemma = tmp / "not_gemma"
not_gemma.mkdir()
(not_gemma / "config.json").write_text(json.dumps({
    **GEMMA_ARCH, "model_type": "llama", "architectures": ["LlamaForCausalLM"],
}))
(not_gemma / "model.safetensors").write_bytes(trained_bytes)
check("unrelated family still -> violation",
      lambda: ident.check(not_gemma, GEMMA_IDENTITY), want_status="violation")

# Regression: the real gemma-3-4b-pt config declares model_type "gemma3" at
# the top level and "gemma3_text" inside text_config. Flattening must NOT let
# the nested value win, or an unmodified checkpoint describes itself as the
# text-only variant and the fingerprint stops meaning what it says.
real_gemma_config = {
    "architectures": ["Gemma3ForConditionalGeneration"],
    "model_type": "gemma3",
    "text_config": {
        "model_type": "gemma3_text",
        "hidden_size": 2560,
        "intermediate_size": 10240,
        "num_hidden_layers": 34,
    },
}
flat = ident.flatten_config(real_gemma_config)
check("nested text_config does not hijack model_type",
      lambda: {"status": flat["model_type"]}, want_status="gemma3")
check("nested text_config does not hijack architectures",
      lambda: {"status": flat["architectures"][0]}, want_status="Gemma3ForConditionalGeneration")
check("nested dimensions are still lifted",
      lambda: {"status": str(flat["num_hidden_layers"])}, want_status="34")

# And the equivalence must not let the prohibited instruct model through.
gemma_instruct = tmp / "gemma_instruct"
gemma_instruct.mkdir()
(gemma_instruct / "config.json").write_text(json.dumps({
    **GEMMA_ARCH, "model_type": "gemma3_text",
    "architectures": ["Gemma3ForCausalLM"],
}))
(gemma_instruct / "model.safetensors").write_bytes(instruct_bytes)
check("prohibited weights still caught despite text-only config",
      lambda: ident.check(gemma_instruct, GEMMA_IDENTITY), want_status="violation")

# --- evidence gate ---
def write_log(d, *, status="success", limit=None, accuracy=0.42, samples=100):
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": status,
        "eval": {"config": ({"limit": limit} if limit is not None else {})},
        "results": {"scores": [{"metrics": {"accuracy": {"value": accuracy}}}]},
        "samples": [{"id": i} for i in range(samples)],
    }
    (d / "log.json").write_text(json.dumps(payload))
    return d


def metrics(path, accuracy=0.42):
    path.write_text(json.dumps({"accuracy": accuracy, "stderr": 0.01}))
    return path


good_logs = write_log(tmp / "logs_good")
good_metrics = metrics(tmp / "metrics_good.json")
check("complete run accepted", lambda: evid.validate(good_metrics, good_logs))

lim_logs = write_log(tmp / "logs_limited", limit=10, samples=10)
check("--limit run rejected",
      lambda: evid.validate(metrics(tmp / "m_lim.json"), lim_logs),
      want_error_fragment="subsetted")

forged = metrics(tmp / "m_forged.json", accuracy=0.97)
check("forged metrics.json rejected",
      lambda: evid.validate(forged, good_logs),
      want_error_fragment="disagrees with the inspect log")

empty = tmp / "logs_empty"
empty.mkdir()
check("no successful log rejected",
      lambda: evid.validate(good_metrics, empty),
      want_error_fragment="found 0")

two = write_log(tmp / "logs_two")
shutil.copy(two / "log.json", two / "log2.json")
check("stale second log rejected",
      lambda: evid.validate(good_metrics, two),
      want_error_fragment="found 2")

failed = write_log(tmp / "logs_failed", status="error")
check("failed run rejected",
      lambda: evid.validate(good_metrics, failed),
      want_error_fragment="found 0")

bad_metrics = tmp / "m_bad.json"
bad_metrics.write_text(json.dumps({"accuracy": "high"}))
check("non-numeric accuracy rejected",
      lambda: evid.validate(bad_metrics, good_logs),
      want_error_fragment="no numeric accuracy")

out_of_range = tmp / "m_range.json"
out_of_range.write_text(json.dumps({"accuracy": 1.5}))
check("out-of-range accuracy rejected",
      lambda: evid.validate(out_of_range, good_logs),
      want_error_fragment="out of range")

# --- custom-harness benchmarks (healthbench / arenahardwriting) ---
# Regression: these run their own grader loop and produce NO inspect log, so
# the inspect rules rejected every honest run (eval_151149 zeroed a complete
# healthbench result with 245 graded examples).
no_logs = tmp / "no_logs_at_all"

hb = tmp / "m_healthbench.json"
hb.write_text(json.dumps({
    "accuracy": 0.18729470596511, "stderr": 0.01836779380453516,
    "n_examples": 245, "total_grader_calls": 1873,
    "by_theme": {"hedging": 0.302}, "by_axis": {"accuracy": 0.249},
}))
check("real healthbench result accepted (custom harness)",
      lambda: evid.validate(hb, no_logs, "custom"))

ahw = tmp / "m_arena.json"
ahw.write_text(json.dumps({"accuracy": 0.42, "stderr": 0.03}))
check("real arenahardwriting result accepted (custom harness)",
      lambda: evid.validate(ahw, no_logs, "custom"))

# ...but the custom path must not accept anything at all.
accuracy_only = tmp / "m_acc_only.json"
accuracy_only.write_text(json.dumps({"accuracy": 0.42}))
check("custom harness still needs stderr",
      lambda: evid.validate(accuracy_only, no_logs, "custom"),
      want_error_fragment="no numeric stderr")

empty_run = tmp / "m_empty.json"
empty_run.write_text(json.dumps({"accuracy": 0.0, "stderr": 0.0}))
check("empty/failed custom run rejected",
      lambda: evid.validate(empty_run, no_logs, "custom"),
      want_error_fragment="not a completed evaluation")

zero_graded = tmp / "m_zero.json"
zero_graded.write_text(json.dumps({"accuracy": 0.5, "stderr": 0.1, "n_examples": 0}))
check("custom run grading zero examples rejected",
      lambda: evid.validate(zero_graded, no_logs, "custom"),
      want_error_fragment="0 graded examples")

# And an inspect benchmark must NOT get the lenient path just because its log
# is missing -- the harness comes from metadata, not from the filesystem.
check("inspect benchmark with no log still rejected",
      lambda: evid.validate(good_metrics, no_logs, "inspect"),
      want_error_fragment="does not exist")

print()
ok = True
for label, passed in results:
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
    ok = ok and passed
shutil.rmtree(tmp, ignore_errors=True)
print()
print("ALL PASS" if ok else "FAILURES PRESENT")
sys.exit(0 if ok else 1)
