#!/bin/bash
# Oracle submission for PostTrainBench verifier smoke tests.
#
# Purpose (mirrors ProgramBench's solution/solve.sh): give the harness a
# known-good submission so the verifier pipeline — contamination judge +
# vLLM benchmark eval + reward extraction — can be exercised end to end,
# and so task upload/validation (which requires solution/solve.sh) passes.
#
# This oracle deliberately does NOT post-train. It saves the *unmodified*
# base model to ./final_model, which yields the model's baseline benchmark
# score: a non-zero, deterministic reward that proves the verifier works.
# A real agent is expected to beat this baseline by actually fine-tuning.
#
# It is generic across every benchmark/model task: the base model id is
# read from metadata.json (baked into the workspace by the adapter), so the
# same script is dropped into all 28 tasks unchanged.
set -euo pipefail

cd /home/agent/workspace

MODEL_ID=$(python3 -c "import json; print(json.load(open('metadata.json'))['model_id'])")
echo "[solve] oracle: saving base model '${MODEL_ID}' to ./final_model"

# Diagnostic only -- never prints the value itself. Some base models (e.g.
# google/gemma-3-4b-pt) are gated on Hugging Face and need an authenticated
# token or the download 401s with GatedRepoError.
#
# Primary token source is metadata.json's "hf_token" field (baked in at
# generation time by the adapter, GATED_MODELS only) -- NOT an env var.
# A real run's diagnostic output here previously proved this harness does
# NOT apply task.toml's [agent.env] to this agent type at all ("no HF
# token env var is set in this process" despite HF_TOKEN being correctly
# configured there), while the token itself was separately confirmed valid
# and genuinely authorized for the gated repo (tested directly against
# huggingface_hub). HF_TOKEN/HUGGING_FACE_HUB_TOKEN env vars are kept as a
# harmless secondary fallback in case that plumbing ever starts working.
TOKEN_SOURCE=$(python3 -c "
import json
try:
    meta = json.load(open('metadata.json'))
    tok = meta.get('hf_token')
    print(f'metadata.json ({len(tok)} chars)' if tok else '')
except Exception:
    print('')
")
if [ -n "$TOKEN_SOURCE" ]; then
    echo "[solve] HF token found in $TOKEN_SOURCE"
elif [ -n "${HF_TOKEN:-}" ]; then
    echo "[solve] HF_TOKEN env var is set in this process (length: ${#HF_TOKEN} chars)"
elif [ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
    echo "[solve] HUGGING_FACE_HUB_TOKEN env var is set in this process (length: ${#HUGGING_FACE_HUB_TOKEN} chars)"
else
    echo "[solve] no HF token found (metadata.json or env var) -- fine for public models, fatal for gated ones"
fi

python3 - "$MODEL_ID" <<'PY'
import json
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

model_id = sys.argv[1]

# metadata.json first (proven-reliable delivery channel on this harness),
# env vars as a harmless fallback. Explicit token= instead of relying on
# huggingface_hub's implicit env-var auto-detection either way.
token = None
try:
    meta = json.load(open("metadata.json"))
    token = meta.get("hf_token")
except Exception:
    pass
if not token:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

# Download the repository's original checkpoint files directly instead of
# loading the model into transformers and calling save_pretrained(). The
# load+save path silently upcasts bf16 weights to FP32 (no precision is
# gained -- it just doubles the checkpoint on disk, e.g. 3.44 GB -> 6.88 GB
# for Qwen3-1.7B), which also made the checkpoint slower to move around.
# snapshot_download copies the repo's files byte-for-byte, so the original
# bf16 size is preserved, and nothing has to be fully loaded into RAM first.
snapshot_download(
    repo_id=model_id,
    local_dir="final_model",
    token=token,
    allow_patterns=[
        "*.json",
        "*.jinja",
        "*.model",
        "*.safetensors",
        "*.tiktoken",
        "*.txt",
        "merges*",
        "tokenizer*",
        "vocab*",
    ],
)

model_dir = Path("final_model")
weight_files = [
    *model_dir.glob("*.safetensors"),
    *model_dir.glob("pytorch_model*.bin"),
]
if not weight_files:
    raise RuntimeError("snapshot_download produced no model weight files")
for weight_file in weight_files:
    if weight_file.stat().st_size == 0:
        raise RuntimeError(f"empty model weight file: {weight_file}")

# Some base models (e.g. google/gemma-3-4b-pt) are multimodal-architected
# upstream (config.json declares "Gemma3ForConditionalGeneration") even
# though this benchmark only ever uses them as text LMs. vLLM expects an
# image processor alongside the model at serve time for these -- without
# one, engine init fails with "Can't load image processor... make sure ...
# contains a preprocessor_config.json file" (confirmed via a real run).
# snapshot_download's "*.json" pattern above likely already pulls
# preprocessor_config.json/processor_config.json since they're plain JSON
# files in the repo, but that hasn't been verified end-to-end for a
# multimodal model through this exact code path -- so this AutoProcessor
# step stays as a belt-and-suspenders save, matching the fix that was
# already confirmed working. For pure-text models with no processor
# config upstream (qwen3-*, smollm3-3b) this just raises, which is
# expected and harmless.
try:
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, token=token)
    processor.save_pretrained("final_model")
    print("[solve] saved image/multimodal processor alongside tokenizer (multimodal-architected base model)")
except Exception as e:
    print(f"[solve] no separate processor to save (expected for text-only models): {type(e).__name__}: {e}")
PY

echo "[solve] final_model contents:"
ls -la final_model

if [[ ! -f final_model/config.json ]] || \
   ! find final_model -maxdepth 1 -type f \
       \( -name '*.safetensors' -o -name 'pytorch_model*.bin' \) \
       -size +0c -print -quit | grep -q .; then
    echo "[solve] ERROR: final_model is missing config.json or non-empty weight files" >&2
    exit 1
fi
echo "[solve] OK: valid final_model/ produced (baseline oracle)"
