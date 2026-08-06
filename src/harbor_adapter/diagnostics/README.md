# Diagnostics

One-off probe tasks for verifying platform behavior. Not benchmarks — nothing here feeds the leaderboard or the main task pipeline.

## posttrainbench-diag-tasktoml-env

**v1 result (run eval_131808, terminus-2/Gemini, Daytona):** `[verifier.env]` delivered; `[agent.env]` did NOT reach the agent process even with literal values — Data-OS's suggestion disproven. But the same run showed our Dockerfile's own `ENV` lines (`NO_PROXY`, `DEBIAN_FRONTEND`) present in the agent's env — meaning **Dockerfile ENV is a delivery channel that works**.

**v2 (current) — question it answers:** which of five candidate channels actually reach the agent's process? Each carries a distinct marker value:

| Channel | Marker | Prior evidence |
|---|---|---|
| `task.toml [agent.env]` | `tokentest-agentenv-HF-AGT111` | broken (v1) |
| Dockerfile `ENV` | `chan-dockerfile-env-DKR555` | works (v1, implicit) |
| `/root/.bashrc` export | `chan-bashrc-export-BSH666` | untested |
| `/etc/profile.d` export | `chan-profiled-export-PRF777` | untested |
| `/etc/environment` line | `chan-etcenv-line-ETC888` | untested |

The agent also records its shell type (login vs not) and PID 1's environ — the latter shows whether `[agent.env]` values are injected at container level but lost at process-exec level, or never injected at all. `reward = 1` if the report exists and **any** agent-side channel delivered. `metrics.json` carries one 0/1 flag per channel.

If Dockerfile ENV confirms, the production fix is one adapter change: emit `ENV HF_TOKEN=...` into `environment/Dockerfile` at generation time (same exposure level as the current metadata.json plaintext — and better ergonomics, since `huggingface_hub` picks the env var up automatically without the agent reading any file).

Background: `[verifier.env]` is proven working in production (the contamination judge authenticates through it). `[agent.env]` is the doubtful one — with `${VAR}` host substitution it was confirmed NOT to reach the oracle agent's process. This probe tests the literal-value variant Data-OS suggested, using **distinct dummy strings per section** (no real secrets) so logs show exactly which section reached which phase:

| Section | HF_TOKEN value |
|---|---|
| `[agent.env]` | `tokentest-agentenv-HF-AGT111` |
| `[verifier.env]` | `tokentest-verifierenv-HF-VRF222` |

**How to run:** upload and run with a real agent (recommended — that's the delivery path we care about for production). The agent's entire job is one shell command that dumps its env to `agent_env_report.txt`; a run should take ~2 minutes of agent time. Running with `--agent oracle` additionally tests the oracle delivery path — worth doing both if cheap.

**Reading the result** — `verifier/test-stdout.txt` prints an explicit summary, and `metrics.json` encodes it:

```json
{"accuracy": 0, "agent_env_delivered": 1, "verifier_env_delivered": 1}
```

- `reward.txt = 1` → both deliveries work → safe to move the HF token into `task.toml` in production tasks.
- `agent_env_delivered: 0` with the report present → Data-OS's suggestion does NOT work for the agent phase → keep the metadata.json channel, and send them the report as evidence.
- Report missing entirely → the agent flaked on the task; rerun — says nothing about env delivery.

The task is CPU-only (no `gpus` key) and tiny. If the platform's validation refuses a GPU-less task, add `gpus = 1` to `[environment]` — it changes nothing about the test.
