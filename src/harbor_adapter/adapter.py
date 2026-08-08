import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

ADAPTER_NAME = "POSTTRAINBENCH"
TEMPLATE_DIR = Path(__file__).parent / "template"

# Text-file extensions/names normalized to LF line endings after generation.
# CRLF anywhere in these breaks things in ways that are easy to miss: a
# shell script with a CRLF shebang ("#!/bin/bash\r") fails on Linux with
# "bad interpreter: No such file or directory" *before it runs a single
# line* — bash never even starts, so nothing downstream (retry logic, EXIT
# traps, reward.txt writers) ever gets a chance to run. This bit us in
# practice: template/tests/test.sh and template/environment/entrypoint.sh
# were checked out from git with CRLF (Windows `core.autocrlf=true`), and
# shutil.copy() propagates a source file's bytes verbatim, so every
# generated task shipped a test.sh the verifier container couldn't exec at
# all — indistinguishable from the outside except as a silent hang followed
# by RewardFileNotFoundError. Separately, Path.write_text() on Windows
# translates '\n' to os.linesep ('\r\n') unless told not to, so
# adapter-generated files (timer.sh, task.toml, instruction.md,
# metadata.json) were equally at risk even though their source strings only
# ever contained '\n'.
_TEXT_SUFFIXES = {
    ".sh", ".py", ".toml", ".md", ".json", ".jsonl", ".txt", ".jinja",
    ".cfg", ".ini", ".yml", ".yaml",
}
# Dotfiles/extensionless files: Path.suffix is '' for names like
# ".dockerignore" (pathlib treats a leading-dot name as having no suffix),
# so these need matching by full name instead.
_TEXT_NAMES = {"Dockerfile", ".dockerignore"}


def _normalize_line_endings(root: Path) -> None:
    """Rewrite every text file under root to use LF-only line endings.

    Run once, at the end of task generation, over the whole task directory
    — covers files written via write_text() (Windows newline translation)
    and files shutil.copy()'d from a CRLF-checked-out source, without
    needing every call site to remember to handle it individually. Skips
    anything that doesn't decode as UTF-8 (defensive; no binary artifacts
    exist in a freshly generated task, but this keeps the pass safe if that
    ever changes).
    """
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in _TEXT_SUFFIXES and path.name not in _TEXT_NAMES:
            continue
        raw = path.read_bytes()
        if b"\r" not in raw:
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        path.write_bytes(normalized.encode("utf-8"))


def _compute_tests_checksums(tests_dir: Path, task_context_names: list[str]) -> dict[str, str]:
    """SHA-256 manifest of the verifier-owned files under tests_dir.

    Used for tamper detection under [verifier].environment_mode = "shared"
    (see task.toml): the agent and verifier now share one container/user,
    so nothing at the OS level stops the agent from editing evaluate.py,
    contamination_judge.py, or the eval templates/code before the verifier
    runs. test.sh recomputes this same manifest at verifier-start and flags
    any mismatch.

    Deliberately scoped to the FIXED set of files _copy_eval_files() itself
    places under tests_dir -- evaluate.py, contamination_judge.py,
    templates/, evaluation_code/, task_context/* -- rather than every file
    under tests_dir. This is called from generate_tests(), where tests_dir
    already contains Dockerfile/test.sh/entrypoint.sh/system_monitor.sh/
    requirements-direct.txt from earlier steps in that method -- hashing
    "whatever's left over" would silently sweep those infra files in too, so
    task_context_names (the exact item names _copy_eval_files() just copied
    from task_context/, if any) must be passed in explicitly rather than
    inferred from a directory scan.

    Called with tests_dir already populated (evaluate.py/templates/etc.
    copied in) but BEFORE metadata.json is written, so the manifest
    naturally excludes metadata.json itself.
    """
    candidate_paths: list[Path] = []
    # fetch_model.py is verifier-owned and security-relevant (it is what
    # re-checks the transferred model's SHA-256 manifest), so it belongs in the
    # tamper manifest alongside evaluate.py and the judge.
    for name in ("evaluate.py", "contamination_judge.py", "fetch_model.py"):
        candidate = tests_dir / name
        if candidate.is_file():
            candidate_paths.append(candidate)
    for dirname in ("templates", "evaluation_code"):
        dir_path = tests_dir / dirname
        if dir_path.is_dir():
            candidate_paths.extend(p for p in dir_path.rglob("*") if p.is_file())
    for name in task_context_names:
        item = tests_dir / name
        if item.is_file():
            candidate_paths.append(item)
        elif item.is_dir():
            candidate_paths.extend(p for p in item.rglob("*") if p.is_file())

    checksums: dict[str, str] = {}
    for path in candidate_paths:
        relpath = path.relative_to(tests_dir).as_posix()
        checksums[relpath] = hashlib.sha256(path.read_bytes()).hexdigest()
    return checksums


# PostTrainBench source directory (relative to repo root)
POSTTRAINBENCH_ROOT = Path(__file__).parent.parent.parent

# Claude-specific instruction clause (from original get_prompt.py)
CLAUDE_CLAUSE = """
You are running in a non-interactive mode. So make sure every process you are running finishes before you write your last message.
"""


@dataclass
class BenchmarkInfo:
    task_id: str           # e.g., "gsm8k"
    benchmark_name: str    # e.g., "GSM8K (Grade School Math 8K)"
    setup_note: str = ""   # Additional setup instructions


@dataclass
class ModelInfo:
    model_id: str          # HuggingFace model ID, e.g., "Qwen/Qwen3-1.7B-Base"
    short_name: str        # Short name for task IDs, e.g., "qwen3-1.7b"


BENCHMARKS = {
    "gsm8k": BenchmarkInfo(
        task_id="gsm8k",
        benchmark_name="GSM8K (Grade School Math 8K)",
        setup_note="- A final note: the evaluate.py script sometimes outputs ERROR warnings. "
                   "Do not be alarmed by this, this is normal behavior for inspect-ai. "
                   "Also if you run into issues with the evaluate.py script, this is likely "
                   "due to memory constraints on the GPU. In this case please decrease "
                   "--max-connections or --max-tokens.\n"
    ),
    "humaneval": BenchmarkInfo(
        task_id="humaneval",
        benchmark_name="HumanEval",
        setup_note="- A final note: the evaluate.py script sometimes outputs ERROR warnings. "
                   "Do not be alarmed by this, this is normal behavior for inspect-ai.\n"
    ),
    "aime2025": BenchmarkInfo(
        task_id="aime2025",
        benchmark_name="AIME 2025",
        setup_note="- A final note: the evaluate.py script sometimes outputs ERROR warnings. "
                   "Do not be alarmed by this, this is normal behavior for inspect-ai.\n"
    ),
    "gpqamain": BenchmarkInfo(
        task_id="gpqamain",
        benchmark_name="GPQA",
        setup_note="- A final note: the evaluate.py script sometimes outputs ERROR warnings. "
                   "Do not be alarmed by this, this is normal behavior for inspect-ai.\n"
    ),
    "bfcl": BenchmarkInfo(
        task_id="bfcl",
        benchmark_name="Berkeley Function Calling Leaderboard",
        setup_note="- A final note: the evaluate.py script sometimes outputs ERROR warnings. "
                   "Do not be alarmed by this, this is normal behavior for inspect-ai.\n"
    ),
    "arenahardwriting": BenchmarkInfo(
        task_id="arenahardwriting",
        benchmark_name="Arena-Hard-v2.0 (Writing)",
        setup_note="",
    ),
    "healthbench": BenchmarkInfo(
        task_id="healthbench",
        benchmark_name="HealthBench",
        setup_note="",
    ),
}

MODELS = {
    "qwen3-1.7b": ModelInfo(
        model_id="Qwen/Qwen3-1.7B-Base",
        short_name="qwen3-1.7b"
    ),
    "qwen3-4b": ModelInfo(
        model_id="Qwen/Qwen3-4B-Base",
        short_name="qwen3-4b"
    ),
    "smollm3-3b": ModelInfo(
        model_id="HuggingFaceTB/SmolLM3-3B-Base",
        short_name="smollm3-3b"
    ),
    "gemma3-4b": ModelInfo(
        model_id="google/gemma-3-4b-pt",
        short_name="gemma3-4b"
    ),
}


class PostTrainBenchAdapter:
    """Adapter to generate Harbor tasks from PostTrainBench configuration."""

    def __init__(
        self,
        output_dir: Path,
        num_hours: int = 10,
        include_claude_clause: bool = True,
        hf_token: str | None = None,
        openai_api_key: str | None = None,
    ):
        """
        Initialize the adapter.

        Args:
            output_dir: Directory where Harbor tasks will be generated.
            num_hours: Number of hours for the training task (default: 10).
            include_claude_clause: Whether to include the Claude non-interactive clause.
            hf_token: Hugging Face access token -- needed whenever the base
                model is gated (e.g. google/gemma-3-4b-pt) or the benchmark's
                dataset is gated (e.g. GPQA's Idavidrein/gpqa). Delivered via
                exactly ONE channel: a literal HF_TOKEN under task.toml's
                [environment.env], the platform's supported sandbox-level
                injection (lands in PID 1's environ, reaches both the agent
                and verifier processes; huggingface_hub auto-detects it).
                [agent.env] is NOT used -- confirmed via diagnostics
                (eval_131808/131841/131953) and by Data-OS that it isn't a
                valid section and is silently ignored. metadata.json is also
                NOT used any more: it was the original channel, but it ships
                inside the exported artifacts and agents were printing it
                into their logs, leaking the live token into downloaded
                evaluation bundles (see _copy_eval_files for the detail).
                Never persisted anywhere outside the one generated task's
                task.toml.
            openai_api_key: OpenAI API key, baked into metadata.json's
                environment/ copy only (agent side), for arenahardwriting/
                healthbench -- both benchmarks' evaluate.py calls an OpenAI
                judge, and the agent needs the key to self-check progress
                with its own copy of evaluate.py during training. Same
                rationale as hf_token: task.toml's [agent.env] OPENAI_API_KEY
                is kept as a first attempt (unlike HF_TOKEN this hasn't been
                confirmed broken for a real agent CLI, only for the oracle),
                but this is the proven-reliable fallback channel in case it
                turns out equally broken -- instruction.md tells the agent to
                check metadata.json if the env var isn't set. Optional: if
                not supplied, the fallback field is simply absent and the
                verifier's own [verifier.env] OPENAI_API_KEY (confirmed
                reliable) still covers final grading either way.
        """
        self.output_dir = Path(output_dir)
        self.num_hours = num_hours
        self.include_claude_clause = include_claude_clause
        self.hf_token = hf_token
        self.openai_api_key = openai_api_key
        self.posttrainbench_root = POSTTRAINBENCH_ROOT

    def _read_benchmark_name(self, benchmark_id: str) -> str:
        """Read the human-readable benchmark name from benchmark.txt."""
        bench_file = self.posttrainbench_root / "src" / "eval" / "tasks" / benchmark_id / "benchmark.txt"
        if bench_file.is_file():
            return bench_file.read_text(encoding="utf-8").strip()
        # Fallback to the dataclass info
        if benchmark_id in BENCHMARKS:
            return BENCHMARKS[benchmark_id].benchmark_name
        raise FileNotFoundError(f"Benchmark file not found: {bench_file}")

    def generate_task_toml(self, task_dir: Path, benchmark_id: str = "", model_key: str = "") -> None:
        """Generate task.toml for the Harbor task."""
        # Copy template and adjust timeout based on num_hours
        template_path = TEMPLATE_DIR / "task.toml"
        target_path = task_dir / "task.toml"

        content = template_path.read_text()

        # Adjust agent timeout based on num_hours
        agent_timeout = self.num_hours * 3600  # Convert hours to seconds
        content = content.replace(
            "timeout_sec = 36000.0",
            f"timeout_sec = {float(agent_timeout)}"
        )

        # [environment.env] -- the platform's SUPPORTED channel for injecting
        # env vars into the sandbox itself: values land in the container's
        # PID 1 environ, so both the agent's and the verifier's processes
        # inherit them. Confirmed empirically via the env-delivery
        # diagnostic (diagnostics/posttrainbench-diag-tasktoml-env, runs
        # eval_131808/131841/131953) and by Data-OS directly: "[agent.env]
        # is not a valid task-level section and is silently ignored;
        # [environment.env] injects variables into the sandbox environment."
        # The previous '[agent.env]' block this replaces was therefore dead
        # config -- it never delivered anything to any process.
        #
        # HF_TOKEN: literal value, whenever one was supplied at generation
        # time. huggingface_hub/transformers/datasets auto-detect the env
        # var, so gated downloads (gemma-3-4b*, GPQA's dataset) just work
        # without the agent having to discover the metadata.json fallback
        # (which stays in place as a proven belt-and-braces second copy).
        #
        # OPENAI_API_KEY (arenahardwriting/healthbench only -- their
        # evaluate.py calls an OpenAI judge the agent needs for self-checks):
        # host-substituted with a ":-" default, same pattern already proven
        # safe and working in [verifier.env].
        env_lines = []
        if self.hf_token:
            env_lines.append(f'HF_TOKEN = "{self.hf_token}"')
        if benchmark_id in ("arenahardwriting", "healthbench"):
            env_lines.append('OPENAI_API_KEY = "${OPENAI_API_KEY:-}"')
        if env_lines:
            content += (
                "\n# Sandbox-level env vars: injected into the container itself, inherited\n"
                "# by both the agent and verifier processes (see adapter.py for the\n"
                "# diagnostic evidence; [agent.env] is ignored by this platform).\n"
                "[environment.env]\n" + "\n".join(env_lines) + "\n"
            )

        # The verifier is a SEPARATE container, so [environment.env] above does
        # not reach it -- that section configures the agent's sandbox only.
        # The relay repo is private, so without the token here the verifier
        # authenticates as anonymous and snapshot_download fails with a 401
        # that reads like "repo not found". Appended to the existing
        # [verifier.env] block rather than emitting a second one, because a
        # duplicate table key is a TOML parse error.
        if self.hf_token:
            marker = 'CODEX_API_KEY = "${OPENAI_API_KEY:-}"'
            if marker not in content:
                raise RuntimeError(
                    "template/task.toml no longer contains the expected "
                    "[verifier.env] anchor; HF_TOKEN would not reach the "
                    "separate verifier and the relay download would 401."
                )
            content = content.replace(
                marker,
                marker
                + "\n# Needed to pull the private relay repo (see [[verifier.collect]]).\n"
                + f'HF_TOKEN = "{self.hf_token}"',
                1,
            )

        target_path.write_text(content)

    def generate_instruction(
        self,
        task_dir: Path,
        model_info: ModelInfo,
        benchmark_info: BenchmarkInfo,
        benchmark_id: str = "",
    ) -> None:
        """Generate instruction.md for the Harbor task."""
        template_path = TEMPLATE_DIR / "instruction.md"
        target_path = task_dir / "instruction.md"

        content = template_path.read_text()

        # Fill in placeholders
        content = content.replace("{model}", model_info.model_id)
        content = content.replace("{benchmark}", benchmark_info.benchmark_name)
        content = content.replace("{num_hours}", str(self.num_hours))
        content = content.replace("{setup_other}", benchmark_info.setup_note)

        # OpenAI restriction + key-fallback note for benchmarks that provide
        # OPENAI_API_KEY to agents. The metadata.json fallback line matters
        # because [agent.env] isn't confirmed reliable on this harness (see
        # generate_task_toml) -- the agent needs to be explicitly told to
        # check metadata.json itself if the env var comes up empty, since
        # nothing else here can fix up code the agent writes on its own.
        if benchmark_id in ("arenahardwriting", "healthbench"):
            content = content.replace(
                "{openai_restriction}",
                "- IMPORTANT: You are NOT allowed to use the OpenAI API for anything but the evaluation script.\n"
                "- OPENAI_API_KEY should already be set in your shell environment for calling evaluate.py. "
                "If it comes up empty or unset, read the \"openai_api_key\" field from metadata.json in this "
                "directory instead and export it yourself before calling evaluate.py.\n"
            )
        else:
            content = content.replace("{openai_restriction}", "")

        # HF_TOKEN note for gated models (gemma-3-4b*) or gated benchmark
        # datasets (gpqamain's Idavidrein/gpqa). The token arrives as a real
        # env var via task.toml's [environment.env] (see generate_task_toml),
        # and huggingface_hub/transformers/datasets auto-detect HF_TOKEN, so
        # gated downloads work without the agent doing anything.
        #
        # The old "if it fails, read hf_token from metadata.json" fallback
        # sentence was REMOVED on purpose. It was written when env delivery
        # was broken, but it actively instructed agents to `cat
        # metadata.json` -- and a real run then printed the live token into
        # trajectory.json, the terminal recording, test-stdout.txt and
        # judge_output.json, all of which get downloaded and shared. The
        # token is no longer in metadata.json at all, so the sentence would
        # also now be wrong.
        if model_info.model_id.startswith("google/gemma-3-4b") or benchmark_id == "gpqamain":
            content = content.replace(
                "{hf_token_note}",
                "- HF_TOKEN is already set in your environment, so gated Hugging Face "
                "downloads should work without any extra setup.\n"
            )
        else:
            content = content.replace("{hf_token_note}", "")

        # Multimodal-processor note for gemma3-4b specifically. Its
        # config.json declares "Gemma3ForConditionalGeneration" (a
        # multimodal architecture) even though it's only ever used as a
        # text LM here -- vLLM refuses to load a checkpoint saved without
        # an image processor alongside it ("Can't load image processor...
        # make sure it contains a preprocessor_config.json file"),
        # confirmed via multiple real runs where an agent's final_model
        # (saved with just model.save_pretrained()+tokenizer.save_pretrained(),
        # the standard/obvious way) failed at verification time for exactly
        # this reason -- a fine-tune that would otherwise have scored fine
        # gets a hard 0 purely from a save-step omission with no other
        # warning anywhere. Telling the agent up front costs one line;
        # finding out from a failed 10-hour run does not.
        if model_info.model_id.startswith("google/gemma-3-4b"):
            content = content.replace(
                "{multimodal_note}",
                f"- IMPORTANT: `{model_info.model_id}`'s config.json declares a multimodal architecture "
                "(Gemma3ForConditionalGeneration) even though you'll only use it as a text model here. "
                "When you save your final_model, also save the processor alongside the tokenizer "
                "(e.g. `AutoProcessor.from_pretrained(model_id).save_pretrained(\"final_model\")`), or "
                "vLLM will fail to load it at evaluation time with \"Can't load image processor\".\n"
            )
        else:
            content = content.replace("{multimodal_note}", "")

        if self.include_claude_clause:
            content += CLAUDE_CLAUSE

        target_path.write_text(content)

    def generate_timer_sh(self, env_dir: Path) -> None:
        """Generate timer.sh script that tracks remaining time.

        Self-initializes: on the first invocation it records the current
        timestamp at the absolute path /timer_start, and every later call
        counts down from there. This deliberately does NOT depend on a
        task.toml healthcheck to seed /timer_start — that healthcheck was
        removed because it raced Modal sandbox startup and required a custom
        ENTRYPOINT the Turness harness doesn't tolerate. The absolute path
        keeps the timer immune to the agent's `cd`s.
        """
        timer_script = f"""#!/bin/bash

NUM_HOURS={self.num_hours}
START_FILE="/timer_start"

# Seed the start time on first call; the countdown runs from here.
if [ ! -f "$START_FILE" ]; then
    date +%s > "$START_FILE"
fi

START_DATE=$(cat "$START_FILE")
DEADLINE=$((START_DATE + NUM_HOURS * 3600))
NOW=$(date +%s)
REMAINING=$((DEADLINE - NOW))

if [ $REMAINING -le 0 ]; then
    echo "Timer expired!"
else
    echo "Remaining time (hours:minutes)":
    HOURS=$((REMAINING / 3600))
    MINUTES=$(((REMAINING % 3600) / 60))
    printf "%d:%02d\\n" $HOURS $MINUTES
fi
"""
        timer_path = env_dir / "timer.sh"
        timer_path.write_text(timer_script)
        timer_path.chmod(0o755)

    def generate_environment(
        self,
        task_dir: Path,
        benchmark_id: str,
        model_info: "ModelInfo",
        benchmark_info: "BenchmarkInfo",
    ) -> None:
        """Generate the environment/ directory: Dockerfile + agent runtime."""
        env_dir = task_dir / "environment"
        env_dir.mkdir(parents=True, exist_ok=True)

        # Copy Dockerfile template and .dockerignore
        shutil.copy(
            TEMPLATE_DIR / "environment" / "Dockerfile",
            env_dir / "Dockerfile"
        )
        dockerignore_src = TEMPLATE_DIR / "environment" / ".dockerignore"
        if dockerignore_src.exists():
            shutil.copy(dockerignore_src, env_dir / ".dockerignore")

        # Build-context support files (entrypoint, system monitor,
        # requirements-direct). Shared with tests/ — see _copy_build_context_support.
        self._copy_build_context_support(env_dir)

        # Eval files: evaluate.py, templates/, optional evaluation_code/
        # and task_context contents, plus contamination_judge.py and
        # metadata.json. The agent gets these in /home/agent/workspace
        # (via the Dockerfile's `COPY .`) for fast iteration during
        # training.
        self._copy_eval_files(
            env_dir, benchmark_id, model_info, benchmark_info,
            include_hf_token=True, include_openai_key=True,
        )

        # publish_model.py — the agent side of the HF relay. Not for the agent
        # to run: the [[verifier.collect]] hook invokes it from
        # /home/agent/workspace after the agent phase has ended, to push
        # final_model to a private repo the verifier then pulls. It lands in
        # the workspace via the Dockerfile's `COPY .`, which means the agent
        # can tamper with it -- that is inherent to the agent having root in
        # its own container, and is why model provenance is established by the
        # verifier-side gates rather than by anything this script reports.
        # Deleting it fails closed: no pointer is written, so the verifier
        # scores 0 with a clear message.
        shutil.copy(
            TEMPLATE_DIR / "environment" / "publish_model.py",
            env_dir / "publish_model.py",
        )

        # timer.sh — agent reads it during the run. Verifier doesn't need it.
        self.generate_timer_sh(env_dir)

    def _copy_build_context_support(self, target_dir: Path) -> None:
        """Copy entrypoint.sh + system_monitor.sh + requirements-direct.txt
        into a Dockerfile build context.

        Both environment/ (agent) and tests/ (kept in sync for the
        dormant separate-verifier Dockerfile -- see generate_tests) use
        the same Dockerfile structure and need these files at build time.
        The canonical sources live under template/environment/ and
        containers/.
        """
        # entrypoint.sh — Dockerfile installs it at /usr/local/bin/ and
        # sets it as ENTRYPOINT so its stdout becomes Modal's live log
        # stream (see template/environment/entrypoint.sh).
        entrypoint_src = TEMPLATE_DIR / "environment" / "entrypoint.sh"
        entrypoint_dst = target_dir / "entrypoint.sh"
        shutil.copy(entrypoint_src, entrypoint_dst)
        entrypoint_dst.chmod(0o755)

        # system_monitor.sh — kicked off by entrypoint.sh as a background
        # daemon; ports condor's src/utils/system_monitor.sh.
        monitor_src = TEMPLATE_DIR / "environment" / "system_monitor.sh"
        monitor_dst = target_dir / "system_monitor.sh"
        shutil.copy(monitor_src, monitor_dst)
        monitor_dst.chmod(0o755)

        # containers/requirements-direct.txt — the Dockerfile pins ML
        # deps from this file (mirrors the condor opus_4_6_1m.def
        # pipeline).
        reqs_src = self.posttrainbench_root / "containers" / "requirements-direct.txt"
        if not reqs_src.exists():
            raise FileNotFoundError(
                f"requirements-direct.txt not found at {reqs_src}; "
                f"the Dockerfile expects it in the build context."
            )
        shutil.copy(reqs_src, target_dir / "requirements-direct.txt")

    def _copy_eval_files(
        self,
        target_dir: Path,
        benchmark_id: str,
        model_info: "ModelInfo",
        benchmark_info: "BenchmarkInfo",
        include_hf_token: bool = False,
        include_checksums: bool = False,
        include_openai_key: bool = False,
    ) -> None:
        """Copy the evaluation pipeline files into target_dir.

        Used for both:
          - environment/ (so the agent has them in /home/agent/workspace
            for iterative testing during training)
          - tests/ (so the verifier can check for tampering before scoring
            against it -- see include_checksums)

        Files copied:
          - evaluate.py            (benchmark-specific)
          - templates/             (chat templates for all model families)
          - evaluation_code/       (arenahardwriting, healthbench only)
          - task_context/<*>       (bfcl has bfcl_evaluation_code.py)
          - contamination_judge.py (judge prompt builder)
          - metadata.json          (benchmark + model info for verifier)

        include_checksums: only meaningful for the tests/ copy (see
        generate_tests). Under [verifier].environment_mode = "shared",
        the agent and verifier share one container/user, so nothing at
        the OS level stops the agent from editing these files before the
        verifier runs. When True, a SHA-256 manifest of everything copied
        above (see _compute_tests_checksums) is embedded into metadata.json
        as "tests_checksums" so tests/test.sh can detect tampering. Not
        set for the environment/ copy -- the agent's own copy of these
        files is expected/allowed to exist and a checksum of it would be
        circular (the agent controls both sides).
        """
        # evaluate.py
        eval_src = self.posttrainbench_root / "src" / "eval" / "tasks" / benchmark_id / "evaluate.py"
        if not eval_src.exists():
            raise FileNotFoundError(f"evaluate.py not found: {eval_src}")
        shutil.copy(eval_src, target_dir / "evaluate.py")

        # templates/
        templates_src = self.posttrainbench_root / "src" / "eval" / "templates"
        if not templates_src.exists():
            raise FileNotFoundError(f"templates directory not found: {templates_src}")
        shutil.copytree(templates_src, target_dir / "templates", dirs_exist_ok=True)

        # evaluation_code/ (arenahardwriting, healthbench)
        eval_code_src = self.posttrainbench_root / "src" / "eval" / "tasks" / benchmark_id / "evaluation_code"
        if eval_code_src.is_dir():
            shutil.copytree(eval_code_src, target_dir / "evaluation_code", dirs_exist_ok=True)

        # task_context/* (bfcl has bfcl_evaluation_code.py)
        task_context_src = self.posttrainbench_root / "src" / "eval" / "tasks" / benchmark_id / "task_context"
        task_context_names: list[str] = []
        if task_context_src.is_dir():
            for item in task_context_src.iterdir():
                dst = target_dir / item.name
                if item.is_dir():
                    shutil.copytree(item, dst, dirs_exist_ok=True)
                else:
                    shutil.copy(item, dst)
                task_context_names.append(item.name)

        # contamination judge script (kept in template/environment/ as
        # the canonical source, copied into both env_dir and tests_dir)
        judge_src = TEMPLATE_DIR / "environment" / "contamination_judge.py"
        if judge_src.exists():
            shutil.copy(judge_src, target_dir / "contamination_judge.py")

        # metadata.json
        metadata = {
            "benchmark_id": benchmark_id,
            "benchmark_name": benchmark_info.benchmark_name,
            "model_id": model_info.model_id,
            "model_short_name": model_info.short_name,
            "num_hours": self.num_hours,
        }
        # NOTE: the HF token is deliberately NOT written here any more.
        #
        # History: metadata.json was originally the ONLY delivery channel we
        # had confirmed working, because task.toml's [agent.env] never
        # reached the agent's process. Once the env-delivery diagnostic
        # proved [environment.env] injects at sandbox level and reaches both
        # the agent and the verifier, this copy became redundant -- and the
        # redundancy had a real cost, not just untidiness:
        #
        #   * environment/metadata.json lands in the agent's own workspace,
        #     which IS exported via [[artifacts]] -- so the token shipped
        #     inside every downloaded evaluation bundle.
        #   * Because instruction.md used to tell the agent to read the
        #     token from this file, agents actually did `cat metadata.json`,
        #     printing the live secret into trajectory.json, the terminal
        #     recording, and test-stdout.txt. The contamination judge then
        #     read the same file, putting it in judge_output.json too.
        #     Confirmed: one real gpqamain pair leaked it across 7 files.
        #
        # The token now travels only via task.toml's [environment.env] (see
        # generate_task_toml). Both consumers already fall back to the env
        # var when metadata.json has no token -- solve.sh reads
        # HF_TOKEN/HUGGING_FACE_HUB_TOKEN, and tests/test.sh's metadata read
        # simply becomes a no-op since HF_TOKEN is already exported -- so
        # nothing needs changing in either script, and a task generated by
        # an older adapter still works unchanged.
        _ = include_hf_token  # retained for call-site compatibility
        # OpenAI API key fallback for arenahardwriting/healthbench's agent
        # copy. task.toml's [agent.env] OPENAI_API_KEY is kept as a first
        # attempt, but it's the same delivery channel that was confirmed
        # broken for HF_TOKEN on the oracle agent -- rather than wait to hit
        # the same failure with a real agent mid-training-run, this gives
        # the agent's own evaluate.py invocations a proven-reliable fallback
        # (instruction.md tells the agent to check here if the env var is
        # missing). Not added to tests/ metadata.json -- [verifier.env]
        # OPENAI_API_KEY is already confirmed reliable there (the
        # contamination judge has used it successfully across many runs).
        if include_openai_key and benchmark_id in ("arenahardwriting", "healthbench") and self.openai_api_key:
            metadata["openai_api_key"] = self.openai_api_key
        # Checksum manifest for tamper detection -- computed here, after
        # every file above has been copied but before metadata.json itself
        # is written (so metadata.json is naturally excluded from its own
        # manifest). See _compute_tests_checksums and this method's
        # include_checksums docstring.
        #
        # Normalize line endings on target_dir FIRST: generate_task() runs
        # _normalize_line_endings() over the whole task dir at the very end
        # (fixing up CRLF from git-checked-out sources and Windows
        # write_text() translation -- see that function's docstring), which
        # happens AFTER this method returns. Hashing before that normalizing
        # pass would bake in checksums for bytes that are about to change,
        # so every verifier run would see a permanent, spurious "TAMPERED"
        # for every text file. Normalizing here first (idempotent, safe to
        # run again later) guarantees the hash matches what test.sh will
        # actually see.
        if include_checksums:
            _normalize_line_endings(target_dir)
            metadata["tests_checksums"] = _compute_tests_checksums(target_dir, task_context_names)
        (target_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    def generate_solution(self, task_dir: Path) -> None:
        """Generate the solution/ directory with the oracle solve.sh.

        Harnesses built on the Turing/Harbor task format validate that every
        task ships an oracle submission at solution/solve.sh (used to smoke-
        test the verifier). Native Harbor doesn't require it, so the adapter
        historically omitted it — which made generated tasks fail upload.

        The oracle is benchmark/model-agnostic: it reads the base model id
        from the workspace metadata.json and saves the unmodified base model
        to final_model/, giving the verifier a valid, non-zero baseline to
        score. See template/solution/solve.sh for the rationale.
        """
        solution_dir = task_dir / "solution"
        solution_dir.mkdir(parents=True, exist_ok=True)

        solve_src = TEMPLATE_DIR / "solution" / "solve.sh"
        solve_dst = solution_dir / "solve.sh"
        shutil.copy(solve_src, solve_dst)
        solve_dst.chmod(0o755)

    def generate_tests(
        self,
        task_dir: Path,
        benchmark_id: str,
        model_info: "ModelInfo",
        benchmark_info: "BenchmarkInfo",
    ) -> None:
        """Generate the tests/ directory.

        Under [verifier].environment_mode = "shared" (see task.toml), Harbor
        runs the verifier inside the agent's own container and copies
        tests/ into /tests at runtime rather than building tests/Dockerfile
        into a separate image. tests/Dockerfile is still generated (kept in
        sync so a future switch back to "separate" mode is a one-line
        task.toml edit) but isn't built/used for normal evaluation today.

        Files placed here:
          - Dockerfile      verifier image (dormant in shared mode, see above)
          - test.sh         the verifier orchestrator
          - entrypoint.sh   PID-1 streamer (matches agent env)
          - system_monitor.sh  background system monitor
          - requirements-direct.txt  pinned ML deps for the Dockerfile
          - evaluate.py + templates/ + evaluation_code/ + task_context/*
            + contamination_judge.py + metadata.json — the eval pipeline.
            metadata.json also carries a checksum manifest of these files
            (see _compute_tests_checksums) so test.sh can detect tampering
            now that the agent shares this filesystem.
        """
        tests_dir = task_dir / "tests"
        tests_dir.mkdir(parents=True, exist_ok=True)

        # Verifier image Dockerfile (canonical source: template/tests/Dockerfile).
        shutil.copy(
            TEMPLATE_DIR / "tests" / "Dockerfile",
            tests_dir / "Dockerfile",
        )

        # Verifier orchestrator (test.sh)
        test_sh_src = TEMPLATE_DIR / "tests" / "test.sh"
        test_sh_dst = tests_dir / "test.sh"
        shutil.copy(test_sh_src, test_sh_dst)
        test_sh_dst.chmod(0o755)

        # Build-context support files (same set the agent env needs).
        self._copy_build_context_support(tests_dir)

        # fetch_model.py — the verifier side of the HF relay. Copied BEFORE
        # _copy_eval_files so it exists when _compute_tests_checksums runs at
        # the end of that call and therefore lands in the tamper manifest.
        fetch_src = TEMPLATE_DIR / "tests" / "fetch_model.py"
        fetch_dst = tests_dir / "fetch_model.py"
        shutil.copy(fetch_src, fetch_dst)
        fetch_dst.chmod(0o755)

        # Eval pipeline (also baked into the agent workspace via
        # environment/, but the verifier reads from /tests/ where these
        # land via the verifier Dockerfile's `COPY .`).
        self._copy_eval_files(
            tests_dir, benchmark_id, model_info, benchmark_info,
            include_hf_token=True, include_checksums=True,
        )

    def generate_task(
        self,
        benchmark_id: str,
        model_key: str,
    ) -> Path:
        """
        Generate a complete Harbor task for a benchmark + model combination.

        Args:
            benchmark_id: The benchmark ID (e.g., "gsm8k").
            model_key: The model key (e.g., "qwen3-1.7b").

        Returns:
            Path to the generated task directory.
        """
        if benchmark_id not in BENCHMARKS:
            raise ValueError(f"Unknown benchmark: {benchmark_id}. Available: {list(BENCHMARKS.keys())}")
        if model_key not in MODELS:
            raise ValueError(f"Unknown model: {model_key}. Available: {list(MODELS.keys())}")

        benchmark_info = BENCHMARKS[benchmark_id]
        model_info = MODELS[model_key]

        # Try to get actual benchmark name from file
        try:
            benchmark_info = BenchmarkInfo(
                task_id=benchmark_info.task_id,
                benchmark_name=self._read_benchmark_name(benchmark_id),
                setup_note=benchmark_info.setup_note,
            )
        except FileNotFoundError:
            pass  # Use default from dataclass

        # Create task directory
        task_id = f"posttrainbench-{benchmark_id}-{model_info.short_name}"
        task_dir = self.output_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)

        print(f"Generating task: {task_id}")

        # Generate all components
        self.generate_task_toml(task_dir, benchmark_id, model_key)
        self.generate_instruction(task_dir, model_info, benchmark_info, benchmark_id)
        self.generate_environment(task_dir, benchmark_id, model_info, benchmark_info)
        self.generate_tests(task_dir, benchmark_id, model_info, benchmark_info)
        self.generate_solution(task_dir)

        # Final pass: force LF line endings everywhere (see _normalize_line_endings
        # docstring — a CRLF shebang makes a script unexecutable on Linux with no
        # useful error surfaced to the agent/verifier logs).
        _normalize_line_endings(task_dir)

        print(f"Task generated at: {task_dir}")
        return task_dir

    def generate_all_tasks(self) -> list[Path]:
        """Generate tasks for all benchmark + model combinations."""
        tasks = []
        for benchmark_id in BENCHMARKS:
            for model_key in MODELS:
                task_dir = self.generate_task(benchmark_id, model_key)
                tasks.append(task_dir)
        return tasks


def list_available_tasks() -> list[str]:
    """List all available task combinations."""
    tasks = []
    for benchmark_id in BENCHMARKS:
        for model_key in MODELS:
            task_id = f"posttrainbench-{benchmark_id}-{MODELS[model_key].short_name}"
            tasks.append(task_id)
    return tasks
