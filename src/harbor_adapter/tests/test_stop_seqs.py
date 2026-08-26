"""Exercise template_stop_seqs() for every benchmark, without importing inspect_evals.

Every benchmark serves its model through a chat template from templates/, and
every one of those templates ends a turn with a token that is NOT the base
model's eos_token_id. If a benchmark does not pass that terminator through to
vLLM, a fine-tune that learned the template's own format never stops, and the
scorer reads whatever the run-on produced. That defect cost BFCL eval_153283
(96 of 100 samples truncated) and GSM8K eval_194142 (3.11% reported for a
model measurably better than its 12.89% untrained baseline).

So this test asserts two things per benchmark: the mapping resolves to the
right terminator per model family, and no template the benchmark can select is
missing from the mapping.
"""
import ast
import os
import pathlib
import types

# Resolved from this file's own location, so the test runs from any checkout:
# tests/ -> harbor_adapter/ -> src/
TASKS = pathlib.Path(__file__).resolve().parents[2] / "eval" / "tasks"
TEMPLATES = TASKS.parent / "templates"

# The terminator each shipped template actually emits, read from the template
# rather than restated here, so a template edit cannot silently disagree.
TERMINATORS = ("<|im_end|>", "<end_of_turn>", "<|eot_id|>")


def template_terminator(name: str) -> str:
    text = (TEMPLATES / name).read_text(encoding="utf-8")
    found = [t for t in TERMINATORS if t in text]
    assert len(found) == 1, f"{name}: expected one terminator, found {found}"
    return found[0]


def load(bench: str) -> dict:
    """Pull just the template-resolution helpers out of a benchmark's evaluate.py."""
    src = (TASKS / bench / "evaluate.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    wanted_fn = {"model_type", "template_kwargs", "template_args", "template_stop_seqs"}
    pieces = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted_fn:
            pieces.append(ast.get_source_segment(src, node))
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "TEMPLATE_STOP_SEQS":
                    pieces.append(ast.get_source_segment(src, node))
    ns = {"os": os, "json": __import__("json")}
    exec("\n\n".join(pieces), ns)  # noqa: S102
    return ns


FAMILIES = ("qwen3", "gemma3", "smollm3", "llama3")
BENCHES = ("aime2025", "arenahardwriting", "bfcl", "gpqamain",
           "gsm8k", "healthbench", "humaneval")

ok = True
for bench in BENCHES:
    ns = load(bench)
    resolve = ns.get("template_kwargs") or ns.get("template_args")
    assert resolve is not None, f"{bench}: no template resolver found"

    def template_for(family):
        args = types.SimpleNamespace(
            model_path=f"/logs/artifacts/final_model_{family}",
            templates_dir="/tests/templates",
        )
        got = resolve(args)
        path = got["chat_template"] if isinstance(got, dict) else got[-1]
        return os.path.basename(path), args

    for family in FAMILIES:
        template, args = template_for(family)
        # llama3.jinja is referenced by every evaluate.py but is not shipped in
        # templates/, so its terminator cannot be read from disk. Skip it here
        # rather than assert against a file that is not there; the coverage
        # check below still requires the mapping to carry an entry for it.
        if not (TEMPLATES / template).exists():
            continue
        expected = [template_terminator(template)]
        got = ns["template_stop_seqs"](args)
        good = got == expected
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] {bench:18} {family:8} -> {template:26} {got}")

    # Every template this benchmark can select must have an entry, or a model
    # family silently gets no stop sequence and regresses to the original bug.
    selectable = {template_for(f)[0] for f in FAMILIES}
    missing = selectable - set(ns["TEMPLATE_STOP_SEQS"])
    good = not missing
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] {bench:18} every selectable template has a stop seq"
          f"{'' if good else f' -- missing {missing}'}")

# The fix is only real if the call site actually passes it to vLLM.
for bench in BENCHES:
    src = (TASKS / bench / "evaluate.py").read_text(encoding="utf-8")
    wired = ("stop_seqs=template_stop_seqs(args)" in src
             or '"stop": template_stop_seqs(args)' in src)
    ok = ok and wired
    print(f"  [{'PASS' if wired else 'FAIL'}] {bench:18} stop sequences reach the generation call")

print()
print("ALL PASS" if ok else "FAILURES PRESENT")
raise SystemExit(0 if ok else 1)
