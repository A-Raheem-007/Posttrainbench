"""Exercise template_stop_seqs() without importing inspect_evals."""
import ast, os, pathlib, types

SRC = pathlib.Path(r"C:/Users/arana/Desktop/PostTrainBench/PostTrainBench/.claude/worktrees/ptb-hf-relay-hardening/src/eval/tasks/bfcl/evaluate.py")
tree = ast.parse(SRC.read_text(encoding="utf-8"))

wanted_fn = {"model_type", "template_kwargs", "template_stop_seqs"}
wanted_var = {"TEMPLATE_STOP_SEQS"}
pieces = []
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in wanted_fn:
        pieces.append(ast.get_source_segment(SRC.read_text(encoding="utf-8"), node))
    if isinstance(node, ast.Assign):
        for t in node.targets:
            if isinstance(t, ast.Name) and t.id in wanted_var:
                pieces.append(ast.get_source_segment(SRC.read_text(encoding="utf-8"), node))

ns = {"os": os, "json": __import__("json")}
exec("\n\n".join(pieces), ns)

cases = [
    ("/logs/artifacts/final_model_qwen3",   ["<|im_end|>"]),
    ("/logs/artifacts/final_model_gemma3",  ["<end_of_turn>"]),
    ("/logs/artifacts/final_model_smollm3", ["<|im_end|>"]),
    ("/logs/artifacts/final_model_llama3",  ["<|eot_id|>"]),
]
ok = True
for path, expected in cases:
    args = types.SimpleNamespace(model_path=path, templates_dir="/tests/templates")
    got = ns["template_stop_seqs"](args)
    good = got == expected
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] {os.path.basename(path):22} -> {got}")

# Every template the mapping can produce must have an entry, or a family
# silently gets no stop sequence and regresses to the bug we just fixed.
templates = set()
for path in cases:
    args = types.SimpleNamespace(model_path=path[0], templates_dir="/t")
    templates.add(os.path.basename(ns["template_kwargs"](args)["chat_template"]))
missing = templates - set(ns["TEMPLATE_STOP_SEQS"])
print(f"  [{'PASS' if not missing else 'FAIL'}] every template family has a stop seq"
      f"{'' if not missing else f' -- missing {missing}'}")
ok = ok and not missing

print()
print("ALL PASS" if ok else "FAILURES PRESENT")
raise SystemExit(0 if ok else 1)
