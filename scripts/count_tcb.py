"""Size of the trusted computing base, measured rather than asserted.

A reviewer asked for the smallest trusted component to be explicit.  The
safety property (a recorded action is not dispatched twice; an undetermined
one is not guessed) depends only on the code below.  It does *not* depend on
the model, the prompt, the distiller's allocator, the anchor extractor, the
capability requirement heuristics, or the side-effect detector (which decides
what gets a ledger entry on traces, but not what the dispatcher does with a
typed call).

Counted: logical source lines -- non-blank, non-comment, excluding docstrings
-- of each trusted function or module, via ``ast`` line spans.
"""
import ast
import io
import json
import os
import tokenize

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
OUT = os.path.join(ROOT, "results", "tcb_loc.json")

#: module -> None (whole module) or the set of top-level names that are trusted
TCB = {
    "runtime/dispatcher.py": None,
    "state/protocol.py": None,
    "runtime/durable.py": None,
    "state/integrity.py": None,
    "runtime/delegation.py": None,
    "state/fingerprint.py": {"ARG_ALIASES", "VOLATILE_ARGS", "_NUM_RE",
                             "_TRAILING_ZEROS", "WS_RE", "canon_arg_name",
                             "canon_arg_value", "typed_signature", "fp_typed",
                             "fp_typed_family"},
}
UNTRUSTED = ["state/schema.py", "state/capability.py", "traces/anchors.py",
             "traces/parser.py", "policies/policies.py"]


def logical_lines(path, names=None):
    src = open(path).read()
    tree = ast.parse(src)
    doc_lines = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef,
                             ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and \
                    isinstance(getattr(body[0], "value", None), ast.Constant) \
                    and isinstance(body[0].value.value, str):
                doc_lines.update(range(body[0].lineno,
                                       body[0].end_lineno + 1))
    keep = None
    if names is not None:
        keep = set()
        for node in tree.body:
            nm = getattr(node, "name", None)
            if nm is None and isinstance(node, ast.Assign):
                nm = getattr(node.targets[0], "id", None)
            if nm in names:
                keep.update(range(node.lineno, node.end_lineno + 1))
    code = set()
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE,
                        tokenize.INDENT, tokenize.DEDENT,
                        tokenize.ENDMARKER):
            continue
        for ln in range(tok.start[0], tok.end[0] + 1):
            code.add(ln)
    lines = code - doc_lines
    if keep is not None:
        lines &= keep
    return len(lines)


def main():
    tcb = {m: logical_lines(os.path.join(SRC, m), n) for m, n in TCB.items()}
    unt = {m: logical_lines(os.path.join(SRC, m)) for m in UNTRUSTED}
    total_src = 0
    for dp, _, fs in os.walk(SRC):
        for f in fs:
            if f.endswith(".py"):
                total_src += logical_lines(os.path.join(dp, f))
    out = {"trusted": tcb, "trusted_total": sum(tcb.values()),
           "untrusted_examples": unt, "src_total": total_src}
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
