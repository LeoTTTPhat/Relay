"""Verify the release manifest and the packaged evaluation inputs."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

def main():
    manifest = ROOT / "MANIFEST.sha256"
    checked = 0
    for line in manifest.read_text().splitlines():
        expected, relative = line.split("  ", 1)
        path = ROOT / relative
        h = hashlib.sha256(path.read_bytes()).hexdigest()
        if h != expected:
            raise SystemExit("Integrity mismatch: " + relative)
        checked += 1
    forbidden = {".tex", ".bib", ".bbl", ".aux", ".out"}
    for path in ROOT.rglob("*"):
        if path.is_file() and ".git" not in path.parts:
            if path.suffix.lower() in forbidden or "paper" in path.relative_to(ROOT).parts or "mobisys" in path.relative_to(ROOT).parts:
                raise SystemExit("Unexpected release file: " + str(path))
    counts = {"verified_files": checked}
    if (ROOT / "data/benchmark/items.jsonl").exists():
        for folder in ["benchmark", "benchmark_rev2611"]:
            rows = [json.loads(line) for line in (ROOT / "data" / folder / "items.jsonl").read_text().splitlines()]
            assert len(rows) == 1036, (folder, len(rows))
            assert len({r["id"] for r in rows}) == len(rows)
            counts[folder] = len(rows)
        counts["counterfactual_variants"] = sum(len(r.get("counterfactual_variants", [])) for r in rows)
        assert counts["counterfactual_variants"] == 2254
    else:
        sys.path.insert(0, str(ROOT / "src"))
        from traces.parser import load_trajectories
        from tok.tokenizer import available
        rows = load_trajectories()
        assert len(rows) == 184, len(rows)
        assert available(), "Exact tokenizer is unavailable"
        counts.update(trajectories=len(rows), exact_tokenizer=True)
        for lang in ["de", "es", "fr"]:
            assert (ROOT / "results" / "e24_mt" / ("table_" + lang + ".json.gz")).exists()
    print(json.dumps(counts, indent=2))

if __name__ == "__main__":
    main()
