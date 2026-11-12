"""Benchmark latency and field accuracy of the certificate pipeline.

Run from the project root (the folder that contains `app/`):

    python benchmark.py                       # OCR + rules only (no AI)
    python benchmark.py --ollama              # include Qwen/Ollama steps
    python benchmark.py --repeat 3            # repeat each file, report p50/p95
    python benchmark.py --labels data/labels/ground_truth.json --samples data

Outputs a console report plus benchmark_results.json / benchmark_results.csv.
Use the same command before and after a change to compare old vs new flow.
"""
import argparse
import csv
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))

import app.pipeline as pl  # noqa: E402

# ---------------------------------------------------------------- stage timing
STAGES = defaultdict(float)


def timed(name, fn):
    def wrapper(*a, **k):
        t = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            STAGES[name] += time.perf_counter() - t
    return wrapper


pl.local_parse = timed("ocr_parse", pl.local_parse)
pl.cloud_parse = timed("cloud_parse", pl.cloud_parse)
pl.classify = timed("classify", pl.classify)
pl.extract_fields = timed("rule_extract", pl.extract_fields)
pl.validate_and_normalize = timed("validate", pl.validate_and_normalize)
try:  # AI stages are imported lazily inside process_document, so patch the modules
    import app.extraction.vision as vis
    import app.classification.llm_classifier as llmc
    import app.validation.reviewer as rev
    vis.render_pages = timed("render_for_ai", vis.render_pages)
    vis.extract_vision = timed("qwen_extract", vis.extract_vision)
    llmc.llm_classify = timed("qwen_classify", llmc.llm_classify)
    rev.review = timed("qwen_review", rev.review)
except Exception:
    pass

# ---------------------------------------------------------------- comparison
def norm(v):
    return None if v in (None, "", "None") else str(v).strip().casefold()


def judge(expected, got):
    """correct | missing | wrong | extra  (wrong = confident wrong value)."""
    e, g = norm(expected), norm(got)
    if e == g:
        return "correct"
    if e is not None and g is None:
        return "missing"
    if e is None and g is not None:
        return "extra"
    return "wrong"


def expected_fields(row):
    return {
        "certificate_name": row.get("certificate_name"),
        "start_date": row.get("issue_date") or row.get("start_date")
        or row.get("training_date") or row.get("examination_date"),
        "end_date": row.get("expire_date"),
        "license_number": row.get("license_number"),
    }


def pct(values, p):
    if not values:
        return 0.0
    values = sorted(values)
    k = min(len(values) - 1, round((p / 100) * (len(values) - 1)))
    return values[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/labels/ground_truth.json")
    ap.add_argument("--samples", default="data", help="folder the label 'file' paths are relative to")
    ap.add_argument("--ollama", action="store_true", help="enable Qwen/Ollama steps")
    ap.add_argument("--repeat", type=int, default=1)
    args = ap.parse_args()

    rows = json.loads(Path(args.labels).read_text(encoding="utf-8"))["examples"]
    results, counts = [], defaultdict(lambda: defaultdict(int))

    for row in rows:
        path = Path(args.samples) / row["file"]
        if not path.exists():
            print(f"SKIP (missing): {path}")
            continue
        exp = expected_fields(row)
        times = []
        out, err = None, None
        for _ in range(args.repeat):
            STAGES.clear()
            t = time.perf_counter()
            try:
                out = pl.process_document(path, args.ollama)
            except Exception as exc:  # keep going; record failure
                err = f"{type(exc).__name__}: {exc}"
            times.append(time.perf_counter() - t)
        stages = dict(STAGES)  # from the last repeat
        data = (out or {}).get("data") or {}
        verdicts = {k: judge(exp[k], data.get(k)) for k in exp}
        for k, v in verdicts.items():
            counts[k][v] += 1
        results.append({
            "file": row["file"],
            "seconds_p50": round(statistics.median(times), 2),
            "seconds_max": round(max(times), 2),
            "pages": (out or {}).get("processing", {}).get("page_count"),
            "requires_review": (out or {}).get("requires_review"),
            "used_ai": (out or {}).get("processing", {}).get("ai_used", False),
            "ocr_path": " > ".join(
                a["engine"] + ("(passed)" if a["passed"] else "(error)" if a.get("error") else "(escalated)")
                for a in (out or {}).get("processing", {}).get("ocr_attempts", [])),
            "review_reasons": (out or {}).get("review_reasons", []),
            "stages": {k: round(v, 2) for k, v in stages.items()},
            "expected": exp,
            "got": {k: data.get(k) for k in exp},
            "verdicts": verdicts,
            "warnings": (out or {}).get("warnings", []),
            "error": err,
        })

    # ------------------------------------------------------------ report
    print(f"\n{'file':45} {'sec':>6} {'pg':>3} {'review':>6}  name  start end   lic   ocr path / ai")
    for r in results:
        v = r["verdicts"]
        mark = lambda x: {"correct": "ok ", "missing": "MIS", "wrong": "BAD", "extra": "EXT"}[x]
        print(f"{r['file'][:45]:45} {r['seconds_p50']:6.1f} {str(r['pages']):>3} {str(r['requires_review']):>6}  "
              f"{mark(v['certificate_name'])}   {mark(v['start_date'])}   {mark(v['end_date'])}   {mark(v['license_number'])}"
              + f"   {r['ocr_path']}" + (" +AI" if r["used_ai"] else "")
              + (f"   ERROR {r['error']}" if r["error"] else ""))

    secs = [r["seconds_p50"] for r in results]
    print(f"\nLatency over {len(secs)} files: p50={pct(secs,50):.1f}s  p95={pct(secs,95):.1f}s  max={max(secs, default=0):.1f}s")

    total = defaultdict(float)
    for r in results:
        for k, v in r["stages"].items():
            total[k] += v
    if total:
        whole = sum(total.values()) or 1
        print("Where time goes (summed over files):")
        for k, v in sorted(total.items(), key=lambda kv: -kv[1]):
            print(f"  {k:15} {v:7.1f}s  {100*v/whole:4.0f}%")

    print("\nAccuracy by field (correct / missing / WRONG / extra):")
    for k, c in counts.items():
        n = sum(c.values())
        print(f"  {k:17} {c['correct']}/{n} correct   missing={c['missing']}  wrong={c['wrong']}  extra={c['extra']}")
    wrong = sum(c["wrong"] for c in counts.values())
    print(f"\nConfident-wrong values: {wrong}  (these are the dangerous ones: they pass without review)")
    unflagged = [r["file"] for r in results
                 if any(v in ("wrong", "extra") for v in r["verdicts"].values()) and r["requires_review"] is False]
    if unflagged:
        print("Wrong AND not flagged for review:", ", ".join(unflagged))

    Path("benchmark_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    with open("benchmark_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "seconds_p50", "pages", "requires_review", "ocr_path", "ai_used", "name", "start_date", "end_date", "license_number"])
        for r in results:
            v = r["verdicts"]
            w.writerow([r["file"], r["seconds_p50"], r["pages"], r["requires_review"], r["ocr_path"], r["used_ai"],
                        v["certificate_name"], v["start_date"], v["end_date"], v["license_number"]])
    print("\nSaved benchmark_results.json and benchmark_results.csv")


if __name__ == "__main__":
    main()