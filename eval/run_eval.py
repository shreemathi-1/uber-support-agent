"""
Run every system (trivial, simple, ours) over the golden set, score them, and write eval/results/.
Runs end to end without labels (plumbing check) and fully offline under CACHE_ONLY=1 once the cache is warm.
"""
import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baselines.simple import SILVER_INFO_PATH, SimpleBaseline, load_or_fit  # noqa: E402
from baselines.trivial import TrivialBaseline  # noqa: E402
from eval.judge import judge  # noqa: E402
from eval.metrics import auto_handle_rate, escalation_metrics, intent_metrics  # noqa: E402
from pipeline import llm  # noqa: E402
from pipeline.cache import CacheMissError  # noqa: E402
from pipeline.draft import MAX_FACT_DISTANCE  # noqa: E402
from pipeline.llm import DailyQuotaExceeded  # noqa: E402
from pipeline.models import ChatRequest, Turn  # noqa: E402
from pipeline.retrieve import get_collection  # noqa: E402
from pipeline.run import run  # noqa: E402
from pipeline.taxonomy import Intent  # noqa: E402

GOLDEN = Path("data/golden_set.csv")
RESULTS = Path("eval/results")
SYSTEMS = ["trivial", "simple", "ours"]
LABELS = [i.value for i in Intent]


class Ours:
    def predict(self, row: dict) -> dict:
        history = [Turn(role="user", content=row["history"])] if row["history"] else []
        r = run(ChatRequest(message=row["text"], history=history))
        return {"intent": r.intent.value, "escalate": r.action == "escalate", "reason": r.reason, "reply": r.reply,
                "retrieved_ids": r.retrieved_ids, "latency_ms": r.latency_ms}


def load_rows(n: int | None) -> list[dict]:
    with GOLDEN.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[:n] if n else rows


def truth_escalate(row: dict) -> bool | None:
    v = row["escalate"].strip().lower()
    return None if not v else v in ("y", "yes", "true", "1")


def facts_seen_by_drafter(retrieved_ids: list[str]) -> list[str]:
    """Help chunks from retrieved_ids ("help:<id>:<distance>") within the drafter's distance filter, as text."""
    ids = []
    for entry in retrieved_ids:
        kind, chunk_id, dist = entry.split(":")
        if kind == "help" and float(dist) <= MAX_FACT_DISTANCE:
            ids.append(chunk_id)
    if not ids:
        return []
    got = get_collection("help").get(ids=ids, include=["documents", "metadatas"])
    return [f"[{m['title']}] {d}" for d, m in zip(got["documents"], got["metadatas"])]


def predict_all(name: str, system, rows: list[dict]) -> list[dict]:
    preds = []
    for i, row in enumerate(rows, 1):
        t0 = time.perf_counter()
        try:
            p = system.predict(row)
        except CacheMissError as err:
            sys.exit(f"CACHE_ONLY=1: cache miss on row {row['id']} in system {name}: {err}")
        except DailyQuotaExceeded as err:
            sys.exit(f"stopped at row {row['id']} in system {name}: {err}")
        p.setdefault("latency_ms", int((time.perf_counter() - t0) * 1000))
        preds.append({"id": row["id"], **p})
        print(f"\r  {name}: {i}/{len(rows)}", end="", file=sys.stderr)
    print(file=sys.stderr)
    return preds


def judge_ours(preds: list[dict], rows: list[dict]) -> None:
    """Judge auto-handled replies only; the escalation template is not a drafted reply."""
    by_id = {r["id"]: r for r in rows}
    for i, p in enumerate(preds, 1):
        if p["escalate"]:
            continue
        try:
            p["judge"] = judge(by_id[p["id"]]["text"], p["reply"], facts_seen_by_drafter(p.get("retrieved_ids", [])))
        except CacheMissError as err:
            sys.exit(f"CACHE_ONLY=1: judge cache miss on row {p['id']}: {err}")
        except Exception as err:  # keep going; the summary reports how many failed
            p["judge_error"] = f"{type(err).__name__}: {err}"
        print(f"\r  judge: {i}/{len(preds)}", end="", file=sys.stderr)
    print(file=sys.stderr)


def summarise(name: str, preds: list[dict], rows: list[dict]) -> dict:
    by_id = {r["id"]: r for r in rows}
    labelled = [(by_id[p["id"]]["intent"].strip(), p["intent"]) for p in preds if by_id[p["id"]]["intent"].strip()]
    esc = [(truth_escalate(by_id[p["id"]]), p["escalate"], p["reason"]) for p in preds if truth_escalate(by_id[p["id"]]) is not None]
    judged = [p["judge"]["total"] for p in preds if p.get("judge")]
    silver = json.loads(SILVER_INFO_PATH.read_text()) if name == "simple" and SILVER_INFO_PATH.exists() else None
    return {
        "system": name,
        "n": len(preds),
        "silver_labels": silver,
        "intent": intent_metrics([t for t, _ in labelled], [p for _, p in labelled], LABELS, RESULTS / f"{name}_confusion.png") if labelled else None,
        "escalation": escalation_metrics([t for t, _, _ in esc], [p for _, p, _ in esc], [r for _, _, r in esc]) if esc else None,
        "auto_handle_rate": auto_handle_rate(preds),
        "judge": {"n": len(judged), "mean_total": round(statistics.mean(judged), 2),
                  "errors": sum(1 for p in preds if p.get("judge_error"))} if judged else None,
        "mean_latency_ms": round(statistics.mean(p["latency_ms"] for p in preds), 1),
    }


def fmt(x, digits: int = 3) -> str:
    return "-" if x is None else f"{x:.{digits}f}" if isinstance(x, float) else str(x)


def write_summary(summaries: list[dict], labelled: bool) -> str:
    (RESULTS / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    lines = ["# Eval summary", ""]
    if not labelled:
        lines += ["**NO LABELS - plumbing check only.** Intent and escalation metrics are empty until data/golden_set.csv is labelled.", ""]
    lines += ["| system | n | intent acc | macro-F1 | esc P | esc R | esc F1 | auto-handle rate | mean judge total (n) | mean latency ms |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for s in summaries:
        i, e, j = s["intent"], s["escalation"], s["judge"]
        lines.append(f"| {s['system']} | {s['n']} | {fmt(i and i['accuracy'])} | {fmt(i and i['macro_f1'])} | "
                     f"{fmt(e and e['precision'])} | {fmt(e and e['recall'])} | {fmt(e and e['f1'])} | "
                     f"{fmt(s['auto_handle_rate'])} | {fmt(j and j['mean_total'], 2)} ({j['n'] if j else 0}) | {fmt(s['mean_latency_ms'], 1)} |")
    for s in summaries:
        if s["escalation"] and s["escalation"]["by_reason"]:
            lines += ["", f"## {s['system']}: escalations by reason", "", "| reason | fired | true escalations | false escalations |", "|---|---|---|---|"]
            lines += [f"| {r} | {c['count']} | {c['true_escalations']} | {c['false_escalations']} |" for r, c in s["escalation"]["by_reason"].items()]
    text = "\n".join(lines) + "\n"
    (RESULTS / "summary.md").write_text(text)
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--systems", default=",".join(SYSTEMS))
    ap.add_argument("--skip-judge", action="store_true")
    ap.add_argument("--n", type=int, default=None, help="first N golden rows (default: all)")
    ap.add_argument("--silver-n", type=int, default=1000, help="pairs to silver-label for the simple baseline (decision #33)")
    args = ap.parse_args()
    llm.batch_mode()
    RESULTS.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.n)
    labelled = any(r["intent"].strip() for r in rows)
    if not labelled:
        print("NO LABELS - plumbing check only")

    systems = {"trivial": lambda: TrivialBaseline([r["intent"] for r in rows if r["intent"].strip()]),
               "simple": lambda: SimpleBaseline(load_or_fit(args.silver_n)),
               "ours": Ours}
    summaries = []
    for name in args.systems.split(","):
        preds = predict_all(name, systems[name](), rows)
        if name == "ours" and not args.skip_judge:
            judge_ours(preds, rows)
        with (RESULTS / f"{name}_predictions.jsonl").open("w") as f:
            f.writelines(json.dumps(p, ensure_ascii=False) + "\n" for p in preds)
        summaries.append(summarise(name, preds, rows))
    print(write_summary(summaries, labelled))


if __name__ == "__main__":
    main()
