"""Evaluate the agent vs. a single-pass baseline on synthetic parts with ground truth.

Metrics
  * part-level: defective-part recall (FAIL or ESCALATE on a defective part
    counts as "caught"), good-part false-reject rate, escalation rate,
    automated accuracy (on parts the system decided itself).
  * defect-level: recall/precision of localized defects (center-in-bbox match).
  * per-degradation breakdown, latency, and a list of failure cases.

Usage: python -m eval.run_eval -n 300 --seed 5000 --out eval/results
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import cv2

from inspectagent.agent import RulePlanner, inspect
from inspectagent.synth import SimulatedStation, make_sample


def _match(pred_bbox, gt_bbox, tol=12) -> bool:
    px, py, pw, ph = pred_bbox
    gx, gy, gw, gh = gt_bbox
    cx, cy = px + pw / 2, py + ph / 2
    return gx - tol <= cx <= gx + gw + tol and gy - tol <= cy <= gy + gh + tol


def score(results):
    agg = defaultdict(lambda: defaultdict(float))
    failures = []
    for mode in ("baseline", "agent"):
        a = agg[mode]
        for r in results:
            gt, pred = r["gt"], r[mode]
            bad = gt["label"] == "FAIL"
            d = pred["disposition"]
            a["n"] += 1
            a["bad"] += bad
            a["good"] += not bad
            if bad and d in ("FAIL", "ESCALATE"):
                a["caught"] += 1
            if bad and d == "PASS":
                a["escaped"] += 1
                failures.append((mode, r["name"], "escaped defect", gt["degradation"]))
            if not bad and d == "FAIL":
                a["false_reject"] += 1
                failures.append((mode, r["name"], "false reject", gt["degradation"]))
            if d == "ESCALATE":
                a["escalated"] += 1
            else:
                a["auto"] += 1
                a["auto_correct"] += (d == "FAIL") == bad
            # defect-level
            gts = [g["bbox"] for g in gt["defects"]]
            preds = [p["bbox"] for p in pred["defects"]]
            a["gt_defects"] += len(gts)
            a["pred_defects"] += len(preds)
            a["tp_gt"] += sum(any(_match(p, g) for p in preds) for g in gts)
            a["tp_pred"] += sum(any(_match(p, g) for g in gts) for p in preds)
            a["ms"] += pred["total_ms"]
            key = f"deg_{gt['degradation']}"
            a[key + "_n"] += 1
            a[key + "_ok"] += (d in ("FAIL", "ESCALATE")) if bad else (d != "FAIL")
    out = {}
    for mode, a in agg.items():
        out[mode] = {
            "parts": int(a["n"]),
            "defective_parts": int(a["bad"]),
            "defective_part_recall": round(a["caught"] / max(a["bad"], 1), 3),
            "escaped_defective_parts": int(a["escaped"]),
            "false_reject_rate": round(a["false_reject"] / max(a["good"], 1), 3),
            "escalation_rate": round(a["escalated"] / a["n"], 3),
            "automated_accuracy": round(a["auto_correct"] / max(a["auto"], 1), 3),
            "defect_recall": round(a["tp_gt"] / max(a["gt_defects"], 1), 3),
            "defect_precision": round(a["tp_pred"] / max(a["pred_defects"], 1), 3),
            "mean_latency_ms": round(a["ms"] / a["n"], 1),
            "safe_outcome_by_degradation": {
                k[4:-2]: round(a[k[:-2] + "_ok"] / a[k], 3)
                for k in sorted(a) if k.startswith("deg_") and k.endswith("_n")},
        }
    return out, failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=5000)
    ap.add_argument("--out", default="eval/results")
    ap.add_argument("--save-failures", type=int, default=8)
    ap.add_argument("--no-recapture", action="store_true",
                    help="evaluate uploads-only mode (agent cannot re-shoot)")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "failures").mkdir(parents=True, exist_ok=True)

    results, planner = [], RulePlanner()
    steps = []
    for seed in range(args.seed, args.seed + args.n):
        img, s = make_sample(seed)
        base = inspect(img, agentic=False)
        ag = inspect(img, planner=planner,
                     camera=None if args.no_recapture else SimulatedStation(seed))
        steps.append(len(ag["trace"]))
        results.append({"name": s.name, "gt": s.to_dict(),
                        "baseline": {k: v for k, v in base.items() if k != "_overlay"},
                        "agent": {k: v for k, v in ag.items() if k != "_overlay"}})
        results[-1]["_ov"] = ag["_overlay"]

    metrics, failures = score(results)
    metrics["agent"]["mean_steps"] = round(statistics.mean(steps), 2)
    tool_use = defaultdict(int)
    for r in results:
        for st in r["agent"]["trace"]:
            tool_use[st["tool"]] += 1
    metrics["agent"]["tool_calls"] = dict(tool_use)

    saved = 0
    for mode, name, why, deg in failures:
        if mode != "agent" or saved >= args.save_failures:
            continue
        r = next(x for x in results if x["name"] == name)
        cv2.imwrite(str(out / "failures" / f"{name}_{why.replace(' ', '_')}_{deg}.png"), r["_ov"])
        saved += 1
    for r in results:
        r.pop("_ov")

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "failures.json").write_text(json.dumps(
        [dict(zip(("mode", "part", "type", "degradation"), f)) for f in failures], indent=1))
    (out / "per_part.json").write_text(json.dumps(
        [{"name": r["name"], "gt": r["gt"]["label"], "degradation": r["gt"]["degradation"],
          "baseline": r["baseline"]["disposition"], "agent": r["agent"]["disposition"],
          "agent_reason": r["agent"]["reason"]} for r in results], indent=1))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
