"""Aggregate results.jsonl into the numbers, tables and figure used in the paper.

    python report.py --out results
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict

from method import SYSTEMS
from suite import SUITE

CATS = ["root", "mid", "leaf", "cosmetic", "hidden"]
CAT_LABEL = {"root": "Root", "mid": "Mid-graph", "leaf": "Leaf", "cosmetic": "Cosmetic", "hidden": "Hidden dep."}


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def sd(xs):
    xs = list(xs)
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def ms(xs):
    return {"mean": mean(xs), "sd": sd(xs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--reps", default="", help="comma-separated repetitions to include (default: all)")
    args = ap.parse_args()
    meta = json.load(open(os.path.join(args.out, "meta.json")))
    theta = meta["theta"]
    recs = [json.loads(l) for l in open(os.path.join(args.out, "results.jsonl"), encoding="utf-8")]
    extra = os.path.join(args.out, "results_extra.jsonl")
    if os.path.exists(extra):
        recs += [json.loads(l) for l in open(extra, encoding="utf-8")]
    if args.reps:
        keep = {int(x) for x in args.reps.split(",")}
        recs = [r for r in recs if r["repeat"] in keep]
    reps = sorted({r["repeat"] for r in recs})

    def default(r):
        return r["system"] in SYSTEMS and (r["theta"] is None or abs(r["theta"] - theta) < 1e-9)

    base = [r for r in recs if default(r)]
    summary = {"meta": meta, "repeats_completed": len(reps), "changes_per_repeat": {}}

    # ---------------------------------------------------------------- efficiency + correctness
    eff, cor = {}, {}
    for s in SYSTEMS:
        per = defaultdict(lambda: defaultdict(list))
        for r in base:
            if r["system"] == s:
                per[r["repeat"]]["r"].append(r)
        rows = {k: [] for k in ["steps", "tokens", "check", "total", "calls", "seconds", "missed", "unnec", "stale", "cutoffs"]}
        for rep, d in per.items():
            rs = d["r"]
            summary["changes_per_repeat"][rep] = len(rs)
            rows["steps"].append(mean(r["steps"] for r in rs))
            rows["tokens"].append(mean(r["tokens"] for r in rs))
            rows["check"].append(mean(r["check_tokens"] for r in rs))
            rows["total"].append(mean(r["tokens"] + r["check_tokens"] for r in rs))
            rows["calls"].append(mean(r["calls"] for r in rs))
            rows["seconds"].append(mean(r["seconds"] for r in rs))
            rows["missed"].append(100 * sum(r["missed"] for r in rs) / max(1, sum(r["must_count"] for r in rs)))
            rows["unnec"].append(100 * sum(r["unnecessary"] for r in rs) / max(1, sum(r["rerun_count"] for r in rs)))
            rows["stale"].append(100 * sum(r["stale"] for r in rs) / max(1, sum(r["must_count"] for r in rs)))
            rows["cutoffs"].append(mean(len(r["cutoffs"]) for r in rs))
        eff[s] = {k: ms(v) for k, v in rows.items()}
    summary["systems"] = eff

    b1_total = {rep: mean(r["tokens"] for r in base if r["system"] == "B1" and r["repeat"] == rep) for rep in reps}
    b1_steps = {rep: mean(r["steps"] for r in base if r["system"] == "B1" and r["repeat"] == rep) for rep in reps}
    savings = {}
    for s in SYSTEMS:
        tok, stp = [], []
        for rep in reps:
            rs = [r for r in base if r["system"] == s and r["repeat"] == rep]
            tok.append(100 * (1 - mean(r["tokens"] + r["check_tokens"] for r in rs) / b1_total[rep]))
            stp.append(100 * (1 - mean(r["steps"] for r in rs) / b1_steps[rep]))
        savings[s] = {"tokens_saved_pct": ms(tok), "steps_saved_pct": ms(stp)}
    summary["savings_vs_B1"] = savings

    # ---------------------------------------------------------------- per category
    by_cat = {}
    for s in SYSTEMS:
        by_cat[s] = {}
        for c in CATS:
            rs = [r for r in base if r["system"] == s and r["category"] == c]
            by_cat[s][c] = {
                "steps": mean(r["steps"] for r in rs),
                "missed_pct": 100 * sum(r["missed"] for r in rs) / max(1, sum(r["must_count"] for r in rs)),
                "unnec_pct": 100 * sum(r["unnecessary"] for r in rs) / max(1, sum(r["rerun_count"] for r in rs)),
                "tokens": mean(r["tokens"] + r["check_tokens"] for r in rs),
            }
    summary["by_category"] = by_cat
    b3_hidden = [r for r in base if r["system"] == "B3" and r["category"] == "hidden"]
    summary["b3_hidden_changes_with_miss"] = [sum(1 for r in b3_hidden if r["missed"] > 0), len(b3_hidden)]
    summary["b3_nonhidden_missed_steps"] = sum(r["missed"] for r in base if r["system"] == "B3" and r["category"] != "hidden")
    p3m = [r for r in base if r["system"] == "P3" and r["missed"] > 0]
    summary["p3_missed_changes_without_rerun"] = [sum(1 for r in p3m if not r["rerun"]), len(p3m)]
    summary["p3_missed_categories"] = sorted({r["category"] for r in p3m})
    summary["must_size_by_category"] = {c: mean(r["must_count"] for r in base if r["system"] == "B1" and r["category"] == c) for c in CATS}

    # ---------------------------------------------------------------- threshold sweep (P3)
    sweep = {}
    for th in sorted(set(meta["sweep"]) | {theta}):
        tok, miss, judge, cut = [], [], [], []
        for rep in reps:
            rs = [r for r in recs if r["system"] == "P3" and r["theta"] is not None and abs(r["theta"] - th) < 1e-9 and r["repeat"] == rep]
            if not rs:
                continue
            tok.append(100 * (1 - mean(r["tokens"] + r["check_tokens"] for r in rs) / b1_total[rep]))
            miss.append(100 * sum(r["missed"] for r in rs) / max(1, sum(r["must_count"] for r in rs)))
            judge.append(mean(r["judge_calls"] for r in rs))
            cut.append(mean(len(r["cutoffs"]) for r in rs))
        sweep[f"{th:.2f}"] = {"tokens_saved_pct": ms(tok), "missed_pct": ms(miss), "judge_calls": ms(judge), "cutoffs": ms(cut)}
    summary["theta_sweep"] = sweep

    # ---------------------------------------------------------------- translation + plan checker + suite
    tr = [json.loads(l) for l in open(os.path.join(args.out, "translation.jsonl"), encoding="utf-8")]
    trans = {}
    for dom in sorted({t["domain"] for t in tr}) + ["All"]:
        ts = [t for t in tr if dom == "All" or t["domain"] == dom]
        trans[dom] = {
            "n": len(ts),
            "key_pct": 100 * mean(t["key_ok"] for t in ts),
            "value_pct": 100 * mean(t["value_ok"] for t in ts),
            "both_pct": 100 * mean(t["key_ok"] and t["value_ok"] for t in ts),
            "errors": [t for t in ts if not (t["key_ok"] and t["value_ok"])],
        }
    summary["translation"] = trans
    summary["plan_checker"] = json.load(open(os.path.join(args.out, "plan_checker.json")))
    suite = {}
    for wf in SUITE:
        d = suite.setdefault(wf.domain, {"workflows": 0, "steps": [], "changes": 0, "facts": []})
        d["workflows"] += 1
        d["steps"].append(len(wf.steps))
        d["changes"] += len(wf.changes)
        d["facts"].append(len(wf.facts))
    summary["suite"] = {k: {"workflows": v["workflows"], "avg_steps": mean(v["steps"]), "changes": v["changes"],
                            "avg_facts": mean(v["facts"]), "examples": [w.title for w in SUITE if w.domain == k]}
                        for k, v in suite.items()}

    # model call statistics from the cache
    cache = [json.loads(l) for l in open(os.path.join(args.out, "llm_cache.jsonl"), encoding="utf-8")]
    summary["unique_model_calls"] = len(cache)
    summary["total_model_seconds"] = sum(c["seconds"] for c in cache)

    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=2)
    make_figure(by_cat, os.path.join(args.out, "fig3_steps_by_category"))
    print(json.dumps({k: summary[k] for k in ["repeats_completed", "systems", "savings_vs_B1", "theta_sweep"]}, indent=1)[:6000])


def make_figure(by_cat, path):
    import fitz
    W, H = 520, 300
    left, right, top, bottom = 46, 10, 16, 62
    plot_w, plot_h = W - left - right, H - top - bottom
    ymax = max(by_cat[s][c]["steps"] for s in SYSTEMS for c in CATS)
    ymax = math.ceil(ymax + 0.5)
    colors = {"B1": "#9E9E9E", "B2": "#6FA8DC", "B3": "#3D6FA6", "P1": "#F6B26B", "P2": "#93C47D", "P3": "#C0504D", "P4": "#7B3F99"}
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
           f'<rect width="{W}" height="{H}" fill="#fff"/>']
    font = 'font-family="Times New Roman"'
    for i in range(ymax + 1):
        y = top + plot_h - plot_h * i / ymax
        out.append(f'<line x1="{left}" y1="{y:.1f}" x2="{W - right}" y2="{y:.1f}" stroke="#e3e3e3" stroke-width="1"/>')
        out.append(f'<text x="{left - 6}" y="{y + 4:.1f}" {font} font-size="12" text-anchor="end">{i}</text>')
    out.append(f'<text x="14" y="{top + plot_h / 2}" {font} font-size="13" text-anchor="middle" transform="rotate(-90 14 {top + plot_h / 2})">Step executions</text>')
    group_w = plot_w / len(CATS)
    bar_w = group_w * 0.8 / len(SYSTEMS)
    for gi, c in enumerate(CATS):
        gx = left + gi * group_w + group_w * 0.1
        for si, s in enumerate(SYSTEMS):
            v = by_cat[s][c]["steps"]
            h = plot_h * v / ymax
            x = gx + si * bar_w
            out.append(f'<rect x="{x:.1f}" y="{top + plot_h - h:.1f}" width="{bar_w - 1:.1f}" height="{h:.1f}" fill="{colors[s]}"/>')
        out.append(f'<text x="{left + gi * group_w + group_w / 2:.1f}" y="{top + plot_h + 16}" {font} font-size="13" text-anchor="middle">{CAT_LABEL[c]}</text>')
    out.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{W - right}" y2="{top + plot_h}" stroke="#000" stroke-width="1"/>')
    lx = left
    for s in SYSTEMS:
        out.append(f'<rect x="{lx}" y="{H - 24}" width="14" height="12" fill="{colors[s]}"/>')
        out.append(f'<text x="{lx + 18}" y="{H - 14}" {font} font-size="13">{s}</text>')
        lx += 64
    out.append("</svg>")
    with open(path + ".svg", "w", encoding="utf-8") as fh:
        fh.write("".join(out))
    doc = fitz.open(path + ".svg")
    doc[0].get_pixmap(matrix=fitz.Matrix(4, 4)).save(path + ".png")


if __name__ == "__main__":
    main()
