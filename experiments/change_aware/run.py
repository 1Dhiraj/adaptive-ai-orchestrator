"""Run the change-aware re-execution experiment.

    python run.py --repeats 3 --out results

Results are appended to <out>/results.jsonl (one record per repeat, change,
configuration and threshold) and are skipped on restart, so the run can be
interrupted and resumed. Model responses are cached in <out>/llm_cache.jsonl.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time

from llm import LLM, MODEL
from method import SYSTEMS, check_plan, handle_change, initial_run, translate, value_matches
from suite import SUITE


def log(out, msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(os.path.join(out, "progress.log"), "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def fault_injection():
    """Plans with one injected fault each; returns (detected, total, per-kind)."""
    kinds = {}
    for wf in SUITE:
        final = wf.steps[-1]
        variants = {}
        v = copy.deepcopy(wf); v.steps[0].deps.append(v.steps[-1].id); variants["cycle"] = v
        v = copy.deepcopy(wf); v.steps[-1].deps.append("missing_step"); variants["missing dependency"] = v
        v = copy.deepcopy(wf); dep_types = {v.step(d).out_type for d in final.deps}
        v.steps[-1].accepts = {t for t in ("code", "json", "text") if t not in dep_types} or {"json"}
        variants["type mismatch"] = v
        v = copy.deepcopy(wf); next(iter(v.facts.values())).owner = "ghost_step"; variants["unknown fact owner"] = v
        v = copy.deepcopy(wf); v.token_budget = 500; variants["token budget"] = v
        for kind, variant in variants.items():
            ok = bool(check_plan(variant))
            d, t = kinds.get(kind, (0, 0))
            kinds[kind] = (d + ok, t + 1)
    return kinds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--theta", type=float, default=0.90)
    ap.add_argument("--sweep", default="0.80,0.85,0.95")
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    llm = LLM(os.path.join(args.out, "llm_cache.jsonl"))
    suite = [w for w in SUITE if not args.only or w.id in args.only.split(",")]
    sweep = [float(x) for x in args.sweep.split(",") if x]

    for wf in suite:
        errors = check_plan(wf)
        if errors:
            raise SystemExit(f"{wf.id} failed the plan checker: {errors}")

    res_path = os.path.join(args.out, "results.jsonl")
    done = set()
    if os.path.exists(res_path):
        for line in open(res_path, encoding="utf-8"):
            r = json.loads(line)
            done.add((r["repeat"], r["change"], r["system"], r["theta"]))

    meta = {"model": MODEL, "repeats": args.repeats, "theta": args.theta, "sweep": sweep,
            "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    json.dump(meta, open(os.path.join(args.out, "meta.json"), "w"), indent=2)

    # change translation accuracy (temperature 0, independent of repeats)
    tr_path = os.path.join(args.out, "translation.jsonl")
    if not os.path.exists(tr_path):
        with open(tr_path, "w", encoding="utf-8") as fh:
            for wf in suite:
                for ch in wf.changes:
                    t = translate(llm, wf, ch.request)
                    fh.write(json.dumps({
                        "workflow": wf.id, "domain": wf.domain, "change": ch.id, "category": ch.category,
                        "pred_key": t["key"], "pred_value": t["new_value"], "key": ch.key, "value": ch.new_value,
                        "key_ok": t["key"] == ch.key, "value_ok": value_matches(t["new_value"], ch.new_value),
                    }) + "\n")
        log(args.out, "translation done")

    json.dump({k: list(v) for k, v in fault_injection().items()},
              open(os.path.join(args.out, "plan_checker.json"), "w"), indent=2)

    for rep in range(args.repeats):
        for wf in suite:
            init = initial_run(llm, wf, f"r{rep}:init:{wf.id}")
            log(args.out, f"repeat {rep} {wf.id}: initial run ready")
            for ch in wf.changes:
                tag = f"r{rep}:change:{ch.id}"
                jobs = [(s, args.theta) for s in SYSTEMS] + [("P3", th) for th in sweep]
                for system, theta in jobs:
                    th = theta if system in ("P2", "P3") else None
                    if (rep, ch.id, system, th) in done:
                        continue
                    m = handle_change(llm, wf, init, ch, system, tag, theta)
                    rec = {"repeat": rep, "workflow": wf.id, "domain": wf.domain, "change": ch.id,
                           "category": ch.category, "system": system, "theta": th, **m}
                    with open(res_path, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(rec) + "\n")
                log(args.out, f"repeat {rep} {ch.id} done")
    log(args.out, "ALL DONE")


if __name__ == "__main__":
    main()
