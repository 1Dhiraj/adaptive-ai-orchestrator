"""Evaluate additional configurations without disturbing a running main run.

Reads the main response cache, writes new responses and records to separate
files (llm_cache_extra.jsonl, results_extra.jsonl).
"""
import argparse, json, os
from llm import LLM
from method import handle_change, initial_run
from suite import SUITE

ap = argparse.ArgumentParser()
ap.add_argument("--reps", default="0")
ap.add_argument("--systems", default="P4")
ap.add_argument("--theta", type=float, default=0.90)
ap.add_argument("--out", default="results")
a = ap.parse_args()
llm = LLM(os.path.join(a.out, "llm_cache_extra.jsonl"))
for line in open(os.path.join(a.out, "llm_cache.jsonl"), encoding="utf-8"):
    if line.strip():
        rec = json.loads(line); llm.cache.setdefault(rec["key"], rec)
path = os.path.join(a.out, "results_extra.jsonl")
done = set()
if os.path.exists(path):
    for line in open(path, encoding="utf-8"):
        r = json.loads(line); done.add((r["repeat"], r["change"], r["system"], r["theta"]))
for rep in [int(x) for x in a.reps.split(",")]:
    for wf in SUITE:
        init = initial_run(llm, wf, f"r{rep}:init:{wf.id}")
        for ch in wf.changes:
            for system in a.systems.split(","):
                if (rep, ch.id, system, a.theta) in done:
                    continue
                m = handle_change(llm, wf, init, ch, system, f"r{rep}:change:{ch.id}", a.theta)
                rec = {"repeat": rep, "workflow": wf.id, "domain": wf.domain, "change": ch.id,
                       "category": ch.category, "system": system, "theta": a.theta, **m}
                with open(path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec) + "\n")
        print("rep", rep, wf.id, "done", flush=True)
