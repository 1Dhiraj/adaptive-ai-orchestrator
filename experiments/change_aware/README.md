# Change-aware re-execution experiment

This folder implements and evaluates the two mechanisms described in the paper
*Adaptive Multi-Agent Orchestration with Signature-Based Selective
Re-execution for LLM Workflows*:

- **Assumption tracking**: every step records which requirement facts it relied
  on (declared by the agent and detected in its output), and the values of those
  facts are part of the step's signature. A change reaches every step that used
  the fact, even when no graph edge connects it to the step that owns the fact.
- **Semantic early cutoff**: when a step re-runs, its new output is compared
  with the cached one by a checker chosen by output type (Python syntax tree or
  normalized code, JSON fields, bag-of-words cosine plus an LLM judge). If the
  outputs are equivalent, the cached output is kept and propagation stops.

It also contains a change translator (plain-English change request to a fact
delta) and a typed plan checker.

## Files

| File | Purpose |
|---|---|
| `method.py` | Fact store, assumption detection, signatures, equivalence checks, change translation, plan checker, and the re-execution policies |
| `suite.py` | 10 workflows in five domains, each with five requirement changes (root, mid-graph, leaf, cosmetic, hidden dependency) and ground-truth labels |
| `llm.py` | Minimal Ollama client with a persistent response cache and per-trial seeds |
| `run.py` | Main experiment: translation accuracy, plan-checker fault injection, and every configuration for every change and repetition |
| `run_extra.py` | Evaluates additional configurations (P4) without disturbing a running main run |
| `report.py` | Aggregates results into `summary.json` and draws the per-category figure |

## Configurations

| ID | Policy |
|---|---|
| B1 | Restart-all |
| B2 | Cone invalidation (owner step and all descendants) |
| B3 | Exact signatures (definition + dependency outputs), owner forced |
| P1 | Assumption-aware signatures, exact-match cutoff |
| P2 | B3 with semantic early cutoff |
| P3 | Assumption tracking and semantic early cutoff |
| P4 | P3 with the owner step always re-executed (added after analysing the first repetition) |

## Running

Requires Python 3.10+ and [Ollama](https://ollama.com) with the model pulled:

```bash
ollama pull qwen2.5:7b
python run.py --repeats 3 --theta 0.90 --sweep 0.80,0.85,0.95 --out results
python run_extra.py --reps 0,1,2 --systems P4 --out results
python report.py --out results
```

Runs can be interrupted and resumed: completed records are skipped and every
model response is cached in `results/llm_cache.jsonl`. On an RTX 4070 one
repetition (50 changes, all configurations) takes about 20 minutes.

## Metrics

- **Steps / tokens / model time** per change (prompt plus completion tokens as
  reported by Ollama; judge tokens are reported separately).
- **Missed re-run rate**: share of must-re-run steps that were reused.
- **Unnecessary re-run rate**: share of executed steps that did not need to run.
- **Stale mentions**: share of must-re-run steps whose final output still
  contains the old value (or an alias) of the changed fact.

Ground truth: each step is labelled with the facts its task depends on. For a
change, the must-re-run set is the labelled steps plus their descendants
(labelled steps only for cosmetic changes).
