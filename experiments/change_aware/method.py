"""Change-aware re-execution: fact store, assumption tracking, signatures,
type-aware equivalence checking, change translation and plan checking.

The six re-execution policies compared in the paper are implemented in
``handle_change``:

  B1  restart-all          every step re-runs after any change
  B2  cone                 the owner step and all of its descendants re-run
  B3  exact signatures     structural signature (definition + dependency
                           outputs), owner forced, exact-match cutoff
  P1  assumptions          signature also covers assumed fact values,
                           exact-match cutoff
  P2  semantic cutoff      B3 with type-aware equivalence cutoff
  P3  full                 assumptions + semantic cutoff
  P4  owner-forced         P3, and the step that owns the changed fact is
                           always re-executed (variant added after analysing
                           the first repetition)
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field

from llm import LLM, stable_seed

SYSTEMS = ["B1", "B2", "B3", "P1", "P2", "P3", "P4"]
TYPES = {"code", "json", "text"}


# --------------------------------------------------------------------------- data model
@dataclass
class Fact:
    key: str
    value: str
    owner: str
    aliases: list = field(default_factory=list)


@dataclass
class Step:
    id: str
    role: str
    task: str
    out_type: str
    deps: list
    uses: set                      # ground-truth requirement keys (authors' labels)
    accepts: set = field(default_factory=lambda: set(TYPES))


@dataclass
class Change:
    id: str
    category: str                  # root | mid | leaf | cosmetic | hidden
    request: str
    key: str
    new_value: str


@dataclass
class Workflow:
    id: str
    domain: str
    title: str
    facts: dict                    # key -> Fact
    steps: list                    # list[Step], any order
    changes: list                  # list[Change]
    token_budget: int = 20000

    def step(self, sid):
        return next(s for s in self.steps if s.id == sid)


# --------------------------------------------------------------------------- graph helpers
def topo_levels(wf: Workflow) -> list:
    remaining = {s.id: set(s.deps) for s in wf.steps}
    done, levels = set(), []
    while remaining:
        ready = sorted(sid for sid, d in remaining.items() if d <= done)
        if not ready:
            raise ValueError("cycle")
        levels.append(ready)
        done |= set(ready)
        for sid in ready:
            del remaining[sid]
    return levels


def descendants(wf: Workflow, roots: set) -> set:
    out, frontier = set(), set(roots)
    while frontier:
        nxt = {s.id for s in wf.steps if set(s.deps) & frontier} - out
        out |= nxt
        frontier = nxt
    return out


def consumers(wf: Workflow, sid: str) -> list:
    return [s for s in wf.steps if sid in s.deps]


def must_rerun(wf: Workflow, change: Change) -> set:
    """Ground truth: direct users of the changed fact, plus their descendants
    unless the change is cosmetic (by definition it does not affect consumers)."""
    users = {s.id for s in wf.steps if change.key in s.uses}
    if change.category == "cosmetic":
        return users
    return users | descendants(wf, users)


# --------------------------------------------------------------------------- plan checker
def check_plan(wf: Workflow) -> list:
    errors = []
    ids = [s.id for s in wf.steps]
    if len(ids) != len(set(ids)):
        errors.append("duplicate step id")
    known = set(ids)
    for s in wf.steps:
        if s.out_type not in TYPES:
            errors.append(f"{s.id}: unknown output type {s.out_type}")
        for d in s.deps:
            if d not in known:
                errors.append(f"{s.id}: missing dependency {d}")
            elif wf.step(d).out_type not in s.accepts:
                errors.append(f"{s.id}: cannot accept {wf.step(d).out_type} from {d}")
    for f in wf.facts.values():
        if f.owner not in known:
            errors.append(f"fact {f.key}: owner {f.owner} does not exist")
    if not errors:
        try:
            topo_levels(wf)
        except ValueError:
            errors.append("dependency cycle")
    estimate = sum(len(build_prompt(wf, s, {k: f.value for k, f in wf.facts.items()}, {})[1]) // 4 + 500
                   for s in wf.steps) if not errors else 0
    if estimate > wf.token_budget:
        errors.append(f"estimated {estimate} tokens exceeds budget {wf.token_budget}")
    return errors


# --------------------------------------------------------------------------- prompting
FORMAT = {
    "code": "Return only the code, at most 35 lines, in one fenced code block.",
    "json": "Return only valid JSON with no code fence and no commentary.",
    "text": "Write at most 150 words.",
}


def build_prompt(wf: Workflow, step: Step, fact_values: dict, dep_outputs: dict):
    reqs = "\n".join(f"- {k}: {v}" for k, v in sorted(fact_values.items()))
    deps = "\n\n".join(f"### Output of step '{d}'\n{dep_outputs[d]}" for d in step.deps if d in dep_outputs)
    user = (
        f"Project: {wf.title}\n\nProject requirements:\n{reqs}\n\n"
        f"Your task: {step.task}\n\nInputs from earlier steps:\n{deps or '(none)'}\n\n"
        f"{FORMAT[step.out_type]}\n"
        "After the deliverable, write one final line exactly in the form\n"
        "ASSUMPTIONS: <comma-separated requirement keys from the list above whose values your deliverable directly depends on; include a key only if changing its value would require changing your deliverable>"
    )
    system = f"You are the {step.role} in an automated project workflow. Follow the project requirements exactly and be concise."
    return system, user


def parse_output(text: str, fact_keys) -> tuple:
    lines = text.rstrip().splitlines()
    declared = set()
    for i in range(len(lines) - 1, -1, -1):
        m = re.match(r"\s*\**\s*assumptions\s*\**\s*:\s*(.*)$", lines[i], re.I)
        if m:
            for tok in re.split(r"[,\s]+", m.group(1)):
                tok = tok.strip().strip("`*.;")
                if tok in fact_keys:
                    declared.add(tok)
            lines = lines[:i]
            break
    return "\n".join(lines).strip(), declared


def mentions(text: str, term: str) -> bool:
    """Case-insensitive match of a term that is not part of a longer word."""
    if not term:
        return False
    pattern = r"(?<![a-z0-9])" + re.escape(term.lower()) + r"(?![a-z0-9])"
    return re.search(pattern, text.lower()) is not None


def detect(content: str, facts: dict) -> set:
    found = set()
    for key, f in facts.items():
        if any(mentions(content, t) for t in [f.value] + list(f.aliases)):
            found.add(key)
    return found


# --------------------------------------------------------------------------- signatures
def _h(*parts) -> str:
    return hashlib.sha256("\x1e".join(parts).encode("utf-8")).hexdigest()


def definition(step: Step) -> str:
    return "|".join([step.role, step.task, step.out_type, ",".join(step.deps)])


def sig_struct(step: Step, canon: dict) -> str:
    return _h(definition(step), *[canon[d] for d in step.deps])


def sig_assume(step: Step, assumed: set, fact_values: dict, canon: dict) -> str:
    facts = [f"{k}={fact_values.get(k, '')}" for k in sorted(assumed)]
    return _h(definition(step), "\x1d".join(facts), *[canon[d] for d in step.deps])


# --------------------------------------------------------------------------- equivalence
def _strip_fence(t: str) -> str:
    m = re.search(r"```[^\n]*\n(.*?)```", t, re.S)
    return m.group(1) if m else t


def normalize_code(t: str) -> str:
    body = _strip_fence(t)
    try:
        return ast.dump(ast.parse(body))
    except SyntaxError:
        body = re.sub(r"/\*.*?\*/", " ", body, flags=re.S)
        body = re.sub(r"(^|\s)(#|//|--)[^\n]*", " ", body)
        return " ".join(body.split())


def bow_cosine(a: str, b: str) -> float:
    ta = Counter(re.findall(r"[a-z0-9]+", a.lower()))
    tb = Counter(re.findall(r"[a-z0-9]+", b.lower()))
    dot = sum(ta[w] * tb[w] for w in ta)
    na = math.sqrt(sum(v * v for v in ta.values()))
    nb = math.sqrt(sum(v * v for v in tb.values()))
    return dot / (na * nb) if na and nb else 0.0


JUDGE_SYSTEM = "You compare two versions of the output of one step in an automated workflow."


def judge_equivalent(llm: LLM, old: str, new: str, consumer_tasks: list) -> tuple:
    if consumer_tasks:
        who = "Later steps consume this output. Their tasks are:\n" + "\n".join(f"- {t}" for t in consumer_tasks)
    else:
        who = "This output is a final deliverable read by the project owner."
    user = (
        f"{who}\n\nVERSION A:\n{old}\n\nVERSION B:\n{new}\n\n"
        "Would replacing VERSION A with VERSION B change anything those readers rely on, such as facts, values, "
        "names, technologies, interfaces, numbers or decisions? Differences in wording, formatting or ordering alone "
        "do not count. Answer with exactly one word: EQUIVALENT or DIFFERENT."
    )
    rec = llm.chat(JUDGE_SYSTEM, user, temperature=0.0, seed=0, num_predict=5)
    answer = rec["text"].strip().upper()
    same = answer.startswith("EQUIVALENT")
    return same, rec


def equivalent(llm: LLM, out_type: str, old: str, new: str, consumer_tasks: list, theta: float) -> tuple:
    """Returns (is_equivalent, judge_record_or_None)."""
    if old == new:
        return True, None
    if out_type == "json":
        try:
            return json.loads(_strip_fence(old)) == json.loads(_strip_fence(new)), None
        except (ValueError, TypeError):
            pass
    if out_type == "code" and normalize_code(old) == normalize_code(new):
        return True, None
    if bow_cosine(old, new) < theta:
        return False, None
    same, rec = judge_equivalent(llm, old, new, consumer_tasks)
    return same, rec


# --------------------------------------------------------------------------- change translation
TRANSLATE_SYSTEM = "You map a change request onto the requirement keys of a project."


def translate(llm: LLM, wf: Workflow, request: str) -> dict:
    reqs = "\n".join(f"- {k}: {f.value}" for k, f in sorted(wf.facts.items()))
    user = (
        f"Current project requirements:\n{reqs}\n\nChange request: \"{request}\"\n\n"
        'Return only JSON of the form {"key": "<one existing key that changes>", "new_value": "<new value as a short string>"}. '
        'If no existing key matches, return {"key": null, "new_value": null}.'
    )
    rec = llm.chat(TRANSLATE_SYSTEM, user, temperature=0.0, seed=0, num_predict=60)
    try:
        data = json.loads(_strip_fence(rec["text"]).strip())
    except ValueError:
        m = re.search(r"\{.*\}", rec["text"], re.S)
        data = json.loads(m.group(0)) if m else {"key": None, "new_value": None}
    return {"key": data.get("key"), "new_value": data.get("new_value"), "record": rec}


def value_matches(predicted, truth: str) -> bool:
    if predicted is None:
        return False
    norm = lambda s: " ".join(re.findall(r"[a-z0-9.%]+", str(s).lower()))
    p, t = norm(predicted), norm(truth)
    return p == t or t in p or p in t and len(p) >= max(3, len(t) // 2)


# --------------------------------------------------------------------------- execution
def run_step(llm: LLM, wf: Workflow, step: Step, fact_values: dict, canon: dict, tag: str) -> dict:
    system, user = build_prompt(wf, step, fact_values, canon)
    rec = llm.chat(system, user, temperature=0.7, seed=stable_seed(tag, step.id, user), num_predict=700)
    content, declared = parse_output(rec["text"], set(fact_values))
    return {"content": content or rec["text"].strip(), "declared": declared, "record": rec}


def initial_run(llm: LLM, wf: Workflow, tag: str) -> dict:
    fact_values = {k: f.value for k, f in wf.facts.items()}
    state = {}
    canon = {}
    for level in topo_levels(wf):
        for sid in level:
            step = wf.step(sid)
            res = run_step(llm, wf, step, fact_values, canon, tag)
            canon[sid] = res["content"]
            assumed = res["declared"] | detect(res["content"], wf.facts)
            state[sid] = {
                "raw": res["content"], "canon": res["content"], "assumed": assumed, "declared": res["declared"],
                "sig_struct": sig_struct(step, canon), "sig_assume": sig_assume(step, assumed, fact_values, canon),
            }
    return state


def handle_change(llm: LLM, wf: Workflow, init_state: dict, change: Change, system: str, tag: str, theta: float) -> dict:
    fact_values = {k: f.value for k, f in wf.facts.items()}
    facts_now = copy.deepcopy(wf.facts)
    old_terms = [wf.facts[change.key].value] + list(wf.facts[change.key].aliases)
    fact_values[change.key] = change.new_value
    facts_now[change.key].value = change.new_value
    facts_now[change.key].aliases = []
    owner = wf.facts[change.key].owner

    state = copy.deepcopy(init_state)
    canon = {sid: st["canon"] for sid, st in state.items()}
    rerun, cutoffs = set(), set()
    m = {"steps": 0, "tokens": 0, "check_tokens": 0, "calls": 0, "judge_calls": 0, "seconds": 0.0}

    for level in topo_levels(wf):
        for sid in level:
            step = wf.step(sid)
            st = state[sid]
            if system == "B1":
                go = True
            elif system == "B2":
                go = sid == owner or bool(set(step.deps) & rerun)
            elif system in ("B3", "P2"):
                go = sid == owner or sig_struct(step, canon) != st["sig_struct"]
            elif system == "P4":
                go = sid == owner or sig_assume(step, st["assumed"], fact_values, canon) != st["sig_assume"]
            else:  # P1, P3
                go = sig_assume(step, st["assumed"], fact_values, canon) != st["sig_assume"]
            if not go:
                continue

            res = run_step(llm, wf, step, fact_values, canon, tag)
            rec = res["record"]
            rerun.add(sid)
            m["steps"] += 1
            m["calls"] += 1
            m["tokens"] += rec["prompt_tokens"] + rec["completion_tokens"]
            m["seconds"] += rec["seconds"]
            new = res["content"]
            assumed_new = res["declared"] | detect(new, facts_now)

            if system in ("P2", "P3", "P4"):
                same, jrec = equivalent(llm, step.out_type, st["canon"], new,
                                        [c.task for c in consumers(wf, sid)], theta)
                if jrec is not None:
                    m["judge_calls"] += 1
                    m["calls"] += 1
                    m["check_tokens"] += jrec["prompt_tokens"] + jrec["completion_tokens"]
                    m["seconds"] += jrec["seconds"]
            elif system in ("B3", "P1"):
                same = new == st["canon"]
            else:
                same = False

            if same:
                cutoffs.add(sid)
                assumed = st["assumed"] | assumed_new
            else:
                canon[sid] = new
                assumed = assumed_new
            st.update(raw=new, canon=canon[sid], assumed=assumed,
                      sig_struct=sig_struct(step, canon),
                      sig_assume=sig_assume(step, assumed, fact_values, canon))

    truth = must_rerun(wf, change)
    def still_old(text):
        # ignore occurrences of the new value, which may contain the old one
        text = re.sub(re.escape(change.new_value), " ", text, flags=re.I)
        return any(mentions(text, t) for t in old_terms)
    stale = sum(1 for sid in truth if still_old(canon[sid]))
    m.update(
        rerun=sorted(rerun), cutoffs=sorted(cutoffs), must=sorted(truth),
        missed=len(truth - rerun), unnecessary=len(rerun - truth), must_count=len(truth),
        rerun_count=len(rerun), stale=stale,
    )
    return m
