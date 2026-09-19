# Adaptive AI Task Orchestrator

## Run as a local application

The release package starts the private local server, chooses an available
port, opens the dashboard, and stores data under the current operating-system
user account:

```bash
npx adaptive-ai-orchestrator
```

Version 1.0.0 ships the verified Windows runtime. The release workflow builds
Windows, macOS and Linux binaries before publishing subsequent multiplatform releases.

For a permanent command:

```bash
npm install --global adaptive-ai-orchestrator
adaptive-ai-orchestrator
```

Ordinary local installations use SQLite, an in-process worker and the OS
credential store. They do not require PostgreSQL, Redis, Docker, a public
domain or a login. PostgreSQL, Redis and OIDC remain available for shared and
multi-machine installations.

A multi-agent system that turns a plain-English project description into a
dependency graph of specialised AI agents, executes it in parallel, routes
around broken tools — and when something changes, **re-runs only the work that
change actually invalidated**.

```bash
pip install -r requirements.txt
python main.py            # the four headline scenarios, no API key needed
```

With no API key set for anything, everything runs on a deterministic offline
LLM, so the demo, the tests and the benchmarks all work out of the box and
give identical results on every machine. Set `NVIDIA_API_KEY` to select NVIDIA
NIM automatically, or select another provider explicitly with `LLM_PROVIDER`.
See [Which LLM provider](#which-llm-provider).

---

## The idea

Most agent frameworks treat a workflow as a script. Change one requirement and
you re-run the script. That is fine when steps are cheap; it is expensive when
every step is an LLM call.

This system models the workflow as a DAG and decides, per step, whether its
cached result is still valid:

```
a step runs  ⟺  it was explicitly forced
             or it has no usable cached result
             or its input signature no longer matches the cached one
```

A step's **input signature** covers its own definition *plus the exact
outputs of its dependencies*. Two things follow:

- **Change propagates on its own.** Edit the database step and the API step's
  signature changes automatically, because its input changed.
- **Propagation stops when it should.** If a re-run reproduces a byte-identical
  output, every downstream signature still matches, and nothing else re-runs.

That second property is the part dependency analysis alone cannot give you.

---

## The four scenarios

```bash
python main.py
```

| Scenario | What happens | Steps re-run (of 4) |
|---|---|---|
| Full run | Cold start | 4 |
| Requirement change | "use MongoDB instead of PostgreSQL" | 2 — frontend and backend reused |
| Tool failure | GitHub API goes down | 3 — automatically switches to `github_cli` |
| Tool repair | GitHub comes back | 0 until you ask for a re-run |

```python
from orchestrator import Workflow

wf = Workflow.from_description("Build a task manager with auth and a Kanban board")
wf.run_full()

wf.impact_of("database")     # what would a change touch? costs nothing to ask
wf.handle_step_change("database", new_requirement="use MongoDB instead")
wf.handle_tool_failure("github")

wf.print_summary()
wf.export_state("report.html", format="html")
```

---

## Give it a task; it tells you what it needs

Planning does more than produce steps. It decides **which specialists the task
needs** and **what capabilities it depends on**, then checks each one against
your actual machine — before spending a token.

```bash
python -m orchestrator.cli plan "Overhaul our employment offer letter for the UK market"
```

```
Specialists to be created:
  - employment_lawyer  (steps: clause_review)
      termination clauses carry jurisdiction-specific risk
  - compensation_analyst  (steps: benchmark)
      equity and severance need benchmarking, not legal review
  - plain_language_editor  (steps: final_draft)
      the final document is read by candidates, not lawyers

Capabilities required:
  mcp server:
    [MISS] filesystem
           why: read the existing contract templates from disk
           setup: npx -y @modelcontextprotocol/server-filesystem ./contracts
  credential:
    [MISS] COMP_BENCHMARK_API_KEY
           setup: set COMP_BENCHMARK_API_KEY in .env.local
  binary:
    [sim]  pandoc
           status: not on PATH; optional, so the run will proceed without it

BLOCKED - 2 requirement(s) must be satisfied first
```

**The specialists are invented per task.** `employment_lawyer` and
`compensation_analyst` are not in any built-in list — the planner created them,
each with its own system prompt, and they are registered automatically. A
biology task gets a `molecular_biologist`; a trial design gets a
`clinical_trial_statistician`.

**Requirements are checked against reality**, not assumed. Five kinds:

| Kind | Checked by |
|---|---|
| `tool` | is it registered, live, broken, or covered by a fallback? |
| `mcp_server` | is it attached and are its tools registered? |
| `credential` | is that environment variable actually set? |
| `binary` | is it on `PATH`? |
| `package` | is it importable? |

Four outcomes, and the distinction matters:

- **ready** — configured and real
- **simulated** — runs, but the tool output is labelled `[simulated:…]`
- **fallback** — unavailable, but a substitute covers it (a broken `github`
  still runs via `github_cli`, so it *degrades*, it does not block)
- **missing** — genuinely blocks; the report tells you the exact fix

```python
wf = Workflow.from_description("...")
wf.print_requirements()
wf.requirements.can_run        # False if anything is hard-missing
wf.requirements.blockers       # what to fix
wf.requirements.degraded       # what will run, but not for real
```

`orchestrator plan` exits non-zero when blocked, and `orchestrator run` refuses
to start unless you pass `--force`. Hand-written workflows get this too —
requirements are inferred from the tools their steps name:

```python
Workflow(my_steps).print_requirements()
```

---

## It asks you for what it needs, then acts

Like n8n: give it a task, and it stops to ask whenever it hits something only
you can decide — then actually performs the work.

```bash
python -m orchestrator.cli run "Send the weekly customer update"
```

```
  [ok] analyse: done
  [ok] draft:   done
  [??] publish: needs 1 input(s): recipient

----------------------------------------------------------------------
  The workflow needs some information from you.
----------------------------------------------------------------------
  Which list should receive this?
    (the update is sent here; it cannot be recalled)
    recipient [email]: customers@example.com

  About to perform a REAL, irreversible action:
    step : publish
    tool : email
    what : email [REAL]: This week's product update -> customers@example.com

    perform it? [y/N]:
```

**Two gates, for two different reasons.**

*Inputs* are facts the system cannot invent — a recipient address, which repo,
a deadline, a choice between approaches. A step declares them and the run
pauses there:

```python
Step(id="publish", description="Send the update.",
     agent_role="devops", requires_tool="email",
     inputs=[
         InputRequest(name="recipient", prompt="Which list?", type="email",
                      why="it cannot be recalled"),
         InputRequest(name="tone", prompt="What tone?", type="choice",
                      options=["formal", "friendly"], default="friendly"),
     ])
```

Typed and validated — `text`, `multiline`, `email`, `url`, `number`,
`boolean`, `choice`, `secret`. A bad value is refused before anything runs,
and a form with one bad field applies **nothing** rather than half-updating
the run. `secret` values are never echoed, never exported, and never put in a
prompt — the agent is told the value exists, not what it is.

Answers feed **both** the agent's prompt *and* the tool call. A recipient
address routes the actual email; it does not depend on the model echoing it
back into its output.

*Actions* are the irreversible half. Tools declare `side_effect` and
`irreversible`, and before an unundoable one runs, the workflow stops and
shows you what is about to happen:

```python
wf.pending_actions()          # {'publish': PendingAction(tool='email', preview='...')}
wf.approve_action("publish")  # then resume() -> it really sends
wf.reject_action("publish", reason="legal has not signed off")
```

Approving costs **zero LLM calls** — the generated payload was kept, so
sign-off never pays to redo the work.

Three modes, because gating a *simulated* action is pointless friction:

| `action_approval` | Behaviour |
|---|---|
| `"live"` (default) | Gate only actions that would really happen |
| `"all"` | Also gate simulated ones, to rehearse the flow without credentials |
| `"never"` | Unattended; act immediately |

Changing an answer re-runs the step: input values are part of the step's
signature, so the adaptive machinery treats them exactly like a changed
requirement.

Both gates work in the CLI (interactive prompts), the dashboard (a form and
an approval card), and the REST API:

```
GET  /api/runs/{id}/inputs
POST /api/runs/{id}/steps/{step}/inputs      {"values": {...}, "resume": true}
GET  /api/runs/{id}/actions
POST /api/runs/{id}/actions/{step}/approve
POST /api/runs/{id}/actions/{step}/reject
```

Use `--no-interaction` to run headless and have it report what a human would
need to supply, or `-y` to auto-approve.

---

## Connect it to any API

The n8n "credentials + HTTP node" idea. Declare a service once; every agent can
then call it by name.

```jsonc
// connections.json  — safe to commit: it names variables, never values
{
  "connections": {
    "stripe": {
      "base_url": "https://api.stripe.com/v1",
      "auth": { "type": "bearer", "token_env": "STRIPE_API_KEY" }
    },
    "notion": {
      "base_url": "https://api.notion.com/v1",
      "auth": { "type": "api_key", "token_env": "NOTION_TOKEN" },
      "headers": { "Notion-Version": "2022-06-28" }
    }
  }
}
```

```bash
python -m orchestrator.cli connections            # list them and their status
python -m orchestrator.cli run "..." --connections
```

Each becomes a tool (`api_stripe`, `api_notion`) that a step can name. The agent
is told the base URL and how to call it, and emits:

```
TOOL_DIRECTIVE: {"method": "GET", "path": "/customers", "query": {"limit": 3}}
```

**Auth is handled for you**, and the token never enters a prompt:

| `type` | Needs |
|---|---|
| `none` | — |
| `bearer` | `token_env` |
| `api_key` | `token_env`, as a header or `query_param` |
| `basic` | `username_env`, `password_env` |
| `oauth2_client_credentials` | `token_url`, `client_id_env`, `client_secret_env` — token fetched and cached |

**Safety is decided per call, not per service.** A `GET` runs freely; a
`POST`/`PUT`/`PATCH`/`DELETE` is treated as irreversible and goes through the
approval gate showing the exact request:

```
POST https://api.stripe.com/v1/customers [REAL] body={"email":"..."}
perform it? [y/N]:
```

Missing credentials **degrade rather than break**: the call is simulated and
labelled, and the requirements report tells you which variable to set. Each API
gets its own capability, so one service never silently falls back to an
unrelated one.

### The four ways to extend what it can do

| | Gives it | Configure with |
|---|---|---|
| **Tools** | New built-in capabilities in Python | subclass `Tool` |
| **MCP servers** | Whole toolkits others already wrote | `.mcp.json` |
| **API connections** | Any HTTP service, authenticated | `connections.json` |
| **Skills** | Better *procedure*, not new capability | `skills/*.md` |

MCP is the widest lever. Attaching the Playwright MCP server, for example, hands
agents 24 real browser tools (`browser_navigate`, `browser_click`,
`browser_fill_form`, `browser_take_screenshot`) with no code changes:

```bash
python -m orchestrator.cli mcp        # see what each server exposes
```

---

## Architecture

```
                       plain English
                            │
   Layer 1  TaskPlanner ────┴──► validated DAG   (repairs cycles, bad refs, dupes)
                            │
   Layer 2  DependencyGraph │   topological order · execution levels · impact analysis
                            │
   Layer 5  Workflow ───────┼── scheduler: parallel levels, retries, approvals
                     ┌──────┼──────┬──────────────┐
                     │      │      │              │
   Layer 4       AgentManager  MemoryManager   Layer 3 ToolManager
                  role prompts  shared context   capability fallback chains
                     │                              │
   Layer 6      EventBus ──► StateManager (SQLite) ──► exports · WebSocket dashboard
```

| File | Responsibility |
|---|---|
| [orchestrator/graph.py](orchestrator/graph.py) | DAG, topological order, execution levels, `downstream_of` |
| [orchestrator/planner.py](orchestrator/planner.py) | English → specialists + requirements + DAG, with graph repair |
| [orchestrator/requirements.py](orchestrator/requirements.py) | What a task needs, checked against the real machine |
| [orchestrator/inputs.py](orchestrator/inputs.py) | Typed, validated questions the workflow puts to you |
| [orchestrator/connections.py](orchestrator/connections.py) | Named API connections with auth, as tools |
| [orchestrator/skills.py](orchestrator/skills.py) | Reference material matched to steps by role/keyword |
| [orchestrator/llm.py](orchestrator/llm.py) | Gemini / Anthropic / OpenAI / Ollama / stub providers |
| [orchestrator/agents.py](orchestrator/agents.py) | Role catalogue, prompt construction, tool hand-off |
| [orchestrator/memory.py](orchestrator/memory.py) | Shared context between agents, **input signatures** |
| [orchestrator/tools/](orchestrator/tools/) | Tool adapters and capability-based fallback routing |
| [orchestrator/tools/mcp.py](orchestrator/tools/mcp.py) | Real MCP client — stdio/SSE/HTTP servers as agent tools |
| [orchestrator/workflow.py](orchestrator/workflow.py) | The scheduler and all the adaptive logic |
| [orchestrator/state.py](orchestrator/state.py) | SQLite persistence and crash recovery |
| [orchestrator/export.py](orchestrator/export.py) | JSON / CSV / static HTML reports |
| [server/app.py](server/app.py) | REST + WebSocket control plane |
| [benchmarks/](benchmarks/) | Reproducible comparison against baselines |

---

## Parallel execution

The graph is grouped into levels; everything in a level runs concurrently.

```python
wf.graph.execution_levels()
# [['requirements'], ['database', 'frontend'], ['backend'], ['testing']]
```

`database` and `frontend` genuinely run at the same time on a thread pool
(`ORCHESTRATOR_MAX_WORKERS`, default 4). Set `parallel=False` for deterministic
sequential execution.

### Conditional branches

A step can run only when an earlier result or incoming webhook field matches a
structured rule:

```python
Step(id="repair", description="Repair the failed checks", agent_role="backend",
     depends_on=["review"],
     condition={"source": "review", "operator": "contains", "value": "failed"})
```

Webhook branches use sources such as `trigger.priority`. Supported operators
are `contains`, `not_contains`, `equals`, `not_equals`, `exists`, and `truthy`.
The system never evaluates Python or JavaScript supplied in a condition. A
false branch is recorded as **not applicable**, and later steps can still use
that decision as context. Conditions can also be edited in the dashboard.

---

## Tools and fallback

Tools are grouped by **capability**. Any tool can stand in for any other tool
with the same capability, so `ToolManager` routes around failures without the
workflow knowing:

| Capability | Chain |
|---|---|
| `vcs` | `github` → `github_cli` → `artifact_store` |
| `database` | `postgres` → `postgres_cli` → `sqlite_local` |
| `ci` | `ci` → `local_test_runner` |
| `notify` | `slack` → `email` → `console_notify` |
| `http` | `rest_api` |

Each chain **ends in something that always works** — writing to `artifacts/`, a
local SQLite file, or stdout — so a workflow can finish even with every remote
service down.

Adapters are real: with `GITHUB_TOKEN` + `GITHUB_REPO` set, the `github` tool
opens actual issues and commits actual files via the REST API. Without
credentials it degrades to a clearly-labelled `[simulated:…]` result rather
than failing, so nothing is ever mistaken for real work in a report. See
[.env.example](.env.example).

Adding your own tool is one class:

```python
from orchestrator.tools import Tool

class GitLabTool(Tool):
    name = "gitlab"
    capability = "vcs"           # now a fallback candidate for github, automatically
    def is_live(self): return bool(os.environ.get("GITLAB_TOKEN"))
    def _run(self, task, context=None): ...
```

---

## MCP (Model Context Protocol)

Agents can call tools on real MCP servers — the same protocol Claude Desktop
and Claude Code speak — and those tools join the same capability/fallback
routing as everything else.

```bash
cp .mcp.json.example .mcp.json         # same format Claude Desktop/Code uses
python -m orchestrator.cli mcp         # list what each server exposes
python -m orchestrator.cli run "..." --mcp
```

```python
wf = Workflow(steps)
wf.attach_mcp(".mcp.json")             # discovers and registers every tool
# now a step can name one directly:
Step(id="read", description="Read the config", agent_role="research",
     requires_tool="fs_read_file")
```

Three transports are supported: `stdio` (subprocess), `sse`, and
`streamable_http`. Each server runs on its own background thread holding a
persistent session, so agents don't pay subprocess-spawn cost per call, and a
server that dies is reconnected automatically on the next call.

**Agents are given each tool's real input schema.** `prompt_hint()` renders the
MCP tool's JSON schema into the prompt:

```
Your deliverable will be passed to the `add` tool.
Add two integers.
Arguments schema: {"a": <integer>, "b": <integer>}   (? = optional)
End your response with a single line supplying its arguments:
TOOL_DIRECTIVE: {"arguments": {...}}
```

Without this, any tool taking more than one argument is effectively
uncallable — the model knows the tool exists but not what a valid call looks
like. This was a real gap found while testing, not a hypothetical.

Give an MCP server a `capability` in the config and its tools become fallbacks
for that chain (and get the chain's fallbacks in return):

```json
{"mcpServers": {"github": {
  "command": "npx", "args": ["-y", "@modelcontextprotocol/server-github"],
  "capability": "vcs", "tool_prefix": "gh_"
}}}
```

Servers that can't be reached are skipped with a warning rather than being
fatal, so one bad entry never stops the run. The `mcp` SDK is optional — the
orchestrator imports it lazily and works fine without it installed.

---

## Skills — procedures, not capabilities

Tools, MCP servers and API connections give an agent new things it can *do*.
A skill gives an agent already capable of the task a better *way* to do it —
your postmortem format, your API conventions, your house style. Same idea as
a Claude Code `SKILL.md`.

```markdown
<!-- skills/incident-postmortem.md -->
---
name: incident-postmortem
description: Our blameless postmortem format
roles: devops, reviewer
keywords: incident, outage, postmortem
---

# Postmortem format
1. Timeline (UTC timestamps)
2. Impact, quantified
3. Root cause — never "human error"
4. Action items, each with an owner and a due date
```

```bash
python -m orchestrator.cli skills                                    # list them
python -m orchestrator.cli skills --for-role backend --for-description "..."  # preview a match
python -m orchestrator.cli run "..." --skills                        # skills/ just works, no flag needed by default
```

A skill applies itself: drop a file in `skills/` and it activates on every
step whose `roles` it lists, or whose description contains one of its
`keywords` — no per-step wiring. Matched skills are folded into that step's
prompt as reference material the agent is told to follow.

```python
wf = Workflow(steps)
wf.attach_skills("skills")          # explicit, like MCP — never auto-scanned
wf.skills.match(some_step)          # what would apply, without running anything
```

Three shipped examples: `api-design-conventions` (backend), `writing-style`
(writer), `incident-postmortem` (devops/reviewer, keyword-triggered).

---

## Changing a requirement

```bash
orchestrator change <run-id> "use PostgreSQL instead of MongoDB"
```

Prints the requirement that changed, every affected step *with the reason it
was affected*, and what stays reused — then asks before re-running anything.

```
Requirement change
  database.engine: 'MongoDB' -> 'PostgreSQL'

Affected steps (3):
  schema                  database.engine: owner (P4)
  backend                 database.engine: assumed fact
  tests                   upstream may change; signature checked at execution

Reused unchanged (2): frontend, requirements

Apply this change and re-run the affected steps? [y/N]
```

`backend` there has **no dependency edge** to `schema`. It re-runs because it
declared it relied on `database.engine` — which is the whole point: plain
dependency tracking would have called it valid and left stale MongoDB code
behind. Run `python examples/12_requirement_change.py` to watch it happen.

`--yes` skips the prompt, `--no-rerun` applies the change without executing.
Over HTTP the same split is `POST /change/prepare` (reads only) and
`POST /change/apply` (refuses anything not explicitly confirmed).

---

## Requirement changes (core Python API)

The planner can extract requirement facts with a key, value, aliases and owner
step. Workers declare assumptions; whole-word matches of fact values or aliases
also record hidden dependencies. Signatures include those facts' current values.
The owner always depends on its own fact (P4), even when it omits its declaration.

```python
from orchestrator import Step, Workflow
from orchestrator.graph import DependencyGraph
from orchestrator.change_aware import Fact

graph = DependencyGraph([
    Step("storage", "Define the database schema", "database"),
    Step("client", "Write a client for the database fact", "backend",
         assumes=["database"]),  # no edge to storage: a hidden dependency
    Step("tests", "Test the client", "testing", depends_on=["client"]),
], facts=[Fact("database", "MongoDB", "storage", ["Mongo"])])
wf = Workflow(graph=graph)
wf.run_full()

proposal = wf.prepare_change("use PostgreSQL instead of MongoDB")
print(proposal)  # old/new values, affected steps and reasons; no tools run
if input("Apply this exact change? [y/N] ").strip().lower() == "y":
    report = wf.apply_change(proposal, confirmed=True)
```

`preview_fact_change({"database": "PostgreSQL"})` provides the same preview
without a translator call. When no key matches, `prepare_change` returns a new
plan plus a diff for review; confirming a re-plan conservatively clears cached
results. Stale or modified previews are rejected. Old pending action approvals
are revoked for affected steps, so a revised payload needs its own approval.

Facts, declared/detected assumptions, actual generated output (`raw_output`),
canonical output (`output`), invalidation reasons and `reused` / `rerun` /
`cut_off` status persist with the workflow. The requirements report checks
output types (`code`, `json`, `text`), accepted dependency types, fact owners,
assumption keys and a **planning token estimate**, not measured API usage.

`ORCHESTRATOR_CHANGE_AWARE=false` keeps structural signature behaviour.
`ORCHESTRATOR_SEMANTIC_CUTOFF=false` is the default: only identical outputs
stop propagation. Opting in adds Python AST equivalence, canonical JSON
comparison and an LLM text judge above the bag-of-words cosine threshold
(`ORCHESTRATOR_EQUIVALENCE_THRESHOLD=0.95`). Unknown code languages use
conservative literal comparison. Tool-backed steps never receive semantic
cutoff. The judge's reported tokens are included in step usage.

Semantic cutoff has **not been measured on NVIDIA**. Existing research results
remain unchanged. CLI/API/dashboard change controls are the next phase;
these Python methods are the current entry points. Offline translation supports
simple `change X to Y` and `use Y instead of X` requests; other requests return
a reviewable generic re-plan rather than claiming full language understanding.

## Which LLM provider

NVIDIA NIM is selected when `NVIDIA_API_KEY` is present. Without it, automatic
selection uses the deterministic **stub**. An explicit `LLM_PROVIDER` overrides
this choice; Gemini, Anthropic, OpenAI and Ollama remain supported.

```dotenv
LLM_PROVIDER=nvidia
NVIDIA_API_KEY=your-key-in-your-local-env-file
NVIDIA_MODEL=nvidia/nemotron-3-super-120b-a12b
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
```

Never commit your key. Optional `NVIDIA_MODEL_AGENTS`, `NVIDIA_MODEL_PLANNER`,
`NVIDIA_MODEL_CHANGE_TRANSLATOR` and `NVIDIA_MODEL_EQUIVALENCE_JUDGE` override
the model for those roles; `NVIDIA_MODEL_<SPECIALIST_ROLE>` overrides an
individual specialist. Empty overrides inherit the base model. Offline or
explicitly injected non-NVIDIA providers remain offline/non-NVIDIA.

```bash
export LLM_PROVIDER=ollama              # force a specific one
python -m orchestrator.cli --provider anthropic run "..."   # or per-invocation
```

```python
from orchestrator import Workflow, build_provider

wf = Workflow(steps, llm=build_provider("openai"))
```

| Provider | Needs | Notes |
|---|---|---|
| `nvidia` | `NVIDIA_API_KEY` | Automatic primary; OpenAI-compatible NVIDIA NIM endpoint |
| `gemini` | `GEMINI_API_KEY` | Select explicitly |
| `anthropic` | `ANTHROPIC_API_KEY` | Claude Messages API |
| `openai` | `OPENAI_API_KEY` | Also fits Azure OpenAI / OpenRouter / vLLM via `OPENAI_BASE_URL` |
| `ollama` | nothing — local server | Free, offline, no rate limit. `ollama pull llama3.1`, then `LLM_PROVIDER=ollama` |
| `stub` | nothing | Deterministic, offline, what every test and benchmark in this repo runs on |

**Ollama is never auto-selected.** It needs no key, so key-presence can't
detect it, and silently assuming a local server is running (and blocking on
a connection attempt) would be the wrong kind of surprise — it has to be
requested explicitly via `LLM_PROVIDER=ollama`.

All four real providers share the same retry/backoff policy and token/cost
accounting as the original Gemini path; none of them require their vendor's
SDK — auth, request and response parsing are implemented directly against
each API's plain HTTP contract in [orchestrator/llm.py](orchestrator/llm.py).

---

## Reliability

- **Retries** — transient failures (429/503/timeouts) retry with exponential
  backoff; deterministic ones fail fast. Configure per step via `max_retries`.
- **Failure isolation** — a failed step cancels only its downstream cone.
  Independent branches still complete.
- **Crash recovery** — every step result is written to SQLite as it lands.

  ```bash
  python -m orchestrator.cli runs            # list persisted runs
  python -m orchestrator.cli resume <run_id> # pick up exactly where it stopped
  ```

- **Human-in-the-loop** — `requires_approval=True` (or `wf.pause_before(id)`)
  stops the run before that step until `wf.approve(id)` and `wf.resume()`.

---

## Live dashboard

```bash
python -m orchestrator.cli serve       # http://127.0.0.1:8000
```

For a production-style two-process deployment:

```bash
docker compose up --build
```

The API runs in queue-only mode and a non-root worker claims durable jobs from
the shared SQLite volume. Both services share the generated-workspace volume;
restart policies and an HTTP health check are included.

Plan a workflow from the browser, watch the DAG light up step by step over a
WebSocket, click any node to change its requirement, break a tool to force the
fallback, approve a paused step, and export JSON/CSV/HTML. Coding workflows also
show every file the agents created, with a safe download button for each file.
The **Run automatically** panel can repeat a workflow at a chosen interval,
pause or resume that schedule, and delete it. Schedules live in SQLite, survive
server restarts, start with fresh data each time, and do not overlap an active
run. Fully self-contained — no CDN, no build step.

The **Start from another app** panel creates an inbound webhook for a saved
workflow. POST a JSON object to the generated URL and the event data becomes
context for every agent and tool in a fresh run. The URL secret is displayed
once; SQLite stores only its SHA-256 digest. Webhooks can be paused or deleted,
and an active run rejects overlapping triggers.

The **Reusable workflows** panel saves the current graph as a named template.
Launching a template creates an independent workflow with empty results, while
saving the same name updates the template. Agent outputs may also contain a
structured `HANDOFF` line addressed to a later step or specialist; handoffs are
persisted in the event log, restored after restart, and shown in the dashboard.

REST surface: `POST /api/runs`, `POST /api/runs/{id}/run`,
`GET /api/runs/{id}/impact/{step}`, `POST /api/runs/{id}/steps/{step}/change`,
`POST /api/runs/{id}/tools/{tool}/break`, `WS /ws/{id}`, and
`GET /api/runs/{id}/export?format=…`. Workspace outputs are available through
`GET /api/runs/{id}/files` and `GET /api/runs/{id}/files/{path}`. Full schema
at `/docs`. Recurring runs are managed through
`GET|POST /api/runs/{id}/schedules` and
`PATCH|DELETE /api/runs/{id}/schedules/{schedule_id}`.
Webhook configuration uses `GET|POST /api/runs/{id}/webhooks` and
`PATCH|DELETE /api/runs/{id}/webhooks/{webhook_id}`; external systems trigger
the workflow with `POST /hooks/{secret}`.

Runs paused before an irreversible action are crash-safe. SQLite stores the
prepared payload locally and restores it after a restart, so approval resumes
the exact pending action without another model call.

---

## Benchmarks

```bash
python -m orchestrator.cli bench       # writes benchmarks/results/
```

Six systems run **identical agent work** through the same deterministic stub
LLM, so the differences come from orchestration strategy alone. Step, call and
token counts are exact and reproduce on any machine; wall time is the median of
3 repeats on this machine (0.05 s simulated latency per LLM call).

| System | Steps run | Tokens | Wall (s) |
|---|---:|---:|---:|
| **adaptive (this project)** | **54** | **18,990** | **2.80** |
| adaptive, downstream-cone invalidation only | 56 | 19,795 | 2.92 |
| restart-all baseline | 71 | 24,240 | 4.49 |
| langgraph | 71 | 24,240 | 3.48 |

Across seven scenarios: **24% fewer step executions and 22% fewer tokens** than
either baseline. Per-scenario, the gap tracks where the change lands:

| Scenario | Ours | Baselines | Saving |
|---|---:|---:|---|
| `full_run` | 5 | 5 | none — nothing to reuse |
| `early_change` (root) | 10 | 10 | none — everything really is invalid |
| `mid_change` (database) | 8 | 10 | 20% |
| `leaf_change` | 6 | 10 | 40% |
| `noop_rerun` | 6 | 10 | 40% |
| `tool_failure` | 7 | 10 | 30% |
| `microservices_change` (8 steps) | 12 | 16 | 25% |

**Reading these honestly.** The saving is zero when a change genuinely
invalidates the whole graph, and largest when it lands late or turns out to be
a no-op. The overall figure depends entirely on the mix of scenarios, so treat
24% as "what this particular mix produced", not a universal number. The
`noop_rerun` row is the only one that isolates signature comparison itself —
everywhere else, signature comparison and plain dependency analysis agree.

**About the LangGraph baseline.** It is real LangGraph, built with the true DAG
edges (so it parallelises exactly as we do — which is why its wall time beats
restart-all) and a SQLite checkpointer. Its fan-in nodes are deferred and
guarded so no node executes twice; without that guard its step counts would
have looked artificially worse. On a requirement change it is re-invoked and
every node re-runs: LangGraph checkpoints let you *resume an interrupted run*,
but the framework has no notion of "this node's inputs are unchanged, skip it".
You can hand-roll that gating on top of it — the point of the comparison is
that you have to.

The matrix also contains real CrewAI 1.x `Agent`/`Task`/`Crew` execution and
real Microsoft AutoGen AgentChat `BaseChatAgent` execution. Both use the same
deterministic provider as the other systems so framework behavior is measured
without network or model variance. Re-run the benchmark to regenerate current
six-system figures; the older four-system table above remains a historical run.

---

## Testing

```bash
python -m pytest tests/ -q
python -m pytest tests/ --cov=orchestrator --cov=server --cov=benchmarks
```

**703 tests collected, approximately 90% coverage.** Unit tests per layer, integration tests for the
four headline scenarios, a full plan→run→change→crash→resume→export lifecycle
test, API and WebSocket tests via `TestClient`, and mocked-transport tests for
the live GitHub/Slack/SendGrid/CI adapters. The MCP tests run against a **real
MCP server** spawned as a subprocess ([tests/fixtures/fake_mcp_server.py](tests/fixtures/fake_mcp_server.py)),
so the client is verified against the actual protocol rather than a mock. The
uncovered remainder is credential-gated code that needs real accounts.

---

## Examples

| | |
|---|---|
| [01_quickstart.py](examples/01_quickstart.py) | Describe → run → report |
| [02_adaptive_reexecution.py](examples/02_adaptive_reexecution.py) | Mid-graph, leaf and no-op changes side by side |
| [03_tools_and_fallback.py](examples/03_tools_and_fallback.py) | Custom tools, breaking a whole chain |
| [04_crash_recovery_and_approval.py](examples/04_crash_recovery_and_approval.py) | Restart mid-run, approval gate |
| [05_custom_roles_and_events.py](examples/05_custom_roles_and_events.py) | New agent roles, subscribing to events |
| [06_mcp_tools.py](examples/06_mcp_tools.py) | Real MCP server tools, schema hints, fallback |
| [07_task_to_finished_work.py](examples/07_task_to_finished_work.py) | Task → invented specialists → requirements → run |
| [08_ask_and_act.py](examples/08_ask_and_act.py) | Asks you for inputs, then gates the irreversible action |
| [09_api_connections.py](examples/09_api_connections.py) | Authenticated calls to any HTTP API (real httpbin requests) |
| [10_skills.py](examples/10_skills.py) | Reference material matched into an agent's prompt |
| [11_hermes_desktop.py](examples/11_hermes_desktop.py) | High-level Word task through the optional Hermes desktop worker |
| [12_requirement_change.py](examples/12_requirement_change.py) | MongoDB → PostgreSQL; catches a step with no edge to the change |
| [13_computer_use.py](examples/13_computer_use.py) | Browser/desktop control and every guard, in simulation |
| [14_skill_discovery.py](examples/14_skill_discovery.py) | Search, review, approve — installs nothing without confirmation |

---

## CLI

```
python -m orchestrator.cli plan   "Build a bookstore API"      # specialists + requirements, no execution
python -m orchestrator.cli plan   "..." --mcp --json-out plan.json
python -m orchestrator.cli run    "Build a bookstore API" --html report.html
python -m orchestrator.cli run    "..." --change database --requirement "use MongoDB"
python -m orchestrator.cli run    "..." --break-tool github
python -m orchestrator.cli run    "..." --no-interaction   # never prompt
python -m orchestrator.cli run    "..." -y                 # auto-approve actions
python -m orchestrator.cli runs                                # list persisted runs
python -m orchestrator.cli resume <run_id>
python -m orchestrator.cli mcp                                 # list MCP server tools
python -m orchestrator.cli connections                         # list API connections
python -m orchestrator.cli skills                              # list skills
python -m orchestrator.cli --provider ollama run "..."          # pick a specific LLM
python -m orchestrator.cli run "..." --mcp                     # give agents those tools
python -m orchestrator.cli demo
python -m orchestrator.cli serve --port 8000
python -m orchestrator.cli bench  --repeats 5
```

`--stub` on any command forces the offline LLM.

## Computer use (browser and desktop)

A step that must operate a real interface asks for the `computer_use`
capability rather than naming a backend. The tool picks one:

    browser (Playwright MCP)  →  desktop (Hermes)  →  nothing

Browser first: anything inside a website is cheaper to drive, easier to
observe, and can be restricted to a list of domains in a way that "control
the computer" cannot.

```bash
ORCHESTRATOR_ALLOW_DESKTOP=1
ORCHESTRATOR_COMPUTER_USE_ALLOW=example.com,notepad
```

Try it with no setup at all — `python examples/13_computer_use.py` runs in
simulation and prints each guard refusing something.

### The rules it enforces

| Rule | What happens |
|---|---|
| Off by default | Nothing runs until `ORCHESTRATOR_ALLOW_DESKTOP` is set |
| Approval first | The tool is irreversible, so the plan is shown and approved before anything moves |
| Name your targets | A step that does not say which sites or apps it touches is refused |
| Allow-list | Targets outside it are refused; **unset permits nothing**, so forgetting fails closed |
| No secrets | Passwords, card numbers, API keys and one-time codes are refused before the backend sees them — ask the person, or read from the vault at use time |
| Bounded | Action and time limits stop a run that goes wrong |
| Logged | Every action appends to `workspace/<run-id>/computer_use.log.jsonl` |

Subdomains are covered by a parent entry (`example.com` allows
`www.example.com`) but lookalikes are not (`notexample.com` is refused).

---

### Hermes desktop worker

Hermes is an optional worker for outcomes that require ordinary desktop
applications. Our orchestrator still owns the plan, input questions, action
approval, persistence and adaptive re-execution. Hermes receives one approved,
bounded desktop objective and returns its result to the workflow.

1. Install Hermes Agent from its official Windows installer.
2. Run `hermes setup`.
3. Run `hermes computer-use install` and `hermes computer-use doctor`.
4. Set `ORCHESTRATOR_ALLOW_DESKTOP=true` in `.env.local`.
5. Restart the dashboard and choose `hermes_desktop` for a desktop step.

The adapter never enables Hermes' `--yolo` option, never requests administrator
elevation, runs in `workspace/<run-id>/`, has a ten-minute timeout, and goes
through the same irreversible-action approval gate as email and database writes.
If Hermes is absent or disabled, the result is explicitly labelled simulated.

---

## Configuration

Everything is optional; copy [.env.example](.env.example) to `.env.local`.

| Variable | Default | Effect |
|---|---|---|
| `LLM_PROVIDER` | auto | Force `nvidia`\|`gemini`\|`anthropic`\|`openai`\|`ollama`\|`stub` |
| `NVIDIA_API_KEY` | — | Auto-select NVIDIA when present; otherwise offline stub |
| `GEMINI_API_KEY` / `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | — | Used only when the corresponding provider is selected |
| `ORCHESTRATOR_CHANGE_AWARE` | `true` | Fact assumptions and owner-forced invalidation |
| `ORCHESTRATOR_SEMANTIC_CUTOFF` | `false` | Enable unmeasured semantic cutoff for reasoning steps |
| `ORCHESTRATOR_EQUIVALENCE_THRESHOLD` | `0.95` | Minimum cosine similarity before a text judge call |
| `ORCHESTRATOR_PLAN_TOKEN_BUDGET` | `20000` | Limit the estimated plan tokens before execution |
| `OLLAMA_MODEL` / `OLLAMA_BASE_URL` | `llama3.1` / `localhost:11434` | Only used when `LLM_PROVIDER=ollama` |
| `ORCHESTRATOR_MAX_WORKERS` | `4` | Parallel step concurrency |
| `ORCHESTRATOR_CONTEXT_BUDGET` | `1200` | Chars of each dependency's output passed downstream |
| `ORCHESTRATOR_DB` | `orchestrator_state.db` | Local SQLite persistence file |
| `ORCHESTRATOR_STATE_URL` | — | Shared PostgreSQL SQLAlchemy URL for multi-machine deployments |
| `ORCHESTRATOR_BROKER_URL` | — | Redis URL enabling Celery distributed workers |
| `ORCHESTRATOR_API_KEY` | — | Optional bearer key protecting every control-plane API and WebSocket |
| `ORCHESTRATOR_API_KEYS` | — | JSON key map; values may include role, organization and user |
| `ORCHESTRATOR_RATE_LIMIT_PER_MINUTE` | `0` | Per-key request limit; `0` disables limiting |
| `ORCHESTRATOR_QUEUE_ONLY` | — | `1` makes the API enqueue work for a separate `orchestrator worker` process |
| `ORCHESTRATOR_STUB_LLM` | — | `1` forces the stub even with a key |
| `ORCHESTRATOR_TEST_CMD` | — | Opt-in before `local_test_runner` executes anything |
| `ORCHESTRATOR_ALLOW_DESKTOP` | — | `true` enables approved Hermes desktop steps |
| `HERMES_EXECUTABLE` | `hermes` | Override the Hermes executable name or path |
| `OIDC_ISSUER` / `OIDC_AUDIENCE` | — | Verify enterprise SSO tokens and enable OIDC login |
| `ORCHESTRATOR_VAULT_KEY` | — | Fernet key for tenant-scoped encrypted local credentials |
| `ORCHESTRATOR_SECRET_BACKEND` | `local` | Set `aws` for AWS Secrets Manager |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | — | Export API traces to an OpenTelemetry collector |

---

## Production deployment

`docker compose up --build` starts PostgreSQL, Redis, two worker classes, the
API, Prometheus and an OpenTelemetry collector. Alembic migrations run before
the API starts. Browser work is routed to a non-root Playwright container;
ordinary work uses the standard queue. Desktop work uses a dedicated Windows
queue started with `scripts/run_desktop_worker.ps1`, which rejects elevated
Administrator sessions.

For multiple machines, use [deploy/kubernetes/orchestrator.yaml](deploy/kubernetes/orchestrator.yaml)
with managed PostgreSQL and Redis. Supply runtime secrets through the platform's
secret operator. The API verifies OIDC signatures, issuer, audience and expiry,
then scopes runs, templates, schedules, webhooks, audit history and encrypted
credentials to the authenticated organization.

Dedicated connectors include GitHub, PostgreSQL, CI, Slack, SendGrid, Gmail,
Stripe, Amazon S3 and generic authenticated REST APIs. Real irreversible calls
still pause for approval.

Operational limits remain explicit: external providers, managed databases and
identity accounts must be supplied by the deployer; desktop automation requires
a Windows worker with Hermes installed; container execution could not be run on
a machine without Docker. These are deployment dependencies rather than mocked
successes.

---

## License

MIT
