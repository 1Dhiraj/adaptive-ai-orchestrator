"""Layer 1 -- plain English in, an executable dependency graph out.

The LLM is asked for JSON, but it is never trusted: whatever comes back is
parsed defensively and then *repaired* (duplicate ids renamed, dangling
dependencies pruned, cycles broken) so the orchestrator always receives a
valid DAG.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .agents import resolve_role
from .graph import DependencyGraph
from .inputs import InputRequest, parse_declared_inputs
from .llm import LLMProvider, get_provider
from .models import LLMUsage, Step
from .requirements import (
    AgentSpec,
    Requirement,
    RequirementsReport,
    infer_requirements,
    merge_requirements,
)
from .tools import canonical_tool_name

PLANNER_SYSTEM_PROMPT = """You are a project planning agent. Given any task,
you decide (a) what specialists are needed, (b) what capabilities the work
depends on, and (c) how to decompose it into a directed acyclic graph.

Respond with ONLY valid JSON, no markdown fences and no commentary:

{
  "agents": [
    {
      "role": "snake_case_specialist_name",
      "system_prompt": "You are a ... . You deliver ... .",
      "why": "one line on why this task needs this specialist",
      "handles": ["task_id", ...]
    }
  ],
  "requirements": [
    {
      "name": "github | DATABASE_URL | filesystem | psycopg | gh",
      "kind": "tool | credential | mcp_server | package | binary",
      "why": "what it is needed for",
      "needed_by": ["task_id", ...],
      "setup": "the exact command or variable that satisfies this",
      "optional": false
    }
  ],
  "tasks": [
    {
      "id": "short_snake_case_id",
      "name": "Human readable name",
      "role": "must match one of the agents above",
      "description": "One or two sentences stating exactly what to produce.",
      "output_type": "code | json | text",
      "accepts": ["code", "json", "text"],
      "assumes": ["requirement_fact_key"],
      "depends_on": ["id_of_prerequisite", ...],
      "tool": "EXACT installed tool name, a missing capability name, or null for work that needs no tool",
      "condition": {"source":"earlier_task_id or trigger.field", "operator":"contains | not_contains | equals | not_equals | exists | truthy", "value":"comparison value"},
      "inputs": [
        {
          "name": "snake_case_name",
          "prompt": "the question to put to the operator",
          "type": "text | multiline | email | url | number | boolean | choice | secret",
          "required": true,
          "default": null,
          "options": ["only for type choice"],
          "why": "what the step does with it"
        }
      ]
    }
  ],
  "facts": [{"key": "database", "value": "MongoDB", "aliases": ["Mongo"], "owner": "task_id"}],
  "notes": ["assumptions or caveats worth surfacing to the operator"]
}

AGENTS - invent the specialists this specific task needs. Do not limit
yourself to generic software roles: a legal review task wants a
"contract_analyst", a biology task wants a "molecular_biologist". Write each
system_prompt so it states the expertise AND the concrete deliverable format.
Reuse one agent across several tasks when the expertise is the same.

FACTS - extract concrete, changeable requirements actually stated by the user.
Do not invent values. Each fact has a unique key, current value, aliases for
that value, and one owner task. Never include secrets or credentials. Declare
assumes on every task directly depending on a fact, including hidden dependencies.
Refer to facts by key in task descriptions instead of hard-coding their values.
Use output_type and accepts to describe deliverables and compatible inputs.

REQUIREMENTS - declare everything the work genuinely depends on:
  - "tool": one of the tools in the supplied catalogue, including workspace
    for creating project files, terminal for restricted build/test commands,
    hermes_desktop for desktop applications, or a connected MCP/API tool.
  - "credential": the exact environment variable name, e.g. GITHUB_TOKEN.
  - "mcp_server": an MCP server that supplies a needed capability. Use the
    real package name in "setup", e.g.
    "npx -y @modelcontextprotocol/server-filesystem ."
  - "binary": an executable that must be on PATH, e.g. "gh", "psql".
  - "package": an importable Python package.
Mark a requirement optional:true when the work can still be completed without
it, just less completely. Declare nothing the task does not actually need --
a pure writing or analysis task may need no capabilities at all.

INPUTS - facts only the operator can supply, that you cannot produce yourself.

Declare an input ONLY for a fact you could not possibly know:
  - a recipient address, an account name, which repository to push to
  - a deadline, a budget, a headcount, a real-world constraint
  - a choice between genuinely different approaches

NEVER declare an input for anything the agent is supposed to WRITE. The whole
point of the step is to produce that. These are all wrong:
  - "What is the body of the email?"        <- you write the body
  - "What is the subject line?"             <- you write the subject
  - "What should the report say?"           <- you write the report
  - "Which template should we use?"         <- you choose it
  - "What content should the page have?"    <- you create it
If a step's job is to draft something, asking the operator to draft it makes
the step pointless.

Also never declare an input for something an earlier step's output provides.
Use type "secret" for credentials. At most 2 inputs per step, and most steps
should have none at all.

TASKS:
- Between 1 and 8 tasks. Use one task for one direct real-world action; use
  several only when distinct outputs or true dependencies exist.
- Every id in depends_on MUST be the id of another task in this list.
- The graph must be acyclic.
- Put tasks that could run at the same time at the same dependency depth --
  do NOT chain everything into a single line unless it truly is sequential.
- "tool" must name something declared in requirements. If the capability is
  missing, keep its name and declare a non-optional tool requirement. The
  system can discover an integration; never replace actual execution with prose.
- Prefer an installed API, then a browser tool, then desktop automation. A
  browser or desktop action may need account sign-in and explicit permission.
- Sending email is real work: preserve the supplied recipient, draft the
  content yourself, and require approval for the exact message before sending.
  Make it ONE adaptive_email task. That tool owns opening Gmail, sign-in,
  drafting, review and sending; never add a separate computer_use/login step.
- A simple browser or desktop instruction is ONE computer_use task. Put the
  target sites/apps and concrete actions in its description; do not add
  separate planning or verification prose steps unless they produce a real
  deliverable.
- A requested file is a real deliverable. If the person asks for a PDF, the
  final task must use artifact_store with an artifacts/*.pdf path and depend
  on every step whose output belongs in the document. Opening/searching a
  browser is not completion until that PDF exists.
- Descriptions state the deliverable, not the process.
- Use condition only for a real branch. Its source must be a dependency's id
  or trigger.field for webhook data. Omit condition for normal tasks.
- For software projects, steps that create or edit code should use workspace;
  build/test commands should be separate terminal steps. Use github only when
  the request actually asks to publish or modify a remote repository.
"""


SOFTWARE_TASK_PATTERN = re.compile(
    r"\b(?:web|mobile|desktop)\s+app(?:lication)?\b|\bwebsite\b|\bsoftware\b|"
    r"\b(?:frontend|backend|full[ -]?stack|api|microservice|codebase)\b|"
    r"\b(?:build|develop|implement|create|write)\b.{0,50}\b"
    r"(?:app|application|system|platform|portal|dashboard|cli|command[ -]?line|"
    r"script|program|package|library)\b|"
    r"\b(?:python|javascript|typescript|java|rust|golang|c\+\+)\b.{0,45}\b"
    r"(?:cli|command[ -]?line|script|program|tool|app|application)\b",
    re.IGNORECASE,
)


def is_software_task(description: str) -> bool:
    """Conservative check used to attach opt-in coding tools before planning."""
    return bool(SOFTWARE_TASK_PATTERN.search(description or ""))


CLARIFIER_SYSTEM_PROMPT = """You are about to plan a task. First decide whether
you need to ask the person anything.

Respond with ONLY valid JSON, no markdown fences:

{
  "questions": [
    {
      "name": "snake_case_name",
      "prompt": "the question, phrased for a non-expert",
      "type": "text | multiline | email | url | number | boolean | choice",
      "options": ["only for type choice"],
      "default": "a sensible default, or null if there isn't one",
      "required": true,
      "why": "one line: what changes in the plan depending on the answer"
    }
  ]
}

Ask ONLY what you genuinely cannot assume and what would materially change
the plan -- scope, audience, platform, budget, deadline, which account or
repository to use, a choice between real alternatives.

Do NOT ask:
- anything you can sensibly decide yourself as the expert
- anything a later step will discover on its own
- for API keys, passwords or credentials (those are handled separately)
- vague questions like "any other requirements?"

Prefer 0-4 questions. Returning an empty list is a good answer when the task
is already clear -- an unnecessary question is worse than none. Offer
"choice" with concrete options rather than open text wherever you can, and
give a sensible "default" so the person can just accept it.
"""


class PlanningError(RuntimeError):
    """The planner could not produce a usable graph."""


@dataclass
class PlanResult:
    graph: DependencyGraph
    usage: LLMUsage = field(default_factory=LLMUsage)
    repairs: List[str] = field(default_factory=list)
    raw_response: str = ""
    used_fallback_plan: bool = False
    #: Specialists the planner decided this task needs, and the capabilities
    #: it depends on -- already checked against the real environment.
    requirements: RequirementsReport = field(default_factory=RequirementsReport)

    @property
    def agents(self) -> List[AgentSpec]:
        return self.requirements.agents

    @property
    def can_run(self) -> bool:
        return self.requirements.can_run

    def print_requirements(self, width: int = 78) -> None:
        self.requirements.print_report(width)


def extract_json(text: str) -> Optional[dict]:
    """Best-effort JSON extraction from an LLM response.

    Handles: clean JSON, ```json fenced blocks, and prose with an object
    embedded somewhere in it.
    """
    if not text:
        return None
    candidate = text.strip()

    fenced = re.search(r"```(?:json)?\s*(.*?)```", candidate, re.DOTALL)
    if fenced:
        candidate = fenced.group(1).strip()

    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    # Fall back to brace matching so trailing prose does not defeat us.
    start = candidate.find("{")
    while start != -1:
        depth, in_string, escaped = 0, False, False
        for i in range(start, len(candidate)):
            ch = candidate[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(candidate[start:i + 1])
                        if isinstance(parsed, dict):
                            return parsed
                    except json.JSONDecodeError:
                        break
        start = candidate.find("{", start + 1)
    return None


#: Words that mark an "input" as actually asking the operator to do the
#: agent's job -- write the body, pick the wording, supply the content.
_DELIVERABLE_WORDS = re.compile(
    r"\b(body|content|copy|text|wording|message|subject|title|headline|"
    r"template|draft|summary|description|outline|script|caption|"
    r"code|snippet|design|layout|schema|query|answer|response)\b",
    re.IGNORECASE)

#: Phrasings that reliably indicate the model is asking for generated output
#: rather than a fact. "What is the name of X" is fine; "what should X say"
#: is the agent's job.
_DELIVERABLE_PHRASES = re.compile(
    r"(what should .* (say|contain|include|look like)|"
    r"write the|provide the (text|content|body|copy)|"
    r"what (is|are) the (body|content|text|subject|wording))",
    re.IGNORECASE)

# Tool readiness is checked from the real registry and converted into one
# setup card when genuinely missing. A planner-generated input asking the
# same question again is unreliable and redundant.
_SETUP_INPUT_PHRASES = re.compile(
    r"(?:have you (?:set up|configured|installed)|"
    r"is .{0,60}(?:set up|configured|installed|available|ready)|"
    r"(?:tool|package|binary|connection).{0,30}(?:available|ready))",
    re.IGNORECASE,
)

#: Too many questions is its own failure: the operator ends up doing the work.
MAX_INPUTS_PER_STEP = 2
MAX_INPUTS_PER_WORKFLOW = 5
MAX_PLAN_ATTEMPTS = 3


def _is_deliverable_request(request: Any) -> bool:
    """True if this 'input' is really asking the operator to do the agent's job."""
    haystack = f"{request.name} {request.prompt}"
    if _DELIVERABLE_PHRASES.search(haystack):
        return True
    # A name like "email_body" or "report_content" is the clearest signal.
    return bool(_DELIVERABLE_WORDS.search(request.name.replace("_", " ")))


def _is_setup_request(request: Any) -> bool:
    haystack = f"{request.name.replace('_', ' ')} {request.prompt}"
    return bool(_SETUP_INPUT_PHRASES.search(haystack))


def filter_inputs(requests: List[InputRequest],
                  limit: int = MAX_INPUTS_PER_STEP) -> List[InputRequest]:
    """Drop questions the agent should be answering itself, then cap the rest.

    Models reliably over-declare inputs -- left alone they will ask the
    operator to supply the email body, which makes the step pointless and the
    workflow feel like an interrogation. Prompt wording alone does not stop
    it, so this is enforced in code.
    """
    kept = [r for r in requests
            if not _is_deliverable_request(r) and not _is_setup_request(r)]
    return kept[:limit]


def _sanitise_id(raw: Any, index: int) -> str:
    text = re.sub(r"[^a-z0-9_]+", "_", str(raw or "").strip().lower()).strip("_")
    return text or f"step_{index + 1}"


def _ensure_requested_artifacts(graph: DependencyGraph, description: str,
                                repairs: List[str]) -> None:
    """Make an explicitly requested PDF an enforceable final graph node."""
    if not re.search(r"(?i)(?:\bpdf\b|\.pdf\b)", description or ""):
        return

    filename_match = re.search(r"(?i)\b([a-z0-9][a-z0-9_.-]{0,80}\.pdf)\b", description)
    filename = filename_match.group(1) if filename_match else "report.pdf"
    filename = re.sub(r"[^A-Za-z0-9_.-]+", "-", filename).strip(".-") or "report.pdf"
    path = f"artifacts/{filename}"

    existing = [step for step in graph if step.requires_tool == "artifact_store"]
    instruction = (
        f"Create the complete requested PDF at {path} from the prerequisite outputs. "
        "Include the substantive results and any source URLs; do not replace them with "
        "a progress note. The step is complete only when artifact_store confirms the "
        "PDF file was written."
    )
    if existing:
        step = existing[-1]
        if ".pdf" not in step.description.lower():
            step.description = f"{step.description.rstrip()} {instruction}"
            repairs.append(f"step '{step.id}': enforced requested PDF artifact at {path}")
        return

    depended_on = {dependency for step in graph for dependency in step.depends_on}
    leaves = [step_id for step_id in graph.topological_order() if step_id not in depended_on]
    base_id = "create_pdf"
    step_id = base_id
    suffix = 2
    while step_id in graph:
        step_id = f"{base_id}_{suffix}"
        suffix += 1
    graph.add(Step(
        id=step_id,
        name="Create PDF artifact",
        description=instruction,
        agent_role="writer",
        requires_tool="artifact_store",
        depends_on=leaves,
        output_type="text",
        accepts=["code", "json", "text"],
    ))
    graph.validate()
    repairs.append(f"added '{step_id}' because the request requires a real PDF artifact")


class TaskPlanner:
    """Turns a project description into a validated :class:`DependencyGraph`."""

    def __init__(self, llm: Optional[LLMProvider] = None,
                 system_prompt: str = PLANNER_SYSTEM_PROMPT):
        self.llm = llm or get_provider()
        self.system_prompt = system_prompt

    # -- public API --------------------------------------------------------

    def clarify(self, project_description: str,
                max_questions: int = 4) -> List[InputRequest]:
        """Ask what the planner needs to know before it can plan properly.

        Returns typed questions to put to the operator. An empty list means
        the task is clear enough to plan as-is -- which is a good outcome, not
        a failure: interrogating someone about an unambiguous request is worse
        than getting on with it.

        A model that returns nonsense here simply yields no questions; a bad
        clarification round must never block planning.
        """
        try:
            response = self.llm.generate(
                prompt=f"Task:\n{project_description}",
                system=CLARIFIER_SYSTEM_PROMPT, json_mode=True)
        except Exception:  # noqa: BLE001 - clarifying is optional, never fatal
            return []

        parsed = extract_json(response.text) or {}
        raw = parsed.get("questions")
        if not isinstance(raw, list):
            return []

        questions = parse_declared_inputs(raw)
        # Credentials are handled by the setup wizard, which knows where to
        # get each one; a model asking for an API key here would be a worse,
        # unguided version of that flow.
        questions = [q for q in questions
                     if not re.search(r"(api[_\s-]?key|token|password|secret)",
                                      f"{q.name} {q.prompt}", re.IGNORECASE)]
        return questions[:max_questions]

    def plan(self, project_description: str, extra_guidance: str = "",
             tools: Optional[Any] = None,
             clarifications: Optional[Dict[str, Any]] = None) -> PlanResult:
        """Plan a task: specialists, prerequisites and an executable DAG.

        ``tools`` is the registry to check requirements against; pass the one
        the workflow will actually use so the readiness report reflects
        reality (including any attached MCP servers).
        """
        # A caller may use TaskPlanner directly instead of going through
        # Workflow.from_description(). Software plans still need to see the
        # safe file and terminal capabilities or valid tool choices from the
        # model are later discarded as unavailable. The workflow path has
        # already registered run-scoped instances, so only fill the gap here.
        if tools is not None and is_software_task(project_description):
            if "workspace" not in tools or "terminal" not in tools:
                from .tools.workspace import coding_team_tools

                for coding_tool in coding_team_tools("planning"):
                    if coding_tool.name not in tools:
                        tools.register(coding_tool)

        prompt = f"Task:\n{project_description}"
        if clarifications:
            # The operator's answers are facts, not suggestions -- say so, or
            # the model treats them as background and plans around them.
            answers = "\n".join(f"- {k}: {v}" for k, v in clarifications.items() if v not in (None, ""))
            if answers:
                prompt += ("\n\nThe person has already answered these; treat them as "
                           f"fixed requirements:\n{answers}")
        if extra_guidance:
            prompt += f"\n\nAdditional constraints:\n{extra_guidance}"

        # Name the real tools. Without this the model invents plausible ones
        # ("email_service", "mailer") and every step using them fails at run
        # time with nothing to call.
        if tools is not None:
            catalogue = "\n".join(
                f"- {t['name']}: {t['description']}" for t in tools.describe())
            if catalogue:
                prompt += ("\n\nInstalled tools (use their exact names). If none can do the work, "
                           "declare a missing tool requirement and keep that capability on its step; "
                           f"do not pretend it was executed:\n{catalogue}")

        from .llm import provider_for_role
        planner_provider = provider_for_role(self.llm, "planner")
        response = planner_provider.generate(
            prompt, system=self.system_prompt, json_mode=True)
        total_usage = response.usage
        parsed = extract_json(response.text)
        attempt = 1

        # A malformed model response must never turn into an unrelated plan.
        # Ask the same planner to repair its answer, preserving the original
        # task and installed-tool catalogue. If it still cannot produce a
        # usable task list, fail closed so the dashboard can offer a retry.
        while (not parsed or not isinstance(parsed.get("tasks"), list)
               or not parsed["tasks"]) and attempt < MAX_PLAN_ATTEMPTS:
            reason = ("not a valid JSON object" if not parsed
                      else "the 'tasks' array was missing or empty")
            previous = (response.text or "(empty response)")[-12000:]
            repair_prompt = (
                f"{prompt}\n\n"
                f"Your previous planning response was {reason}. Repair it and return "
                "the complete plan again. It must follow the JSON schema in the system "
                "message, contain a non-empty tasks array, use only capabilities the "
                "task actually needs, and contain no prose or markdown fences.\n\n"
                f"Previous response:\n{previous}"
            )
            response = planner_provider.generate(
                repair_prompt, system=self.system_prompt, json_mode=True)
            total_usage = total_usage + response.usage
            parsed = extract_json(response.text)
            attempt += 1

        if not parsed or not isinstance(parsed.get("tasks"), list) or not parsed["tasks"]:
            raise PlanningError(
                f"The AI planner did not return a valid task plan after "
                f"{MAX_PLAN_ATTEMPTS} attempts. No fallback plan was created; retry the "
                "request."
            )

        repairs: List[str] = []
        if attempt > 1:
            repairs.append(f"AI planner repaired its response on attempt {attempt}")
        # Agents are parsed first so the graph can resolve step roles against
        # them -- otherwise a declared 'medical_writer' fuzzy-matches into the
        # built-in 'writer' and the specialist is silently lost.
        agents = self._parse_agents(parsed, repairs)
        declared_roles = {a.role for a in agents}

        available = {t["name"] for t in tools.describe()} if tools is not None else None
        graph, graph_repairs = self.build_graph(parsed["tasks"], declared_roles, available)
        repairs.extend(graph_repairs)
        _ensure_requested_artifacts(graph, project_description, repairs)
        from .change_aware import FactStore, check_plan
        from .config import settings

        raw_facts = parsed.get("facts") or []
        normalised_facts = []
        for raw_fact in raw_facts:
            fact = raw_fact
            if isinstance(raw_fact, dict):
                value = raw_fact.get("value")
                if value is not None and not isinstance(value, (str, list, dict)):
                    fact = dict(raw_fact)
                    fact["value"] = json.dumps(value, ensure_ascii=False)
                    repairs.append(
                        f"normalised fact '{raw_fact.get('key', '(unknown)')}' value to text"
                    )
            normalised_facts.append(fact)

        try:
            graph.facts = FactStore(normalised_facts)
        except (TypeError, ValueError) as exc:
            raise PlanningError(f"invalid requirement facts: {exc}") from exc
        validation = check_plan(graph, settings.plan_token_budget)
        # Now that the graph exists, record which steps each specialist covers.
        for agent in agents:
            if not agent.handles:
                agent.handles = [sid for sid in graph.topological_order()
                                 if graph.get(sid).agent_role == agent.role]

        declared = self._parse_requirements(parsed, repairs)

        # Whatever the planner declared, every tool the steps actually name is
        # added too -- the report must never under-state what is needed.
        report = RequirementsReport(
            requirements=merge_requirements(declared, infer_requirements(graph)),
            agents=agents,
            notes=[str(n) for n in (parsed.get("notes") or []) if str(n).strip()],
            plan_errors=validation["errors"],
            estimated_tokens=validation["estimated_tokens"],
        )
        if tools is None:
            from .tools import default_tool_manager

            tools = default_tool_manager()
        report.check(tools)

        return PlanResult(
            graph=graph,
            usage=total_usage,
            repairs=repairs,
            raw_response=response.text,
            used_fallback_plan=False,
            requirements=report,
        )

    # -- parsing helpers ---------------------------------------------------

    @staticmethod
    def _parse_agents(parsed: Dict[str, Any], repairs: List[str]) -> List[AgentSpec]:
        """Read the planner's invented specialists, keeping only usable ones."""
        from .agents import _slug_role

        specs: List[AgentSpec] = []
        seen: set = set()
        for raw in parsed.get("agents") or []:
            if not isinstance(raw, dict):
                continue
            spec = AgentSpec.from_dict(raw)
            if not spec.role or not spec.system_prompt:
                repairs.append(f"dropped agent '{spec.role or '(unnamed)'}' (missing prompt)")
                continue
            spec.role = _slug_role(spec.role) or "generic"
            if spec.role in seen:
                repairs.append(f"dropped duplicate agent '{spec.role}'")
                continue
            seen.add(spec.role)
            specs.append(spec)
        return specs

    @staticmethod
    def _parse_requirements(parsed: Dict[str, Any], repairs: List[str]) -> List[Requirement]:
        requirements: List[Requirement] = []
        for raw in parsed.get("requirements") or []:
            if not isinstance(raw, dict) or not str(raw.get("name", "")).strip():
                repairs.append("dropped a malformed requirement entry")
                continue
            requirements.append(Requirement.from_dict(raw))
        return requirements

    @staticmethod
    def resolve_tool(name: Optional[str], available: Optional[set]) -> tuple:
        """Map a planner-chosen tool onto one that actually exists.

        Models invent plausible-sounding tool names (``email_service``,
        ``mailer``, ``db``). Left alone, the step fails at runtime with
        nothing to call and the requirements report demands a credential for
        a tool that does not exist. So: canonicalise, then match by substring
        against the real registry, then give up and drop the reference rather
        than plan a step that cannot possibly run.

        Returns ``(resolved_name_or_None, repair_note_or_None)``.
        """
        canonical = canonical_tool_name(name)
        if not canonical:
            return None, None
        if canonical in {"email", "gmail"} and available and "adaptive_email" in available:
            return "adaptive_email", "email uses reviewed API/browser delivery"
        if available and "computer_use" in available and (
                canonical in {"desktop_native", "hermes_desktop", "desktop_automation"}
                or canonical.startswith("web_browser_")
                or canonical.startswith("browser_")):
            return "computer_use", (
                f"low-level tool '{canonical}' cannot complete a multi-action goal; "
                "using guarded computer_use instead")
        if available is None or canonical in available:
            return canonical, None

        # "email_service" -> "email", "mail_sender" -> "gmail", "db" -> handled
        # by aliases above. Prefer the longest overlap so "postgres_cli" does
        # not win over "postgres" for a request of "postgres".
        words = {w for w in re.split(r"[^a-z0-9]+", canonical.lower()) if len(w) > 2}
        candidates = [
            real for real in available
            if real in canonical or canonical in real
            or words & {w for w in re.split(r"[^a-z0-9]+", real.lower()) if len(w) > 2}
        ]
        if candidates:
            best = sorted(candidates, key=lambda r: (-len(set(r) & set(canonical)), len(r)))[0]
            return best, f"tool '{canonical}' does not exist; using '{best}' instead"

        return canonical, (f"tool '{canonical}' does not exist; discover and connect "
                           "the missing capability before execution")

    @staticmethod
    def build_graph(tasks: List[Dict[str, Any]],
                    declared_roles: Optional[set] = None,
                    available_tools: Optional[set] = None) -> tuple:
        """Convert raw task dicts into a valid DAG, recording every repair.

        ``declared_roles`` are specialists the planner defined explicitly; a
        step naming one keeps it verbatim instead of being fuzzy-matched onto
        a built-in role.
        """
        repairs: List[str] = []
        steps: List[Step] = []
        seen_ids: Dict[str, int] = {}

        for index, task in enumerate(tasks):
            if not isinstance(task, dict):
                repairs.append(f"dropped non-object task at index {index}")
                continue

            step_id = _sanitise_id(task.get("id"), index)
            if step_id in seen_ids:
                seen_ids[step_id] += 1
                new_id = f"{step_id}_{seen_ids[step_id]}"
                repairs.append(f"duplicate id '{step_id}' renamed to '{new_id}'")
                step_id = new_id
            seen_ids.setdefault(step_id, 0)

            # resolve_role, not normalise_role: a specialist the planner
            # invented must survive here, or it is silently flattened away.
            role = resolve_role(str(task.get("role") or "generic"), declared_roles)
            description = str(task.get("description") or task.get("name") or step_id).strip()

            raw_deps = task.get("depends_on") or []
            if isinstance(raw_deps, str):
                raw_deps = [raw_deps]
            deps = [_sanitise_id(d, 0) for d in raw_deps if str(d).strip()]

            tool = task.get("tool")
            if isinstance(tool, str) and tool.strip().lower() in {"none", "null", ""}:
                tool = None
            tool, tool_repair = TaskPlanner.resolve_tool(tool, available_tools)
            if tool_repair:
                repairs.append(f"step '{step_id}': {tool_repair}")

            condition = task.get("condition")
            if isinstance(condition, dict):
                valid_ops = {"contains", "not_contains", "equals", "not_equals",
                             "exists", "truthy"}
                source = str(condition.get("source", "")).strip()
                operator = str(condition.get("operator", "truthy")).strip().lower()
                if not source or operator not in valid_ops:
                    repairs.append(f"step '{step_id}': dropped invalid condition")
                    condition = None
                else:
                    condition = {"source": source, "operator": operator,
                                 "value": condition.get("value")}
            else:
                condition = None

            steps.append(Step(
                id=step_id,
                name=str(task.get("name") or "").strip(),
                description=description,
                agent_role=role,
                depends_on=deps,
                requires_tool=tool,
                inputs=filter_inputs(parse_declared_inputs(task.get("inputs"))),
                condition=condition,
                output_type=str(task.get("output_type", "text")),
                accepts=task.get("accepts", ["code", "json", "text"]),
                assumes=task.get("assumes", []),
            ))

        if not steps:
            raise PlanningError("planner produced no usable tasks")

        # adaptive_email owns its browser lifecycle. Models sometimes insert
        # a separate "open Gmail/login" computer_use step; that wastes a full
        # model call and, worse, opens a different browser session from the
        # reviewed draft. Remove only those clearly mail-specific precursors,
        # preserving genuine browser research that may feed the message.
        mail_browser = {
            step.id for step in steps
            if step.requires_tool == "computer_use"
            and re.search(
                r"(?i)(?:gmail|mail\.google|email).{0,50}(?:login|sign[ -]?in|browser|prepare)|"
                r"(?:login|sign[ -]?in|browser|prepare).{0,50}(?:gmail|mail\.google|email)",
                step.description,
            )
        }
        if mail_browser and any(step.requires_tool == "adaptive_email" for step in steps):
            by_id = {step.id: step for step in steps}
            for step in steps:
                if step.id in mail_browser:
                    continue
                rewritten: List[str] = []
                for dependency in step.depends_on:
                    if dependency in mail_browser:
                        rewritten.extend(by_id[dependency].depends_on)
                    else:
                        rewritten.append(dependency)
                step.depends_on = list(dict.fromkeys(rewritten))
            for step_id in sorted(mail_browser):
                repairs.append(
                    f"dropped redundant mail-browser step '{step_id}'; adaptive_email owns Gmail")
            steps = [step for step in steps if step.id not in mail_browser]

        # Cap questions across the whole workflow, keeping the earliest steps'
        # (they block the most). Being asked eight things up front reads as an
        # interrogation and is usually a sign the model over-declared.
        budget = MAX_INPUTS_PER_WORKFLOW
        for step in steps:
            if budget <= 0:
                if step.inputs:
                    repairs.append(
                        f"dropped {len(step.inputs)} question(s) on '{step.id}' "
                        f"(workflow limit of {MAX_INPUTS_PER_WORKFLOW} reached)")
                step.inputs = []
                continue
            if len(step.inputs) > budget:
                repairs.append(f"trimmed questions on '{step.id}' to fit the workflow limit")
                step.inputs = step.inputs[:budget]
            budget -= len(step.inputs)

        graph = DependencyGraph(steps)

        for step_id, dep in graph.prune_dangling_dependencies():
            repairs.append(f"dropped dependency '{step_id}' -> '{dep}' (no such step)")
        for step_id, dep in graph.break_cycles():
            repairs.append(f"broke cycle by removing dependency '{step_id}' -> '{dep}'")

        graph.validate()
        return graph, repairs

    def revise_step(self, step: Step, new_requirement: str) -> str:
        """Rewrite a step's description to fold in a changed requirement.

        Used by ``Workflow.handle_step_change`` when the caller supplies a new
        requirement rather than just invalidating the step.
        """
        response = self.llm.generate(
            prompt=(
                f"Current task description:\n{step.description}\n\n"
                f"New requirement from the client:\n{new_requirement}\n\n"
                "Rewrite the task description so it satisfies the new requirement. "
                "Reply with the rewritten description only, one or two sentences."
            ),
            system="You rewrite task descriptions precisely and concisely.",
        )
        revised = response.text.strip()
        return revised or f"{step.description} (updated: {new_requirement})"

def plan_from_json(path: str) -> DependencyGraph:
    """Load a hand-written plan from a JSON file.

    Accepts either ``{"tasks": [...]}`` or a bare list of task objects.
    """
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, list):
        tasks = data
    elif isinstance(data, dict):
        tasks = data.get("tasks", [])
    else:
        raise PlanningError(f"{path}: expected an object or a list of tasks")
    graph, _ = TaskPlanner.build_graph(tasks)
    return graph
