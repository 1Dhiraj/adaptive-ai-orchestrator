"""Build the First Review deck from the team's own Zeroth Review template.

Keeps slide 1 (title/team) and the template's design -- logo, fonts, footer,
slide numbers -- and rebuilds the body to cover the 11 points the First
Review circular asks for, in that order.
"""

from __future__ import annotations

import copy
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Emu, Inches, Pt

TEMPLATE = Path("C:/Users/91934/Downloads/Zeroth Review - Multi-Agent Orchestrator system.pptx")
OUTPUT = Path("First Review - Multi-Agent Orchestrator system.pptx")

REVIEW_DATE = "12/08/2026"
FOOTER = "First Review , Department of Computer Science and Engineering, VCET, Madurai"

# Template palette, sampled from the Zeroth Review deck.
NAVY = RGBColor(0x1F, 0x3B, 0x57)
TEAL = RGBColor(0x1B, 0x6C, 0x8A)
ORANGE = RGBColor(0xE8, 0x71, 0x22)
GREEN = RGBColor(0x1E, 0x7A, 0x3C)
PURPLE = RGBColor(0x8E, 0x24, 0x9E)
GREY = RGBColor(0xE9, 0xEC, 0xEF)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
DARK = RGBColor(0x21, 0x25, 0x29)

# One type scale for the whole deck. Every slide draws from these four values
# only -- the first build let each slide pick its own size, which is why body
# text ranged from 9pt to 20pt and the deck read as uneven.
TITLE = 36      # slide titles
BODY = 18       # every bullet at level 0, on every slide
SUB = 16        # level-1 sub-bullets
TABLE = 14      # every table cell
CAPTION = 14    # notes and captions sitting outside the bullet placeholder
DIAGRAM = 12    # labels inside drawn boxes, which are too small to take BODY


# ---------------------------------------------------------------------------
# Template plumbing
# ---------------------------------------------------------------------------

def load() -> Presentation:
    if not TEMPLATE.exists():
        raise SystemExit(f"template not found: {TEMPLATE}")
    return Presentation(str(TEMPLATE))


def logo_blob(prs: Presentation) -> bytes:
    """The VCET logo image, lifted from an existing slide."""
    for slide in prs.slides:
        for shape in slide.shapes:
            if shape.shape_type == 13:  # PICTURE
                return shape.image.blob
    raise SystemExit("no logo picture found in the template")


def drop_slides_after_first(prs: Presentation) -> None:
    """Remove every slide except slide 1, which the team wants kept."""
    xml_slides = prs.slides._sldIdLst
    for sld in list(xml_slides)[1:]:
        rid = sld.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
        prs.part.drop_rel(rid)
        xml_slides.remove(sld)


def add_slide(prs: Presentation, title: str, logo: bytes):
    """A content slide carrying the template's logo, footer and numbering."""
    slide = prs.slides.add_slide(prs.slide_layouts[1])   # Title and Content
    slide.shapes.title.text = title

    # Match the template: logo top-left, title indented past it.
    slide.shapes.add_picture(__import__("io").BytesIO(logo), 0, Inches(0.20),
                             width=Inches(1.46), height=Inches(1.50))
    t = slide.shapes.title
    t.left, t.top, t.width, t.height = Inches(1.46), Inches(0.40), Inches(10.95), Inches(1.20)
    t.text_frame.paragraphs[0].runs[0].font.size = Pt(TITLE)

    for ph in slide.placeholders:
        idx = ph.placeholder_format.idx
        if idx == 10:
            ph.text_frame.text = REVIEW_DATE
        elif idx == 11:
            ph.text_frame.text = FOOTER
    return slide


def body_of(slide):
    for ph in slide.placeholders:
        if ph.placeholder_format.idx == 1:
            return ph
    return None


def bullets(slide, items, size=BODY, spacing=8):
    """Fill the content placeholder. Items are (text, level) or plain strings."""
    body = body_of(slide)
    body.left, body.top = Inches(0.92), Inches(1.85)
    body.width, body.height = Inches(11.50), Inches(4.90)
    tf = body.text_frame
    tf.word_wrap = True
    tf.clear()
    for i, item in enumerate(items):
        text, level = (item if isinstance(item, tuple) else (item, 0))
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.level = level
        p.space_after = Pt(spacing)
        # Split on ** so "**Lead-in:** body" becomes a bold run then a normal
        # one. Setting p.text and stripping asterisks afterwards leaves the
        # closing pair stranded mid-sentence.
        for j, chunk in enumerate(text.split("**")):
            if not chunk:
                continue
            run = p.add_run()
            run.text = chunk
            run.font.size = Pt(size if level == 0 else SUB)
            run.font.bold = (j % 2 == 1)   # odd chunks sat between ** markers
    return body


def remove_body(slide):
    body = body_of(slide)
    if body is not None:
        body._element.getparent().remove(body._element)


def box(slide, x, y, w, h, text, fill, font=WHITE, size=DIAGRAM, bold=True, radius=True):
    from pptx.enum.shapes import MSO_SHAPE

    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE, x, y, w, h)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = fill
    shape.shadow.inherit = False
    tf = shape.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = Inches(0.05)
    p = tf.paragraphs[0]
    p.text = text
    p.alignment = PP_ALIGN.CENTER
    for run in p.runs:
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = font
    return shape


def arrow(slide, x1, y1, x2, y2, colour=NAVY):
    from pptx.enum.shapes import MSO_CONNECTOR

    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, x1, y1, x2, y2)
    c.line.color.rgb = colour
    c.line.width = Pt(1.5)
    return c


def table(slide, rows, x, y, w, h, col_widths=None, header_fill=NAVY, size=TABLE):
    shape = slide.shapes.add_table(len(rows), len(rows[0]), x, y, w, h)
    tbl = shape.table
    if col_widths:
        for i, cw in enumerate(col_widths):
            tbl.columns[i].width = cw
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            cell = tbl.cell(r, c)
            cell.text = str(val)
            cell.margin_left = cell.margin_right = Inches(0.06)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            for p in cell.text_frame.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(size)
                    run.font.bold = (r == 0)
                    run.font.color.rgb = WHITE if r == 0 else DARK
            if r == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = header_fill
    return tbl


# ---------------------------------------------------------------------------
# The deck
# ---------------------------------------------------------------------------

def build() -> None:
    prs = load()
    logo = logo_blob(prs)

    # -- slide 1: keep, but retitle the review ---------------------------
    first = prs.slides[0]
    for shape in first.shapes:
        if not shape.has_text_frame:
            continue
        for p in shape.text_frame.paragraphs:
            for run in p.runs:
                run.text = (run.text.replace("Zeroth Review", "First Review")
                                    .replace("18/07/2026", REVIEW_DATE))
    drop_slides_after_first(prs)

    # -- 2. Contents ------------------------------------------------------
    s = add_slide(prs, "Contents", logo)
    bullets(s, [
        "Problem Statement & Motivation",
        "Objectives of the Project",
        "Literature Review",
        "Proposed Methodology / Approach",
        "System Architecture",
        "Novelty / Contribution",
        "Expected Outcomes & Results Achieved",
        "Implementation Plan & Timeline",
        "Base Paper & Target Conference / Journal",
        "References",
    ], spacing=10)

    # -- 3. Problem Statement & Motivation --------------------------------
    s = add_slide(prs, "Problem Statement & Motivation", logo)
    bullets(s, [
        "Existing multi-agent frameworks (AutoGen, CrewAI, LangGraph) execute tasks "
        "as a fixed pipeline: Plan → Build → Test.",
        "A single mid-execution requirement change — e.g. \"use PostgreSQL instead of "
        "MongoDB\" — forces a full restart or manual re-running of steps.",
        "This wastes execution time, LLM tokens and API cost, which grows linearly "
        "with workflow size.",
        "No standard mechanism exists to determine which parts of a workflow a change "
        "actually invalidates, or to recover when a tool fails mid-task.",
        "Motivation: in real projects requirements change constantly. An orchestrator "
        "should adapt a running workflow, not restart it.",
    ])

    # -- 4. Objectives ----------------------------------------------------
    s = add_slide(prs, "Objectives of the Project", logo)
    bullets(s, [
        "To design and implement an adaptive, capability-aware orchestrator that "
        "selectively re-executes only the affected part of a workflow when "
        "requirements change or a tool fails.",
        ("Decompose a plain-English task into a dependency-tracked workflow graph (DAG)", 1),
        ("Dynamically create role-specialised agents based on the task, not a fixed list", 1),
        ("Select and swap tools (MCP, REST, CLI) transparently via a Tool Manager", 1),
        ("Detect change and identify only the genuinely affected downstream nodes", 1),
        ("Recover automatically from tool failure using capability-equivalent alternatives", 1),
        ("Reuse prior workflow memory to reduce redundant execution and cost", 1),
        ("Gate irreversible real-world actions behind explicit human approval", 1),
    ], spacing=5)

    # -- 5. Literature Review --------------------------------------------
    s = add_slide(prs, "Literature Review", logo)
    remove_body(s)
    table(s, [
        ["Paper", "Approach", "Limitation Addressed"],
        ["AutoGen (Wu et al., 2023)",
         "Multi-agent conversation framework; agents converse to solve tasks",
         "Conversational loops are token-heavy; no dependency tracking or selective re-run"],
        ["MetaGPT (Hong et al., 2023)",
         "Assigns SOP-based roles (PM, architect, engineer) in a fixed assembly line",
         "Roles and pipeline are hard-coded; a change restarts the line"],
        ["ReAct (Yao et al., 2022)",
         "Interleaves reasoning traces with tool actions in a single agent",
         "Single-agent, linear; no orchestration or failure recovery across steps"],
        ["Reflexion (Shinn et al., 2023)",
         "Verbal self-reflection to retry failed attempts",
         "Retries the same step; cannot reason about workflow-level impact"],
        ["Toolformer (Schick et al., 2023)",
         "LLM learns when to call APIs via self-supervision",
         "Tool choice is static at inference; no fallback when a tool is unavailable"],
    ], Inches(0.55), Inches(1.75), Inches(12.2), Inches(4.6),
        col_widths=[Inches(2.7), Inches(4.4), Inches(5.1)])

    # -- 6. Research Gap --------------------------------------------------
    s = add_slide(prs, "Research Gap Identified", logo)
    bullets(s, [
        "**Gap 1 — No impact analysis:** existing systems cannot compute which steps a "
        "requirement change actually invalidates.",
        "**Gap 2 — Restart-on-change:** adaptation is achieved by re-running the whole "
        "pipeline, wasting prior valid work.",
        "**Gap 3 — Static tool binding:** a failed or unavailable tool halts the workflow; "
        "substitution is manual.",
        "**Gap 4 — No memory reuse:** identical sub-tasks are recomputed across runs.",
        "**Gap 5 — Unsafe autonomy:** irreversible actions (email, commits, migrations) "
        "execute without a human checkpoint.",
        "**Our work targets all five within a single orchestration algorithm.**",
    ])

    # -- 7. Proposed Methodology ------------------------------------------
    s = add_slide(prs, "Proposed Methodology / Approach", logo)
    bullets(s, [
        "**1. Task Planning** — an LLM decomposes the task into a validated DAG, inventing "
        "the specialist agents and declaring the capabilities it needs.",
        "**2. Dependency Graph** — steps are topologically ordered; independent steps are "
        "grouped into levels and executed in parallel.",
        "**3. Input Fingerprinting** — each step is hashed over its own definition plus the "
        "exact outputs of its dependencies.",
        "**4. Selective Re-execution** — a step runs only if forced, uncached, or its "
        "fingerprint changed. Change therefore propagates automatically and stops "
        "when a re-run reproduces identical output.",
        "**5. Capability-based Tool Routing** — tools are grouped by capability; the manager "
        "falls back to an equivalent tool when one fails.",
        "**6. Human-in-the-loop Gate** — irreversible actions pause and show exactly what "
        "will happen before executing.",
    ], spacing=4)

    # -- 8. Core mechanism -------------------------------------------------
    s = add_slide(prs, "Core Mechanism: Selective Re-execution", logo)
    remove_body(s)
    tb = s.shapes.add_textbox(Inches(0.92), Inches(1.75), Inches(11.5), Inches(0.9))
    p = tb.text_frame.paragraphs[0]
    p.text = "A step re-runs  ⟺  it was forced,  or has no cached result,  " \
             "or its input fingerprint no longer matches"
    p.runs[0].font.size = Pt(CAPTION); p.runs[0].font.bold = True; p.runs[0].font.color.rgb = NAVY

    lbl = s.shapes.add_textbox(Inches(0.92), Inches(2.60), Inches(11.5), Inches(0.3))
    lp = lbl.text_frame.paragraphs[0]
    lp.text = "Traditional systems — full restart on requirement change"
    lp.runs[0].font.size = Pt(CAPTION); lp.runs[0].font.bold = True; lp.runs[0].font.color.rgb = ORANGE

    W, H, GAP = Inches(2.55), Inches(0.62), Inches(0.28)
    names = ["Frontend", "Backend", "Database", "Testing"]
    for i, n in enumerate(names):
        box(s, Inches(0.92) + i * (W + GAP), Inches(2.95), W, H, n, ORANGE)

    lbl2 = s.shapes.add_textbox(Inches(0.92), Inches(3.85), Inches(11.5), Inches(0.3))
    lp2 = lbl2.text_frame.paragraphs[0]
    lp2.text = "Our orchestrator — selective re-execution (MongoDB → PostgreSQL)"
    lp2.runs[0].font.size = Pt(CAPTION); lp2.runs[0].font.bold = True; lp2.runs[0].font.color.rgb = GREEN

    for i, n in enumerate(names):
        reused = i < 2          # frontend and backend keep their cached output
        box(s, Inches(0.92) + i * (W + GAP), Inches(4.20), W, H, n,
            GREEN if reused else ORANGE)

    key = s.shapes.add_textbox(Inches(0.92), Inches(5.00), Inches(11.5), Inches(0.35))
    kp = key.text_frame.paragraphs[0]
    kp.text = "green = reused (0 tokens, 0 s)      orange = re-executed"
    kp.runs[0].font.size = Pt(CAPTION); kp.runs[0].font.color.rgb = DARK

    note = s.shapes.add_textbox(Inches(0.92), Inches(5.45), Inches(11.5), Inches(1.1))
    np_ = note.text_frame; np_.word_wrap = True
    n1 = np_.paragraphs[0]
    n1.text = ("Tool-failure recovery: if an assigned tool becomes unavailable, the Tool "
               "Manager substitutes a capability-equivalent alternative "
               "(GitHub API → GitHub CLI → local artifact store) so execution continues.")
    n1.runs[0].font.size = Pt(CAPTION)

    # -- 9. System Architecture -------------------------------------------
    s = add_slide(prs, "System Architecture", logo)
    remove_body(s)
    cx = Inches(5.20)
    box(s, cx, Inches(1.70), Inches(2.9), Inches(0.45), "User", NAVY)
    box(s, Inches(3.60), Inches(2.35), Inches(6.1), Inches(0.50),
        "Adaptive AI Orchestrator", ORANGE)

    layer2 = [("Task Planner", TEAL), ("Workflow Engine", TEAL), ("Memory Manager", TEAL)]
    for i, (n, c) in enumerate(layer2):
        box(s, Inches(1.30) + i * Inches(3.55), Inches(3.10), Inches(3.20), Inches(0.48), n, c)

    box(s, Inches(1.30), Inches(3.90), Inches(4.9), Inches(0.48), "Agent Manager", GREEN)
    box(s, Inches(6.90), Inches(3.90), Inches(4.9), Inches(0.48), "Tool Manager", PURPLE)

    agents = ["Planner-created\nspecialist agents", "Role prompts\n+ shared context"]
    for i, n in enumerate(agents):
        box(s, Inches(1.30) + i * Inches(2.55), Inches(4.70), Inches(2.35), Inches(0.60),
            n, GREY, font=DARK, size=DIAGRAM, bold=False)
    tools = ["MCP servers\n(Playwright, …)", "REST / API\nconnections", "CLI & local\nfallbacks"]
    for i, n in enumerate(tools):
        box(s, Inches(6.90) + i * Inches(1.68), Inches(4.70), Inches(1.55), Inches(0.60),
            n, GREY, font=DARK, size=DIAGRAM, bold=False)

    box(s, Inches(1.30), Inches(5.55), Inches(10.5), Inches(0.45),
        "Persistence (SQLite)  •  Event Bus  •  Requirements Checker  •  Approval Gate  •  Dashboard",
        NAVY)

    cap = s.shapes.add_textbox(Inches(1.30), Inches(6.10), Inches(10.5), Inches(0.4))
    cp = cap.text_frame.paragraphs[0]
    cp.text = "Agents and tools are created / selected on demand — nothing here is hard-coded."
    cp.runs[0].font.size = Pt(CAPTION); cp.runs[0].font.italic = True; cp.runs[0].font.color.rgb = DARK
    cp.alignment = PP_ALIGN.CENTER

    # -- 10. Novelty -------------------------------------------------------
    s = add_slide(prs, "Novelty / Contribution of the Work", logo)
    bullets(s, [
        "**1. Fingerprint-based change propagation.** A step's hash covers its definition "
        "plus its dependencies' exact outputs — so change propagates on its own, and "
        "stops the moment a re-run reproduces identical output. Dependency analysis "
        "alone cannot do this.",
        "**2. Capability-equivalent tool fallback.** Tools are grouped by capability with "
        "chains ending in an always-available local tool, so a workflow completes even "
        "with every remote service down.",
        "**3. Planner-invented specialists.** Agent roles are generated per task "
        "(e.g. clinical_trial_statistician, employment_lawyer), not chosen from a fixed list.",
        "**4. Pre-execution requirements analysis.** The system reports what it needs "
        "(credentials, MCP servers, binaries) and whether it can run — before spending a token.",
        "**5. Cost-free approval of irreversible actions.** The generated payload is held, "
        "so human sign-off does not re-invoke the LLM.",
    ], spacing=4)

    # -- 11. Expected Outcomes + results ----------------------------------
    s = add_slide(prs, "Expected Outcomes & Results Achieved", logo)
    remove_body(s)
    tb = s.shapes.add_textbox(Inches(0.92), Inches(1.75), Inches(11.5), Inches(0.4))
    p = tb.text_frame.paragraphs[0]
    p.text = "Benchmarked against LangGraph and a restart-all baseline over 6 scenarios " \
             "(identical agent work, deterministic LLM)"
    p.runs[0].font.size = Pt(CAPTION); p.runs[0].font.italic = True; p.runs[0].font.color.rgb = DARK

    table(s, [
        ["Metric", "Restart-all", "LangGraph", "Our System"],
        ["Total step executions", "71", "71", "54  (−24%)"],
        ["Total tokens consumed", "24,240", "24,240", "18,990  (−22%)"],
        ["Leaf-change scenario", "10 steps", "10 steps", "6 steps  (−40%)"],
        ["No-op re-run scenario", "10 steps", "10 steps", "6 steps  (−40%)"],
        ["Tool-failure recovery", "manual", "manual", "automatic fallback"],
        ["Workflow restart", "full", "full", "selective subgraph"],
    ], Inches(1.30), Inches(2.30), Inches(10.5), Inches(3.0),
        col_widths=[Inches(3.6), Inches(2.2), Inches(2.2), Inches(2.5)])

    out = s.shapes.add_textbox(Inches(1.30), Inches(5.45), Inches(10.5), Inches(1.1))
    of = out.text_frame; of.word_wrap = True
    o1 = of.paragraphs[0]
    o1.text = "Saving scales with where the change lands: 0% when a change invalidates the " \
              "whole graph, up to 40% for late or no-op changes."
    o1.runs[0].font.size = Pt(CAPTION); o1.runs[0].font.color.rgb = DARK

    # -- 12. Implementation status ----------------------------------------
    s = add_slide(prs, "Implementation Status (Working Prototype)", logo)
    bullets(s, [
        "**Core engine complete** — planner, dependency graph, agent manager, tool manager, "
        "memory, persistence, event bus; 610+ automated tests, ~90% code coverage.",
        "**Real MCP integration** — connects to live MCP servers over stdio/SSE/HTTP; "
        "Playwright MCP contributes 24 browser tools to the agents.",
        "**Real tool execution** — GitHub REST, PostgreSQL, Gmail SMTP, Slack, and "
        "authenticated API connections (bearer / basic / API-key / OAuth2).",
        "**End-to-end demonstration** — a workflow that opens a live web page in a real "
        "browser, scrapes the weather, drafts an email and delivers it via Gmail SMTP, "
        "pausing for human approval before sending.",
        "**Web dashboard** — live dependency graph over WebSocket, guided credential setup, "
        "workflow editing, and JSON / CSV / HTML export.",
    ], spacing=5)

    # -- 13. Implementation Plan & Timeline (Gantt) -----------------------
    s = add_slide(prs, "Implementation Plan & Timeline", logo)
    remove_body(s)
    # Weeks 1-3 are what the team actually built between the Zeroth Review
    # (18/07/2026) and this one. Everything past week 3 is planned work and is
    # labelled as such -- no completion is claimed for it.
    WEEKS = 8
    phases = [
        ("Literature survey & requirements", 0, 1, TEAL, True),
        ("Core orchestrator (planner + graph)", 0, 2, TEAL, True),
        ("Agent & tool layer + MCP", 1, 3, TEAL, True),
        ("Adaptive re-execution & recovery", 2, 3, TEAL, True),
        ("Dashboard, memory & testing", 2, 3, TEAL, True),
        ("Extended benchmarking (AutoGen / CrewAI)", 3, 5, GREEN, False),
        ("Paper writing & submission", 5, 8, ORANGE, False),
    ]
    left0, top0 = Inches(4.35), Inches(2.15)
    week_w = Inches(1.00)
    row_h, row_gap = Inches(0.36), Inches(0.10)

    for w in range(WEEKS):
        lab = s.shapes.add_textbox(left0 + w * week_w, Inches(1.80),
                                   week_w, Inches(0.25))
        lp = lab.text_frame.paragraphs[0]; lp.text = f"W{w + 1}"
        lp.runs[0].font.size = Pt(DIAGRAM); lp.runs[0].font.color.rgb = DARK
        lp.alignment = PP_ALIGN.CENTER

    for i, (name, start, end, colour, done) in enumerate(phases):
        y = top0 + i * (row_h + row_gap)
        lbl = s.shapes.add_textbox(Inches(0.60), y, Inches(3.70), row_h)
        lf = lbl.text_frame; lf.word_wrap = True
        lf.vertical_anchor = MSO_ANCHOR.MIDDLE
        lp = lf.paragraphs[0]; lp.text = name
        lp.runs[0].font.size = Pt(DIAGRAM); lp.runs[0].font.color.rgb = DARK
        bar = box(s, left0 + start * week_w, y, (end - start) * week_w, row_h,
                  "completed" if done else "planned", colour)
        if not done:
            bar.fill.fore_color.rgb = colour

    legend = s.shapes.add_textbox(Inches(0.60), Inches(5.35), Inches(12.2), Inches(1.0))
    lf = legend.text_frame; lf.word_wrap = True
    l1 = lf.paragraphs[0]
    l1.text = "Weeks 1–3 completed (18/07/2026 – 12/08/2026): working prototype with tested " \
              "adaptive re-execution, tool fallback and MCP integration."
    l1.runs[0].font.size = Pt(CAPTION); l1.runs[0].font.bold = True; l1.runs[0].font.color.rgb = GREEN
    l2 = lf.add_paragraph()
    l2.text = "Weeks 4–8 are planned work: benchmarking against additional frameworks, " \
              "then paper preparation and submission."
    l2.runs[0].font.size = Pt(CAPTION); l2.runs[0].font.color.rgb = DARK

    # -- 14. Base paper -> what we took -> target venue --------------------
    # The base paper summary is drawn from its IEEE Xplore record and its
    # published abstract. The full text is paywalled and has not been read,
    # so nothing below claims more than the abstract states.
    s = add_slide(prs, "Identified Conference / Journal", logo)
    bullets(s, [
        "**Base paper —** “Dynamic Workflow Orchestration with Executability Verification "
        "for Multi-Agent AutoML Task Execution,” L. Deng et al., CAIBDA 2026, IEEE.",
        ("DOI: 10.1109/CAIBDA70336.2026.11621590  ·  ieeexplore.ieee.org/document/11621590", 1),
        "**What we took from it —** it models multi-agent execution as a directed acyclic "
        "graph whose orchestrator builds the task graph and assigns subtasks to specialised "
        "agents. Our workflow engine follows the same structure.",
        "Its subagents verify capability–task matching before executing; we generalised that "
        "into capability-based tool routing with automatic fallback.",
        "It repairs by replanning after a failure, but does not decide what remains valid "
        "when a requirement changes — that gap is our contribution.",
        "**Where we will publish —** the base paper sits in the IEEE conference literature "
        "on multi-agent orchestration, so we target the same venue class:",
        ("IEEE ICCCNT — annual IEEE series; 17th edition at IIT Delhi, June 2026; "
         "IEEE Xplore indexed. 2027 dates not yet announced.", 1),
        ("Springer ICACDS / ICACCS (backup, Scopus indexed)  ·  IEEE Access "
         "(extended journal version).", 1),
    ], spacing=4)

    # -- 15. References ----------------------------------------------------
    s = add_slide(prs, "References", logo)
    bullets(s, [
        "[1] L. Deng et al., “Dynamic Workflow Orchestration with Executability Verification "
        "for Multi-Agent AutoML Task Execution,” CAIBDA, 2026.  (base paper)",
        "[2] Q. Wu et al., “AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent "
        "Conversation Framework,” arXiv:2308.08155, 2023.",
        "[3] S. Hong et al., “MetaGPT: Meta Programming for a Multi-Agent Collaborative "
        "Framework,” arXiv:2308.00352, 2023.",
        "[4] S. Yao et al., “ReAct: Synergizing Reasoning and Acting in Language Models,” "
        "arXiv:2210.03629, 2022.",
        "[5] N. Shinn et al., “Reflexion: Language Agents with Verbal Reinforcement "
        "Learning,” arXiv:2303.11366, 2023.",
        "[6] T. Schick et al., “Toolformer: Language Models Can Teach Themselves to Use "
        "Tools,” arXiv:2302.04761, 2023.",
        "[7] Anthropic, “Model Context Protocol Specification,” 2024. "
        "modelcontextprotocol.io",
        "[8] LangChain, “LangGraph Documentation,” 2024. langchain-ai.github.io/langgraph",
    ], spacing=2)

    # -- 16. Thank You -----------------------------------------------------
    s = prs.slides.add_slide(prs.slide_layouts[0])
    s.shapes.title.text = "Thank You"
    s.shapes.add_picture(__import__("io").BytesIO(logo), 0, Inches(0.20),
                         width=Inches(1.46), height=Inches(1.50))
    for ph in list(s.placeholders):
        idx = ph.placeholder_format.idx
        if idx == 10:
            ph.text_frame.text = REVIEW_DATE
        elif idx == 11:
            ph.text_frame.text = FOOTER
        elif idx == 1:
            ph._element.getparent().remove(ph._element)

    prs.save(str(OUTPUT))
    print(f"written: {OUTPUT.resolve()}")
    print(f"slides: {len(prs.slides)}")


if __name__ == "__main__":
    build()
