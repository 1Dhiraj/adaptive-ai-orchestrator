"""Layer 6b -- observability exports: JSON, CSV and a static HTML report."""

from __future__ import annotations

import csv
import html
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:  # pragma: no cover
    from .workflow import Workflow

STATUS_COLOURS = {
    "done": "#2ecc71",
    "skipped": "#3498db",
    "failed": "#e74c3c",
    "cancelled": "#7f8c8d",
    "paused": "#f39c12",
    "running": "#9b59b6",
    "pending": "#5d6470",
    "stale": "#e67e22",
}


def _write(path: str, content: str) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return str(target.resolve())


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------


def export_json(workflow: "Workflow", path: str = "results.json") -> str:
    return _write(path, json.dumps(workflow.to_dict(), indent=2, default=str))


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------


def export_csv(workflow: "Workflow", path: str = "results.csv") -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "step_id", "name", "role", "depends_on", "status", "duration_s", "attempts",
        "tool_requested", "tool_used", "used_fallback", "llm_calls", "prompt_tokens",
        "completion_tokens", "total_tokens", "cost_usd", "input_hash", "output_hash",
        "error", "output",
    ]
    with target.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for step_id in workflow.graph.topological_order():
            step = workflow.graph.get(step_id)
            result = workflow.results[step_id]
            writer.writerow({
                "step_id": step_id,
                "name": step.name,
                "role": step.agent_role,
                "depends_on": "|".join(step.depends_on),
                "status": result.status.value,
                "duration_s": round(result.duration_s, 4),
                "attempts": result.attempts,
                "tool_requested": result.tool_requested or "",
                "tool_used": result.tool_used or "",
                "used_fallback": result.used_fallback,
                "llm_calls": result.usage.calls,
                "prompt_tokens": result.usage.prompt_tokens,
                "completion_tokens": result.usage.completion_tokens,
                "total_tokens": result.usage.total_tokens,
                "cost_usd": result.usage.cost_usd,
                "input_hash": result.input_hash,
                "output_hash": result.output_hash,
                "error": result.error or "",
                # Collapse newlines so the row stays one line in a spreadsheet.
                "output": (result.output or "").replace("\r", " ").replace("\n", " ")[:2000],
            })
    return str(target.resolve())


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------


def _svg_dag(workflow: "Workflow") -> str:
    """Layered DAG drawing. Levels become rows; steps within a level, columns."""
    levels: List[List[str]] = workflow.graph.execution_levels()
    if not levels:
        return "<p>empty graph</p>"

    box_w, box_h = 190, 58
    gap_x, gap_y = 40, 84
    widest = max(len(level) for level in levels)
    width = widest * (box_w + gap_x) + gap_x
    height = len(levels) * (box_h + gap_y) + gap_y

    centres: Dict[str, tuple] = {}
    for row, level in enumerate(levels):
        row_width = len(level) * (box_w + gap_x) - gap_x
        start_x = (width - row_width) / 2
        for col, step_id in enumerate(level):
            x = start_x + col * (box_w + gap_x)
            y = gap_y / 2 + row * (box_h + gap_y)
            centres[step_id] = (x, y)

    edges: List[str] = []
    for step_id, (x, y) in centres.items():
        for dep in workflow.graph.get(step_id).depends_on:
            if dep not in centres:
                continue
            dx, dy = centres[dep]
            edges.append(
                f'<path d="M {dx + box_w / 2} {dy + box_h} '
                f'C {dx + box_w / 2} {dy + box_h + gap_y / 2}, '
                f'{x + box_w / 2} {y - gap_y / 2}, {x + box_w / 2} {y}" '
                f'fill="none" stroke="#4a5162" stroke-width="1.6" marker-end="url(#arrow)"/>'
            )

    boxes: List[str] = []
    for step_id, (x, y) in centres.items():
        step = workflow.graph.get(step_id)
        result = workflow.results[step_id]
        colour = STATUS_COLOURS.get(result.status.value, "#5d6470")
        tool = f" - {result.tool_used or step.requires_tool}" if step.requires_tool else ""
        boxes.append(
            f'<g><rect x="{x}" y="{y}" width="{box_w}" height="{box_h}" rx="8" '
            f'fill="#1b1f2a" stroke="{colour}" stroke-width="2.5"/>'
            f'<text x="{x + 12}" y="{y + 22}" fill="#e8ecf4" font-size="13" '
            f'font-family="ui-monospace,monospace">{html.escape(step_id[:22])}</text>'
            f'<text x="{x + 12}" y="{y + 41}" fill="{colour}" font-size="11" '
            f'font-family="system-ui,sans-serif">{result.status.value}'
            f'{html.escape(tool)} - {result.duration_s:.2f}s</text></g>'
        )

    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
        f'aria-label="workflow dependency graph">'
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
        'markerHeight="6" orient="auto-start-reverse">'
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#4a5162"/></marker></defs>'
        + "".join(edges) + "".join(boxes) + "</svg>"
    )


def export_html(workflow: "Workflow", path: str = "dashboard.html") -> str:
    metrics: Dict[str, Any] = workflow.metrics()
    usage = metrics["usage"]

    tiles = [
        ("Steps", metrics["steps"]),
        ("Executed", metrics["by_status"].get("done", 0)),
        ("Reused", metrics["by_status"].get("skipped", 0)),
        ("Failed", metrics["by_status"].get("failed", 0)),
        ("LLM calls", usage["calls"]),
        ("Tokens", f"{usage['total_tokens']:,}"),
        ("Cost", f"${usage['cost_usd']:.6f}"),
        ("Step time", f"{metrics['total_step_seconds']:.2f}s"),
    ]
    tile_html = "".join(
        f'<div class="tile"><div class="tile-label">{html.escape(str(label))}</div>'
        f'<div class="tile-value">{html.escape(str(value))}</div></div>'
        for label, value in tiles
    )

    cards: List[str] = []
    for step_id in workflow.graph.topological_order():
        step = workflow.graph.get(step_id)
        result = workflow.results[step_id]
        colour = STATUS_COLOURS.get(result.status.value, "#5d6470")
        badges = [f'<span class="badge" style="background:{colour}">{result.status.value}</span>',
                  f'<span class="badge muted">{html.escape(step.agent_role)}</span>']
        if step.requires_tool:
            label = html.escape(result.tool_used or step.requires_tool)
            if result.used_fallback:
                badges.append(f'<span class="badge warn">fallback -> {label}</span>')
            else:
                badges.append(f'<span class="badge muted">{label}</span>')
        body = html.escape(result.error or result.output or "(no output)")
        cards.append(f"""
      <details class="card"{' open' if result.status.value == 'failed' else ''}>
        <summary>
          <span class="card-id">{html.escape(step_id)}</span>
          {''.join(badges)}
          <span class="card-meta">{result.duration_s:.2f}s &middot;
            {result.usage.total_tokens} tok &middot;
            deps: {html.escape(', '.join(step.depends_on) or '-')}</span>
        </summary>
        <p class="desc">{html.escape(step.description)}</p>
        <pre>{body}</pre>
      </details>""")

    runs: List[str] = []
    for report in metrics["runs"]:
        runs.append(
            f"<tr><td>{html.escape(report['reason'])}</td>"
            f"<td>{report['steps_executed']}</td><td>{report['steps_reused']}</td>"
            f"<td>{report['reuse_ratio'] * 100:.0f}%</td>"
            f"<td>{report['duration_s']:.2f}s</td>"
            f"<td>{report['usage']['total_tokens']:,}</td></tr>"
        )

    tool_rows = "".join(
        f"<tr><td>{html.escape(t['name'])}</td><td>{html.escape(t['capability'])}</td>"
        f"<td>{'live' if t['live'] else 'simulated'}</td>"
        f"<td>{'BROKEN' if t['broken'] else 'ok'}</td>"
        f"<td>{html.escape(', '.join(t['fallbacks']) or '-')}</td>"
        f"<td>{t['calls']}</td></tr>"
        for t in workflow.tools.describe()
    )

    generated = time.strftime("%Y-%m-%d %H:%M:%S")
    document = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Orchestrator run {html.escape(workflow.run_id)}</title>
<style>
  :root {{ color-scheme: dark; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; padding:2rem; background:#0f1115; color:#e8ecf4;
         font-family: system-ui, -apple-system, "Segoe UI", sans-serif; line-height:1.5; }}
  h1 {{ font-size:1.4rem; margin:0 0 .25rem; }}
  h2 {{ font-size:1.05rem; margin:2.5rem 0 .75rem; color:#aab3c5;
        text-transform:uppercase; letter-spacing:.08em; }}
  .sub {{ color:#8b93a7; font-size:.9rem; margin:0 0 1.5rem; }}
  .tiles {{ display:grid; gap:.75rem; grid-template-columns:repeat(auto-fit,minmax(130px,1fr)); }}
  .tile {{ background:#161a23; border:1px solid #242a36; border-radius:10px; padding:.85rem 1rem; }}
  .tile-label {{ font-size:.72rem; text-transform:uppercase; letter-spacing:.07em; color:#8b93a7; }}
  .tile-value {{ font-size:1.35rem; font-weight:600; margin-top:.2rem; }}
  .graph {{ background:#12151d; border:1px solid #242a36; border-radius:10px;
            padding:1rem; overflow-x:auto; }}
  .card {{ background:#161a23; border:1px solid #242a36; border-radius:10px;
           padding:.75rem 1rem; margin-bottom:.6rem; }}
  .card summary {{ cursor:pointer; display:flex; gap:.5rem; align-items:center; flex-wrap:wrap; }}
  .card-id {{ font-family:ui-monospace,monospace; font-weight:600; }}
  .card-meta {{ color:#8b93a7; font-size:.8rem; margin-left:auto; }}
  .desc {{ color:#aab3c5; font-size:.88rem; margin:.6rem 0 .4rem; }}
  .badge {{ font-size:.7rem; padding:.15rem .5rem; border-radius:999px; color:#0f1115;
            font-weight:700; text-transform:uppercase; letter-spacing:.04em; }}
  .badge.muted {{ background:#2b3242; color:#c3cad8; font-weight:600; }}
  .badge.warn {{ background:#f39c12; }}
  pre {{ background:#0c0e13; border:1px solid #242a36; border-radius:8px; padding:.8rem;
         overflow-x:auto; white-space:pre-wrap; word-break:break-word; font-size:.82rem;
         margin:0; max-height:420px; }}
  table {{ border-collapse:collapse; width:100%; font-size:.86rem; }}
  th, td {{ border-bottom:1px solid #242a36; padding:.5rem .7rem; text-align:left; }}
  th {{ color:#8b93a7; font-weight:600; text-transform:uppercase; font-size:.72rem;
        letter-spacing:.06em; }}
  footer {{ margin-top:3rem; color:#5d6470; font-size:.78rem; }}
</style></head>
<body>
  <h1>Adaptive AI Task Orchestrator</h1>
  <p class="sub">run <code>{html.escape(workflow.run_id)}</code>
     &middot; {html.escape(workflow.description or 'no description')}
     &middot; generated {generated}</p>

  <div class="tiles">{tile_html}</div>

  <h2>Dependency graph</h2>
  <div class="graph">{_svg_dag(workflow)}</div>

  <h2>Steps</h2>
  {''.join(cards)}

  <h2>Run history</h2>
  <table>
    <tr><th>Trigger</th><th>Executed</th><th>Reused</th><th>Reuse</th><th>Wall time</th><th>Tokens</th></tr>
    {''.join(runs) or '<tr><td colspan="6">no runs yet</td></tr>'}
  </table>

  <h2>Tools</h2>
  <table>
    <tr><th>Tool</th><th>Capability</th><th>Mode</th><th>Health</th><th>Fallback chain</th><th>Calls</th></tr>
    {tool_rows}
  </table>

  <footer>Static export. For the live view run
    <code>python -m orchestrator.cli serve</code>.</footer>
</body></html>"""
    return _write(path, document)
