"""Run the benchmark matrix and write JSON + Markdown + HTML reports.

    python -m benchmarks.run_benchmarks
    python -m orchestrator.cli bench --repeats 5

Everything runs against the deterministic stub LLM, so the step/token counts
are exact and reproducible on any machine. Only ``wall_s`` varies, and the
harness reports the median across repeats.
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .baselines import default_adapters
from .harness import Scenario, TrialResult, default_scenarios, run_matrix

REFERENCE_SYSTEM = "adaptive (this project)"


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _by_scenario(results: List[TrialResult]) -> Dict[str, List[TrialResult]]:
    grouped: Dict[str, List[TrialResult]] = {}
    for result in results:
        grouped.setdefault(result.scenario, []).append(result)
    return grouped


def _delta(ours: float, theirs: float) -> str:
    """How much less our system used, as a percentage of the baseline."""
    if theirs == 0:
        return "n/a"
    change = (theirs - ours) / theirs * 100
    if abs(change) < 0.5:
        return "same"
    return f"{change:+.0f}%"


def summarise(results: List[TrialResult], scenarios: List[Scenario]) -> Dict[str, Any]:
    descriptions = {s.name: s.description for s in scenarios}
    summary: Dict[str, Any] = {"scenarios": {}, "totals": {}}

    for scenario_name, group in _by_scenario(results).items():
        reference = next((r for r in group if r.system == REFERENCE_SYSTEM), None)
        rows = []
        for result in group:
            total = result.total
            row = {
                "system": result.system,
                "executions": total.executions,
                "llm_calls": total.llm_calls,
                "tokens": total.tokens,
                "wall_s": round(total.wall_s, 3),
                "cost_usd": round(total.cost_usd, 8),
                "notes": result.notes,
            }
            if reference and result.system != REFERENCE_SYSTEM:
                ref = reference.total
                row["vs_reference"] = {
                    "executions": _delta(ref.executions, total.executions),
                    "tokens": _delta(ref.tokens, total.tokens),
                    "wall_s": _delta(ref.wall_s, total.wall_s),
                }
            rows.append(row)
        summary["scenarios"][scenario_name] = {
            "description": descriptions.get(scenario_name, ""),
            "rows": rows,
        }

    # Aggregate across every scenario.
    totals: Dict[str, Dict[str, float]] = {}
    for result in results:
        bucket = totals.setdefault(result.system, {"executions": 0, "tokens": 0, "wall_s": 0.0})
        bucket["executions"] += result.total.executions
        bucket["tokens"] += result.total.tokens
        bucket["wall_s"] += result.total.wall_s

    reference_total = totals.get(REFERENCE_SYSTEM)
    for system, bucket in totals.items():
        entry = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in bucket.items()}
        if reference_total and system != REFERENCE_SYSTEM:
            entry["vs_reference"] = {
                "executions": _delta(reference_total["executions"], bucket["executions"]),
                "tokens": _delta(reference_total["tokens"], bucket["tokens"]),
                "wall_s": _delta(reference_total["wall_s"], bucket["wall_s"]),
            }
        summary["totals"][system] = entry

    return summary


def to_markdown(summary: Dict[str, Any], repeats: int, latency_s: float) -> str:
    lines = [
        "# Benchmark results",
        "",
        f"Deterministic stub LLM, {latency_s:.3f}s simulated latency per call, "
        f"median of {repeats} repeat(s). Step, call and token counts are exact and "
        "reproducible; wall time is machine-dependent.",
        "",
        "`vs_reference` is measured against **adaptive (this project)** — a negative "
        "percentage means that system used *more* than we did.",
        "",
    ]

    for name, block in summary["scenarios"].items():
        lines += [f"## `{name}`", "", block["description"], "",
                  "| System | Steps run | LLM calls | Tokens | Wall (s) | vs reference |",
                  "|---|---:|---:|---:|---:|---|"]
        for row in block["rows"]:
            versus = row.get("vs_reference")
            versus_text = (
                f"steps {versus['executions']}, tokens {versus['tokens']}, time {versus['wall_s']}"
                if versus else "(reference)"
            )
            lines.append(
                f"| {row['system']} | {row['executions']} | {row['llm_calls']} | "
                f"{row['tokens']:,} | {row['wall_s']:.2f} | {versus_text} |"
            )
        lines.append("")

    lines += ["## Totals across all scenarios", "",
              "| System | Steps run | Tokens | Wall (s) | vs reference |",
              "|---|---:|---:|---:|---|"]
    for system, entry in summary["totals"].items():
        versus = entry.get("vs_reference")
        versus_text = (
            f"steps {versus['executions']}, tokens {versus['tokens']}, time {versus['wall_s']}"
            if versus else "(reference)"
        )
        lines.append(
            f"| {system} | {entry['executions']} | {entry['tokens']:,} | "
            f"{entry['wall_s']:.2f} | {versus_text} |"
        )
    lines.append("")
    return "\n".join(lines)


def to_html(summary: Dict[str, Any], repeats: int, latency_s: float) -> str:
    def table(rows: List[Dict[str, Any]], columns: List[tuple]) -> str:
        head = "".join(f"<th>{html.escape(label)}</th>" for label, _ in columns)
        body = []
        for row in rows:
            highlight = ' class="ref"' if row["system"] == REFERENCE_SYSTEM else ""
            cells = "".join(f"<td>{html.escape(str(fn(row)))}</td>" for _, fn in columns)
            body.append(f"<tr{highlight}>{cells}</tr>")
        return f"<table><tr>{head}</tr>{''.join(body)}</table>"

    def versus(row: Dict[str, Any]) -> str:
        v = row.get("vs_reference")
        if not v:
            return "reference"
        return f"steps {v['executions']} · tokens {v['tokens']} · time {v['wall_s']}"

    columns = [
        ("System", lambda r: r["system"]),
        ("Steps run", lambda r: r["executions"]),
        ("LLM calls", lambda r: r["llm_calls"]),
        ("Tokens", lambda r: f"{r['tokens']:,}"),
        ("Wall (s)", lambda r: f"{r['wall_s']:.2f}"),
        ("vs reference", versus),
    ]

    sections = "".join(
        f"<h2><code>{html.escape(name)}</code></h2><p class='desc'>"
        f"{html.escape(block['description'])}</p>{table(block['rows'], columns)}"
        for name, block in summary["scenarios"].items()
    )

    total_rows = [{"system": s, **e} for s, e in summary["totals"].items()]
    total_columns = [
        ("System", lambda r: r["system"]),
        ("Steps run", lambda r: r["executions"]),
        ("Tokens", lambda r: f"{r['tokens']:,}"),
        ("Wall (s)", lambda r: f"{r['wall_s']:.2f}"),
        ("vs reference", versus),
    ]

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Orchestrator benchmarks</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ margin:0; padding:2rem; background:#0f1115; color:#e8ecf4; line-height:1.55;
          font-family:system-ui,-apple-system,"Segoe UI",sans-serif; }}
  h1 {{ font-size:1.4rem; margin:0 0 .3rem; }}
  h2 {{ font-size:1rem; margin:2.2rem 0 .3rem; color:#aab3c5; }}
  .sub, .desc {{ color:#8b93a7; font-size:.87rem; }}
  .sub {{ max-width:70ch; margin:0 0 1rem; }}
  .desc {{ margin:.15rem 0 .6rem; }}
  table {{ border-collapse:collapse; width:100%; font-size:.86rem;
           background:#161a23; border:1px solid #242a36; border-radius:8px; overflow:hidden; }}
  th, td {{ padding:.5rem .7rem; border-bottom:1px solid #242a36; text-align:left; }}
  th {{ color:#8b93a7; font-size:.68rem; text-transform:uppercase; letter-spacing:.06em;
        background:#12151d; }}
  td:not(:first-child) {{ text-align:right; font-variant-numeric:tabular-nums; }}
  td:last-child {{ text-align:left; color:#8b93a7; font-size:.8rem; }}
  tr.ref {{ background:#182437; }}
  tr.ref td:first-child {{ font-weight:650; color:#6ea8fe; }}
  code {{ background:#1b2029; padding:.1rem .35rem; border-radius:4px; }}
</style></head>
<body>
<h1>Adaptive orchestration benchmarks</h1>
<p class="sub">Deterministic stub LLM, {latency_s:.3f}s simulated latency per call,
median of {repeats} repeat(s). Every system executes identical agent work, so
differences come from orchestration strategy alone. Step, call and token counts
are exact and reproducible; wall time is machine-dependent. Percentages compare
each system against <b>adaptive (this project)</b> — negative means that system
used more than we did.</p>
{sections}
<h2>Totals across all scenarios</h2>
{table(total_rows, total_columns)}
</body></html>"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(repeats: int = 3, scenarios: Optional[List[str]] = None,
         output: str = "benchmarks/results", latency_s: float = 0.05) -> int:
    selected = default_scenarios()
    if scenarios:
        wanted = set(scenarios)
        selected = [s for s in selected if s.name in wanted]
        if not selected:
            print(f"no scenarios matched {sorted(wanted)}", file=sys.stderr)
            return 2

    print(f"Running {len(selected)} scenario(s) x {len(default_adapters(latency_s))} systems "
          f"x {repeats} repeat(s)...")
    results = run_matrix(
        adapters=default_adapters(latency_s),
        scenarios=selected,
        repeats=repeats,
        on_progress=lambda label: print(f"  . {label}", flush=True),
    )

    summary = summarise(results, selected)
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "results.json").write_text(
        json.dumps({"repeats": repeats, "latency_s": latency_s,
                    "raw": [r.to_dict() for r in results], "summary": summary},
                   indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(to_markdown(summary, repeats, latency_s), encoding="utf-8")
    (out_dir / "report.html").write_text(to_html(summary, repeats, latency_s), encoding="utf-8")

    print("\n" + to_markdown(summary, repeats, latency_s))
    print(f"written: {out_dir / 'results.json'}, {out_dir / 'report.md'}, "
          f"{out_dir / 'report.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
