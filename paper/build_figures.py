"""Generate the paper's figures as PNGs sized for an IEEE column.

Figures are drawn at single-column width (3.3 in) so they sit inside the
two-column body without overflowing. Fig. 3 and Fig. 4 plot the exact numbers
in benchmarks/results/report.md -- no illustrative data.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path("figures")
OUT.mkdir(exist_ok=True)

COL_W = 3.3          # IEEE single-column width, inches
DPI = 400

NAVY = "#2C2C2C"     # Dark gray
TEAL = "#555555"     # Medium gray
ORANGE = "#888888"   # Light gray
GREEN = "#2C2C2C"    # Dark gray (emphasis)
PURPLE = "#666666"   # Medium-dark gray
GREY = "#D3D3D3"     # Light gray
DARK = "#000000"     # Black

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 6.5,
})


def box(ax, x, y, w, h, text, fc, tc="white", fs=6.0, weight="bold", r=0.012):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=%f" % r,
        linewidth=0.5, edgecolor=fc, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            color=tc, fontsize=fs, fontweight=weight, zorder=3, linespacing=1.25)


def arrow(ax, p1, p2, colour=NAVY, lw=0.6, style="-|>"):
    ax.add_patch(FancyArrowPatch(
        p1, p2, arrowstyle=style, mutation_scale=5,
        linewidth=lw, color=colour, shrinkA=0, shrinkB=0, zorder=1))


def canvas(h):
    fig, ax = plt.subplots(figsize=(COL_W, h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.02,
                facecolor="white")
    plt.close(fig)
    print("  %s" % path)


# ---------------------------------------------------------------------------
# Fig. 1 -- layered system architecture
# ---------------------------------------------------------------------------

def fig1():
    fig, ax = canvas(2.75)
    H = 0.075

    box(ax, 0.36, 0.905, 0.28, H, "User", NAVY)
    box(ax, 0.13, 0.795, 0.74, H, "Adaptive AI Orchestrator", ORANGE)
    arrow(ax, (0.50, 0.905), (0.50, 0.870))

    # coordination layer
    for i, name in enumerate(["Task\nPlanner", "Workflow\nEngine", "Memory\nManager"]):
        x = 0.045 + i * 0.325
        box(ax, x, 0.645, 0.285, H + 0.015, name, TEAL)
        arrow(ax, (0.50, 0.795), (x + 0.1425, 0.735))

    # execution layer
    box(ax, 0.045, 0.505, 0.42, H, "Agent Manager", GREEN)
    box(ax, 0.535, 0.505, 0.42, H, "Tool Manager", PURPLE)
    arrow(ax, (0.475, 0.645), (0.255, 0.580), GREEN)
    arrow(ax, (0.525, 0.645), (0.745, 0.580), PURPLE)

    # leaves
    for i, name in enumerate(["Planner-created\nspecialists", "Role prompts +\nshared context"]):
        box(ax, 0.045 + i * 0.215, 0.355, 0.195, 0.10, name, GREY, DARK, 5.2, "normal")
    for i, name in enumerate(["MCP\nservers", "REST /\nAPI", "CLI &\nlocal"]):
        box(ax, 0.535 + i * 0.145, 0.355, 0.125, 0.10, name, GREY, DARK, 5.2, "normal")

    # cross-cutting services
    box(ax, 0.045, 0.215, 0.91, H, "Persistence (SQLite)  •  Event Bus  •  Requirements\n"
        "Checker  •  Approval Gate  •  Dashboard", NAVY, fs=5.4)

    ax.text(0.5, 0.135, "Agents and tools are created and selected on demand;\n"
                        "no role or tool binding is fixed at design time.",
            ha="center", va="center", fontsize=5.4, style="italic", color=DARK)
    save(fig, "fig1_architecture.png")


# ---------------------------------------------------------------------------
# Fig. 2 -- selective re-execution on the worked example
# ---------------------------------------------------------------------------

def fig2():
    fig, ax = canvas(1.95)
    W, H = 0.235, 0.115

    ax.text(0.02, 0.95, "Requirement change: MongoDB $\\rightarrow$ PostgreSQL",
            fontsize=6.0, fontweight="bold", color=DARK, va="center")

    box(ax, 0.05, 0.66, 0.26, H, "Requirements", GREEN)

    mids = [("Frontend", GREEN), ("Backend", GREEN), ("Database", ORANGE)]
    for i, (name, c) in enumerate(mids):
        x = 0.03 + i * 0.325
        box(ax, x, 0.40, W, H, name, c)
        arrow(ax, (0.18, 0.66), (x + W / 2, 0.515), "#8A94A0", 0.55)

    box(ax, 0.355, 0.14, W, H, "Integration testing", ORANGE, fs=5.4)
    for i in range(3):
        arrow(ax, (0.03 + i * 0.325 + W / 2, 0.40), (0.4725, 0.255), "#8A94A0", 0.55)

    # legend
    ax.add_patch(FancyBboxPatch((0.05, 0.005), 0.035, 0.045,
                 boxstyle="round,pad=0,rounding_size=0.008",
                 linewidth=0, facecolor=GREEN))
    ax.text(0.10, 0.028, "reused (0 tokens)", fontsize=5.4, va="center", color=DARK)
    ax.add_patch(FancyBboxPatch((0.50, 0.005), 0.035, 0.045,
                 boxstyle="round,pad=0,rounding_size=0.008",
                 linewidth=0, facecolor=ORANGE))
    ax.text(0.55, 0.028, "re-executed", fontsize=5.4, va="center", color=DARK)
    save(fig, "fig2_selective.png")


# ---------------------------------------------------------------------------
# Fig. 3 -- signature propagation and termination
# ---------------------------------------------------------------------------

def fig3():
    fig, ax = canvas(1.55)
    W, H = 0.175, 0.135
    y = 0.52
    xs = [0.045, 0.29, 0.535, 0.78]
    names = ["A", "B", "C", "D"]
    cols = [ORANGE, ORANGE, ORANGE, GREEN]

    ax.text(0.02, 0.93, "A changes; C re-runs but reproduces its previous output",
            fontsize=5.8, fontweight="bold", color=DARK, va="center")

    for i, (x, nm, c) in enumerate(zip(xs, names, cols)):
        box(ax, x, y, W, H, nm, c, fs=7)
        if i:
            arrow(ax, (xs[i - 1] + W, y + H / 2), (x, y + H / 2), "#8A94A0", 0.6)

    labels = ["Sig changed\n(edited)", "Sig changed\n(A differs)",
              "Sig changed\n(B differs)", "Sig unchanged\n(C identical)"]
    for x, lab, c in zip(xs, labels, cols):
        ax.text(x + W / 2, y - 0.10, lab, ha="center", va="top",
                fontsize=5.0, color=c if c == GREEN else DARK)

    ax.plot([0.755, 0.755], [0.30, 0.78], linestyle=(0, (2, 2)),
            linewidth=0.7, color=GREEN)
    ax.text(0.762, 0.20, "propagation stops here", fontsize=5.2,
            color=GREEN, style="italic", va="center", ha="right")
    save(fig, "fig3_propagation.png")


# ---------------------------------------------------------------------------
# Fig. 4 -- per-scenario step executions (measured)
# ---------------------------------------------------------------------------

def fig4():
    scen = ["full", "early", "mid", "leaf", "noop", "tool", "micro"]
    prop = [5, 10, 8, 6, 6, 7, 12]
    cone = [5, 10, 8, 6, 8, 7, 12]
    rest = [5, 10, 10, 10, 10, 10, 16]

    fig, ax = plt.subplots(figsize=(COL_W, 1.75))
    x = range(len(scen))
    w = 0.27
    ax.bar([i - w for i in x], prop, w, label="Proposed", color=GREEN)
    ax.bar(list(x), cone, w, label="Cone ablation", color=TEAL)
    ax.bar([i + w for i in x], rest, w, label="Restart / LangGraph", color=ORANGE)

    ax.set_ylabel("Step executions", fontsize=6)
    ax.set_xticks(list(x))
    ax.set_xticklabels(scen, fontsize=5.8)
    ax.tick_params(axis="y", labelsize=5.8)
    ax.legend(fontsize=5.2, frameon=False, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, 1.19), columnspacing=1.0, handlelength=1.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(0.5)
    ax.grid(axis="y", linewidth=0.3, alpha=0.35)
    ax.set_axisbelow(True)
    ax.set_yticks([0, 4, 8, 12, 16])
    ax.set_ylim(0, 19)
    # Only noop_rerun separates the ablation from the proposed rule; label it.
    ax.annotate("ablation diverges", xy=(4.02, 8.3), xytext=(4.1, 13.5),
                fontsize=4.8, color=DARK, ha="center",
                arrowprops=dict(arrowstyle="-|>", mutation_scale=4,
                                linewidth=0.5, color=DARK))
    fig.tight_layout()
    save(fig, "fig4_per_scenario.png")


# ---------------------------------------------------------------------------
# Fig. 5 -- aggregate totals (measured)
# ---------------------------------------------------------------------------

def fig5():
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 1.5))
    names = ["Prop.", "Cone", "Rest.", "LangG."]
    cols = [GREEN, TEAL, ORANGE, "#C0562A"]

    for ax, vals, title, fmt in [
        (axes[0], [54, 56, 71, 71], "Step executions", "%d"),
        (axes[1], [19196, 20016, 24502, 24502], "Tokens", "%s"),
    ]:
        bars = ax.bar(names, vals, color=cols, width=0.62)
        ax.set_title(title, fontsize=6, pad=3)
        ax.tick_params(labelsize=5.4)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_linewidth(0.5)
        ax.set_ylim(0, max(vals) * 1.22)
        for b, v in zip(bars, vals):
            lab = "{:,}".format(v) if fmt == "%s" else str(v)
            ax.text(b.get_x() + b.get_width() / 2, v * 1.03, lab,
                    ha="center", va="bottom", fontsize=4.9)
    fig.tight_layout()
    save(fig, "fig5_aggregate.png")


if __name__ == "__main__":
    print("figures:")
    fig1(); fig2(); fig3(); fig4(); fig5()
