#!/usr/bin/env python3
"""Plot the frozen-Onett frame-time trajectory of the Flip port from the
onett-frame-time.csv next to this script's output. Usage:
plot_onett_frame_time.py <csv> <png>"""
import csv, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open(sys.argv[1])))
x = [int(r["step"]) for r in rows]
y = [float(r["frame_ms"]) for r in rows]
INK, INK2, GRID, SURF, SERIES = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb", "#2a78d6"
fig, ax = plt.subplots(figsize=(15, 8.5), dpi=130, facecolor=SURF)
ax.set_facecolor(SURF)
ax.set_yscale("log")
PHASE = {"aurora": ("#2a78d6", "In the Aurora PR set"), "dawn-hatch": ("#eb6834", "Needs the Dawn escape hatch / direct GLES"),
         "mixed": ("#eb6834", None), "platform": ("#1baf7a", "Platform: driver, cores, governors")}
ax.plot(x, y, color="#a9a8a1", linewidth=2, solid_joinstyle="round", solid_capstyle="round", zorder=3)
for r, xi, yi in zip(rows, x, y):
    ax.scatter([xi], [yi], s=72, color=PHASE[r["phase"]][0], edgecolors=SURF, linewidths=2, zorder=4)
handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=8, color=c, label=l) for c, l in PHASE.values() if l]
ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=9, labelcolor=INK2, bbox_to_anchor=(1.0, 0.86))
for r, xi, yi in zip(rows, x, y):
    ax.annotate(f'{r["version"]}: {r["label"]}', (xi, yi), xytext=(0, 12), textcoords="offset points",
                rotation=62, ha="left", va="bottom", fontsize=8.2, color=INK2, rotation_mode="anchor")
for xi, yi in ((x[0], y[0]), (x[-1], y[-1])):
    ax.annotate(f"{yi:g} ms", (xi, yi), xytext=(0, -16), textcoords="offset points", ha="center", va="top",
                fontsize=10, color=INK, fontweight="bold")
ax.axhline(16.7, color=GRID, linewidth=1, zorder=1)
ax.text(x[-1] + 0.15, 16.7, "60 Hz (16.7 ms)", color=INK2, fontsize=8.5, va="center")
ax.set_yticks([300, 200, 100, 50, 30, 20, 16.7])
ax.set_yticklabels(["300", "200", "100", "50", "30", "20", "16.7"], color=INK2)
ax.set_xticks(x); ax.set_xticklabels([r["version"] for r in rows], rotation=45, ha="right", color=INK2, fontsize=8)
ax.grid(axis="y", color=GRID, linewidth=1); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
ax.tick_params(colors=INK2, length=0)
ax.set_ylabel("Frame time, ms (log scale)", color=INK2)
ax.set_xlabel("Build", color=INK2)
ax.set_ylim(13, 900); ax.set_xlim(0.5, len(x) + 1.2)
from matplotlib.ticker import NullFormatter, NullLocator
ax.yaxis.set_minor_formatter(NullFormatter()); ax.yaxis.set_minor_locator(NullLocator())
fig.suptitle("Melee native port on the Miyoo Flip V2: frozen Onett frame time per optimization",
             x=0.01, ha="left", color=INK, fontsize=13, y=0.985)
fig.text(0.01, 0.01, "Mean presented frame time on the frozen Onett reference scene (RK3566, Mali-G52). "
         "Points before v79 mix measurement setups; see native/FLIP_PERFORMANCE_WINS.md for sources.",
         color=INK2, fontsize=8)
fig.tight_layout(rect=(0, 0.03, 1, 0.965))
fig.savefig(sys.argv[2], facecolor=SURF)
print("wrote", sys.argv[2])
