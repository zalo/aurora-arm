#!/usr/bin/env python3
"""Chart for the CPU vertex decoding PR: FIFO translation worker and frame time per commit,
Melee on the Miyoo Flip V2 (RK3566 / Mali-G52). Usage: plot_cpu_vertex_decode.py <png>"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
FIFO, FRAME = "#2a78d6", "#eb6834"
fig, (ax, bx) = plt.subplots(1, 2, figsize=(14, 6.2), dpi=130, facecolor=SURF, gridspec_kw={"width_ratios": [1.25, 1]})

# Frozen Onett: FIFO translation worker and presented frame, ms per frame.
onett = [("Generic per-vertex\nCPU decoder (before)", 28.6, 30.7),
         ("Specialized vertex\nloaders (commit 1)", 24.4, 27.2),
         ("+ Resident display-list\ngeometry (commit 3)\nuploads 3.8 -> 0.3 MB/frame", 16.4, 25.7)]
xs = range(len(onett)); w = 0.36
for i, (label, fifo, frame) in enumerate(onett):
    ax.bar(i - w / 2, fifo, w, color=FIFO, zorder=3)
    ax.bar(i + w / 2, frame, w, color=FRAME, zorder=3)
    ax.text(i - w / 2, fifo + 0.5, f"{fifo:g}", ha="center", va="bottom", fontsize=10, color=INK)
    ax.text(i + w / 2, frame + 0.5, f"{frame:g}", ha="center", va="bottom", fontsize=10, color=INK)
ax.set_xticks(list(xs)); ax.set_xticklabels([o[0] for o in onett], color=INK2, fontsize=9)
ax.set_title("Frozen Onett stage (447 display-list calls per frame)", loc="left", color=INK, fontsize=11)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=FIFO, label="FIFO translation worker"),
                   plt.Rectangle((0, 0), 1, 1, color=FRAME, label="Presented frame")],
          loc="upper right", frameon=False, fontsize=9, labelcolor=INK2)
ax.set_ylim(0, 36)

# Fountain of Dreams: ~1850 GX_POINTS draws, ~25k point sprites per frame; FIFO worker only.
fountain = [("Generic decoder,\nquad expansion (before)", 46, "46"),
            ("Specialized loaders,\nline/point expansion (commit 1)", 23, "23"),
            ("Instanced point\nsprites (commit 4)", 21, "21*")]
for i, (label, v, txt) in enumerate(fountain):
    bx.bar(i, v, 0.5, color=FIFO, zorder=3)
    bx.text(i, v + 0.7, txt, ha="center", va="bottom", fontsize=10, color=INK)
bx.set_xticks(range(len(fountain))); bx.set_xticklabels([f[0] for f in fountain], color=INK2, fontsize=9)
bx.set_title("Fountain of Dreams (~25k point sprites per frame)", loc="left", color=INK, fontsize=11)
bx.set_ylim(0, 54)
bx.text(0.0, -0.3, "* measured against a 25 ms FIFO worker at that stage of the port (-4 ms);\n"
        "  the quad expansion wrote ~3 MB of corner records per frame.", transform=bx.transAxes,
        fontsize=8, color=INK2, va="top")

for a in (ax, bx):
    a.set_facecolor(SURF); a.grid(axis="y", color=GRID, linewidth=1); a.set_axisbelow(True)
    for s in ("top", "right"): a.spines[s].set_visible(False)
    for s in ("left", "bottom"): a.spines[s].set_color(GRID)
    a.tick_params(colors=INK2, length=0)
    a.set_ylabel("ms per frame", color=INK2)
fig.suptitle("CPU vertex decoding in Aurora: Melee on a Miyoo Flip V2 (RK3566, Mali-G52 GLES)", x=0.01, ha="left",
             color=INK, fontsize=13, y=0.985)
fig.text(0.01, 0.01, "Mean per-frame times from the port's instrumented builds (melee-native native/FLIP_PERFORMANCE_WINS.md). "
         "The storage-buffer vertex path cannot run on this GPU: the driver exposes zero vertex-stage storage blocks.",
         color=INK2, fontsize=8)
fig.tight_layout(rect=(0, 0.05, 1, 0.95))
fig.savefig(sys.argv[1], facecolor=SURF)
print("wrote", sys.argv[1])
