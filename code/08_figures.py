"""
Figures 2 and 3.

Figure 2 (construction flow) is a four-panel diagram that summarises how
the supplier-base digital disparity measure is built. It is generated
schematically, not from the data.

Figure 3 (network heterogeneity) plots the supplier networks of four
representative focal firms in 2024, with each supplier coloured by its
patent-based digital intensity. The intent is to make visible the
within-network distribution of digital capability that the disparity
measure is designed to capture.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

from utils import DATA_DIR, ensure_output_dir

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9,
    "savefig.dpi": 300,
    "mathtext.fontset": "stix",
})

ACCENT = "#1f3a5e"
LIGHT = "#e8eef5"
GREY = "#bfc6cf"
TEXT = "#222222"
ARROW = "#888888"


# ============================================================
# Figure 2: schematic construction flow (2x2 layout)
# ============================================================
def build_figure2(save_to: Path) -> None:
    """Schematic explainer; no data input. Mirrors Figure 2 in the paper."""
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.4))

    def panel(ax, title):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        ax.text(0.5, 1.02, title, fontsize=10.5, fontweight="bold",
                color="black", ha="center", va="bottom")

    # --- Step 1: a focal firm with first- and second-order suppliers ---
    ax = axes[0, 0]; panel(ax, "Step 1: Network construction")
    cx, cy = 0.50, 0.62
    ax.scatter([cx], [cy], s=1100, c="black", zorder=5)
    ax.text(cx, cy, "Focal\nfirm", color="white", fontsize=8.5,
            ha="center", va="center", linespacing=1.0, zorder=6)
    n1 = 4; r1 = 0.22; first = []
    for i in range(n1):
        a = np.pi/4 + 2*np.pi*i/n1
        x = cx + r1 * np.cos(a); y = cy + r1 * np.sin(a) * 0.95
        first.append((x, y, a))
        ax.plot([cx, x], [cy, y], color=ACCENT, linewidth=0.9, zorder=2)
        ax.scatter([x], [y], s=420, c="#7d8a99", edgecolors=ACCENT, linewidths=0.7, zorder=4)
    for x, y, a in first:
        for d in (-0.45, +0.45):
            a2 = a + d
            sx = x + 0.16 * np.cos(a2); sy = y + 0.16 * np.sin(a2) * 0.95
            ax.plot([x, sx], [y, sy], color=ARROW, linewidth=0.7,
                    linestyle=(0, (3, 2.2)), zorder=1)
            ax.scatter([sx], [sy], s=160, facecolors="white",
                       edgecolors="#444", linewidths=0.7, zorder=3)
    ly1, ly2 = 0.12, 0.04
    ax.scatter([0.04], [ly1], s=70, c="black");                                            ax.text(0.08, ly1, "Focal firm", fontsize=8.2, va="center")
    ax.scatter([0.34], [ly1], s=70, c="#7d8a99", edgecolors=ACCENT, linewidths=0.6);       ax.text(0.38, ly1, "First-order suppliers", fontsize=8.2, va="center")
    ax.scatter([0.04], [ly2], s=55, facecolors="white", edgecolors="#444", linewidths=0.7);ax.text(0.08, ly2, "Second-order suppliers", fontsize=8.2, va="center")

    # --- Step 2: node-level digitalisation via IPC categories + recruitment ---
    ax = axes[0, 1]; panel(ax, "Step 2: Node-level digitalization")
    ax.text(0.50, 0.93, "Digital IPC categories", fontsize=9.5, fontweight="bold",
            color=TEXT, ha="center", va="top")
    ax.plot([0.05, 0.95], [0.875, 0.875], color=GREY, linewidth=0.7)
    ipc = [("G06F", "Electric digital data processing"),
           ("H04L", "Digital information transmission"),
           ("G06Q", "Data processing systems"),
           ("G06N", "Computational AI models"),
           ("G06K", "Data recognition and representation")]
    y0, dy = 0.81, 0.066
    for i, (code, desc) in enumerate(ipc):
        yy = y0 - i*dy
        ax.text(0.07, yy, code, fontsize=9, fontweight="bold", color=TEXT, va="center")
        ax.text(0.27, yy, desc, fontsize=9, color=TEXT, va="center")
    ax.plot([0.05, 0.95], [0.460, 0.460], color=GREY, linewidth=0.7)
    ax.text(0.50, 0.42, "number of digital patents", fontsize=9, color=TEXT,
            ha="center", va="top", style="italic")
    ax.plot([0.20, 0.80], [0.345, 0.345], color=TEXT, linewidth=0.8)
    ax.text(0.50, 0.34, "total number of patents",  fontsize=9, color=TEXT,
            ha="center", va="top", style="italic")
    ax.text(0.50, 0.20, "Digital intensity", fontsize=9.5, fontweight="bold",
            color=TEXT, ha="center", va="top")
    ax.text(0.50, 0.11, r"$d_j$  for each supplier node",
            fontsize=9.5, color=TEXT, ha="center", va="top")

    # --- Step 3: imputation for unlisted nodes ---
    ax = axes[1, 0]; panel(ax, "Step 3: Imputation for unlisted nodes")
    ax.scatter([0.06], [0.86], s=110, c="black", zorder=3)
    ax.text(0.12, 0.86, "Listed supplier nodes", fontsize=9.5, fontweight="bold",
            color=TEXT, va="center")
    ax.text(0.97, 0.86, "3,174", fontsize=9.5, color=TEXT, ha="right", va="center")
    ax.plot([0.05, 0.95], [0.795, 0.795], color=GREY, linewidth=0.6)
    ax.text(0.12, 0.71,
            r"Direct measurement of $d_j$ from"+"\npatent and recruitment records",
            fontsize=9, color=TEXT, va="top", linespacing=1.35, style="italic")
    ax.scatter([0.06], [0.46], s=110, facecolors="white", edgecolors="black", linewidths=0.9, zorder=3)
    ax.text(0.12, 0.46, "Unlisted supplier nodes", fontsize=9.5, fontweight="bold",
            color=TEXT, va="center")
    ax.text(0.97, 0.46, r"~103,030", fontsize=9.5, color=TEXT, ha="right", va="center")
    ax.plot([0.05, 0.95], [0.395, 0.395], color=GREY, linewidth=0.6)
    ax.text(0.12, 0.31, "Industry-year mean digital ratio\nimputed from listed firms",
            fontsize=9, color=TEXT, va="top", linespacing=1.35, style="italic")
    ax.text(0.50, 0.10, "Imputation compresses heterogeneity",
            fontsize=9, fontweight="bold", color=TEXT, ha="center", va="center")
    ax.text(0.50, 0.03, "disparity estimates are conservative",
            fontsize=8.8, color=TEXT, ha="center", va="center", style="italic")

    # --- Step 4: range over the supplier base ---
    ax = axes[1, 1]; panel(ax, "Step 4: Disparity calculation")
    ax.text(0.50, 0.93, r"Distribution of $d_j$ across supplier base",
            fontsize=9.5, fontweight="bold", color=TEXT, ha="center", va="top")
    y_axis = 0.70
    ax.plot([0.08, 0.92], [y_axis, y_axis], color="#555555", linewidth=0.8, solid_capstyle="round")
    for xt in (0.08, 0.92):
        ax.plot([xt, xt], [y_axis-0.012, y_axis+0.012], color="#555555", linewidth=0.8)
    node_xs = [0.08, 0.18, 0.27, 0.40, 0.51, 0.63, 0.78, 0.92]
    for x in node_xs:
        ax.scatter([x], [y_axis], s=85, c="#222222", edgecolors="black", linewidths=0.5, zorder=3)
    for x, lab in [(node_xs[0], r"min $d_j$"), (node_xs[-1], r"max $d_j$")]:
        ax.plot([x, x], [y_axis+0.020, y_axis+0.080], color="#555555", linewidth=0.7)
        ax.text(x, y_axis+0.090, lab, fontsize=9, fontweight="bold", color=TEXT, ha="center", va="bottom")
    y_brk = y_axis - 0.13
    ax.plot([node_xs[0], node_xs[-1]], [y_brk, y_brk], color=TEXT, linewidth=0.9)
    for x in (node_xs[0], node_xs[-1]):
        ax.plot([x, x], [y_brk, y_brk+0.030], color=TEXT, linewidth=0.9)
    ax.text(0.50, y_brk-0.045, "Range = disparity", fontsize=9.5,
            fontweight="bold", color=TEXT, ha="center", va="top")
    ax.text(0.50, 0.28,
            r"$\mathrm{DD}_{it} = \mathrm{max}(d_{jt}) - \mathrm{min}(d_{jt})$",
            fontsize=11, color=TEXT, ha="center", va="center")
    ax.text(0.50, 0.18,
            r"over $j$ in supplier set $S_{it}$ of firm $i$, year $t$",
            fontsize=8.8, color=TEXT, ha="center", va="center", style="italic")
    ax.text(0.50, 0.06, "Firm-year supplier-base\ndigital disparity index",
            fontsize=9.5, fontweight="bold", color=TEXT, ha="center", va="center", linespacing=1.25)

    plt.subplots_adjust(left=0.03, right=0.985, top=0.93, bottom=0.025,
                        wspace=0.10, hspace=0.32)
    plt.savefig(save_to, bbox_inches="tight")
    plt.close()


# ============================================================
# Figure 3: within-network heterogeneity for four focal firms
# ============================================================
def build_figure3(save_to: Path) -> None:
    net = pd.read_csv(DATA_DIR / "supplier_network_examples.csv",
                      dtype={"focal_stkcd": str, "supplier_stkcd": str})

    # Four representative focal firms with industry labels for the captions
    examples = [
        ("002179", "Aviation Optronics",  "Electronics"),
        ("601016", "Energy Wind Power",   "Power generation"),
        ("000338", "Weichai Power",       "Automotive"),
        ("000858", "Wuliangye",           "Beverage"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(6.5, 6.0))
    axes = axes.flatten()
    cmap = plt.get_cmap("RdYlBu_r")
    norm = mcolors.Normalize(vmin=0, vmax=1)

    for ax, (code, firm, ind) in zip(axes, examples):
        sub = net[net["focal_stkcd"] == code].copy()
        # Restrict to listed suppliers with measurable digital ratios
        listed = sub[sub["patent_digital_ratio"].notna()]\
                   .drop_duplicates("supplier_stkcd")\
                   .sort_values("patent_digital_ratio")\
                   .reset_index(drop=True)
        n = len(listed)
        # Place suppliers at equal angles on a unit circle around the focal firm
        radius = 0.95
        for i in range(n):
            angle = 2 * np.pi * i / max(n, 1) - np.pi / 2
            x = radius * np.cos(angle); y = radius * np.sin(angle)
            ax.plot([0, x], [0, y], color="#cccccc", linewidth=0.7, zorder=1)
        ax.scatter(0, 0, s=1400, c="#e8eef5", marker="s",
                   edgecolors="#1f3a5e", linewidths=1.0, zorder=4)
        ax.text(0, 0, code, color="#1f3a5e", fontsize=8.5,
                ha="center", va="center", zorder=5)
        for i, row in listed.iterrows():
            angle = 2 * np.pi * i / max(n, 1) - np.pi / 2
            x = radius * np.cos(angle); y = radius * np.sin(angle)
            d = float(row["patent_digital_ratio"])
            color = cmap(norm(d))
            ax.scatter(x, y, s=720, c=[color], marker="o",
                       edgecolors="#222222", linewidths=0.7, zorder=3)
            ax.text(x, y, f"{d:.2f}", color="black", fontsize=8.5,
                    ha="center", va="center", zorder=5)
        ax.set_xlim(-1.55, 1.55); ax.set_ylim(-1.95, 1.55)
        ax.set_aspect("equal"); ax.axis("off")
        d_min = listed["patent_digital_ratio"].min()
        d_max = listed["patent_digital_ratio"].max()
        d_range = d_max - d_min
        ax.text(0, -1.55, f"{firm} ({code})\n{ind}, {n} suppliers, range = {d_range:.2f}",
                ha="center", va="top", fontsize=9.5)

    # Color bar
    cax = fig.add_axes([0.18, 0.045, 0.66, 0.014])
    sm_ = cm.ScalarMappable(cmap=cmap, norm=norm); sm_.set_array([])
    cb = plt.colorbar(sm_, cax=cax, orientation="horizontal")
    cb.set_label("Patent-based digital intensity (share of digital patents)",
                 fontsize=9, labelpad=4)
    cb.ax.tick_params(labelsize=8.5)

    plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.10,
                        wspace=0.06, hspace=0.18)
    plt.savefig(save_to, bbox_inches="tight")
    plt.close()


def main():
    out_dir = ensure_output_dir()
    f2 = out_dir / "figure2_construction_flow.png"
    f3 = out_dir / "figure3_supplier_network.png"
    build_figure2(f2)
    build_figure3(f3)
    print(f"Saved: {f2}")
    print(f"Saved: {f3}")


if __name__ == "__main__":
    main()
