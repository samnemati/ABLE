"""Exploratory figures for stage 1: assays, CDR-H3 lengths, features vs assays.

All correlations here are exploratory and uncorrected for multiple comparisons.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

from fvcast.data import ASSAYS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

# Palette: one blue for single-series marks; blue <-> gray <-> red for signed
# correlations, so zero reads as "nothing" and sign reads as hue.
BLUE = "#2a78d6"
DIVERGING = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"

plt.rcParams.update(
    {
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9,
    }
)

table = pd.read_csv(OUT / "t_features.csv", index_col="name")
assays = list(ASSAYS)
features = [c for c in table.columns if c not in assays]
# A feature with one value in every antibody has no correlation to compute.
# Under IMGT, CDR-L2 is three residues long in essentially all antibodies.
constant = [c for c in features if table[c].nunique() == 1]
features = [c for c in features if c not in constant]
print(f"Constant features left out of the correlations: {', '.join(constant) or 'none'}")


def spearman_matrix(rows: list[str], cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rho = pd.DataFrame(index=rows, columns=cols, dtype=float)
    p = rho.copy()
    for r in rows:
        for c in cols:
            rho.loc[r, c], p.loc[r, c] = spearmanr(table[r], table[c])
    return rho, p


def heatmap(ax, rho: pd.DataFrame, p: pd.DataFrame | None = None):
    image = ax.imshow(rho.values, cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(rho.shape[1]), rho.columns, rotation=60, ha="right")
    ax.set_yticks(range(rho.shape[0]), rho.index)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if p is not None:
        for i in range(rho.shape[0]):
            for j in range(rho.shape[1]):
                if p.iat[i, j] < 0.05:
                    ax.text(j, i, "•", ha="center", va="center", color=INK, fontsize=8)
    return image


# 1. Distribution of each assay.
fig, axes = plt.subplots(3, 4, figsize=(11, 7))
for ax, key in zip(axes.flat, assays):
    ax.hist(table[key], bins=20, color=BLUE, edgecolor="#fcfcfb", linewidth=1)
    title = ASSAYS[key]["label"]
    if ASSAYS[key]["quality"] != "continuous":
        title += f"\n[{ASSAYS[key]['quality'].replace('_', ', ')}]"
    ax.set_title(title, fontsize=8.5)
    ax.yaxis.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
fig.suptitle("Distributions of the 12 Jain assays (n = 137)", color=INK)
fig.tight_layout()
fig.savefig(OUT / "f_assay_distributions.png", dpi=150)
plt.close(fig)

# 2. Assay-assay Spearman correlations, ordered by hierarchical clustering so
# assays that measure related properties sit next to each other.
rho, _ = spearman_matrix(assays, assays)
order = leaves_list(linkage(squareform(1 - rho.values, checks=False), "average"))
rho = rho.iloc[order, order]
fig, ax = plt.subplots(figsize=(6.5, 5.5))
image = heatmap(ax, rho)
fig.colorbar(image, ax=ax, shrink=0.8, label="Spearman rho")
ax.set_title("Correlations between assays", color=INK)
fig.tight_layout()
fig.savefig(OUT / "f_assay_correlations.png", dpi=150)
plt.close(fig)

# 3. CDR-H3 length, the most variable part of an antibody.
lengths = table["len_cdrh3"].astype(int)
fig, ax = plt.subplots(figsize=(6, 3.5))
bins = np.arange(lengths.min(), lengths.max() + 2) - 0.5
ax.hist(lengths, bins=bins, color=BLUE, edgecolor="#fcfcfb", linewidth=1.5)
ax.axvline(lengths.median(), color=INK_2, linewidth=1, linestyle="--")
ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
ax.text(lengths.median(), ax.get_ylim()[1] * 0.97, f"median {lengths.median():.0f}", color=INK_2, ha="center", va="top",
        bbox={"facecolor": "#fcfcfb", "edgecolor": "none", "pad": 1})
ax.set_xticks(range(lengths.min(), lengths.max() + 1, 2))
ax.set_xlabel("CDR-H3 length (residues, IMGT)")
ax.set_ylabel("Antibodies")
ax.yaxis.grid(True, color=GRID, linewidth=0.5)
ax.set_axisbelow(True)
ax.set_title("CDR-H3 length across 137 clinical antibodies", color=INK)
fig.tight_layout()
fig.savefig(OUT / "f_cdrh3_length.png", dpi=150)
plt.close(fig)

# 4. Baseline features vs assays.
rho, p = spearman_matrix(features, assays)
fig, ax = plt.subplots(figsize=(8, 8))
image = heatmap(ax, rho, p)
fig.colorbar(image, ax=ax, shrink=0.6, label="Spearman rho")
ax.set_title("Baseline features vs assays\n(• p < 0.05, uncorrected, exploratory)", color=INK)
fig.tight_layout()
fig.savefig(OUT / "f_features_vs_assays.png", dpi=150)
plt.close(fig)

long = rho.stack().rename("rho").to_frame().join(p.stack().rename("p_uncorrected"))
long.index.names = ["feature", "assay"]
long["note"] = "exploratory, uncorrected"
long.to_csv(OUT / "t_features_vs_assays.csv")

# Sanity checks reported in the README.
hic_smac = spearmanr(table["hic"], table["smac"])
print(f"Median CDR-H3 length: {lengths.median():.0f} (range {lengths.min()}-{lengths.max()})")
print(f"HIC vs SMAC: Spearman rho = {hic_smac.statistic:.2f}, p = {hic_smac.pvalue:.1e}")
print(f"Feature-assay pairs with p < 0.05: {(p.values < 0.05).sum()} of {p.size}")
