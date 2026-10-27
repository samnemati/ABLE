"""Stage 2: compare four antibody representations on the 12 Jain assays.

Representations: stage 1 baseline features, IMGT-aligned one-hot, ESM-2 650M
embeddings, AbLang2 embeddings. Each is evaluated with ridge regression under
random and sequence-identity-grouped 5-fold cross-validation, 10 repeats.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from able.data import ASSAYS, load_jain
from able.embeddings import load_cache
from able.encodings import aligned_positions, identity_groups, identity_matrix, onehot_aligned
from able.evaluate import holm, paired_bootstrap, permutation_pvalue, repeated_cv

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
CACHE = ROOT / "cache"

# Fixed before looking at any stage 2 result: one assay per property family.
HEADLINE = ["hic", "acsins", "psr", "fab_tm"]
REPRESENTATIONS = ["baseline", "onehot", "esm2", "ablang2"]
LABELS = {
    "baseline": "Baseline features",
    "onehot": "One-hot (IMGT)",
    "esm2": "ESM-2 650M",
    "ablang2": "AbLang2",
}
COLORS = {"baseline": "#898781", "onehot": "#2a78d6", "esm2": "#eb6834", "ablang2": "#1baf7a"}
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"

df = load_jain(ROOT / "data" / "jain.csv")
names = list(df.index)

# Representations, all as matrices with rows in the same antibody order.
maps = [aligned_positions(r["vh"], r["vl"], n) for n, r in df.iterrows()]
stage1 = pd.read_csv(OUT / "t_features.csv", index_col="name").loc[names]
X = {
    "baseline": stage1.drop(columns=list(ASSAYS)).to_numpy(dtype=float),
    "onehot": onehot_aligned(maps)[0],
    "esm2": load_cache(CACHE / "esm2_650m.npz", names),
    "ablang2": load_cache(CACHE / "ablang2.npz", names),
}

groups = identity_groups(maps, min_identity=0.8)
sizes = np.bincount(groups)[1:]
pd.DataFrame({"name": names, "group": groups}).to_csv(OUT / "t_stage2_groups.csv", index=False)
ident = identity_matrix(maps)
cross_max = ident[groups[:, None] != groups[None, :]].max()
print(
    f"Identity groups (>= 0.8, single linkage): {len(sizes)} groups, largest {sizes.max()}, "
    f"singletons {(sizes == 1).sum()}, max identity across groups {cross_max:.3f}"
)

rows, oof = [], {}
for rep in REPRESENTATIONS:
    for assay in ASSAYS:
        y = df[assay].to_numpy(dtype=float)
        for split, split_groups in (("random", None), ("grouped", groups)):
            result = repeated_cv(X[rep], y, split_groups)
            oof[(rep, assay, split)] = result["oof"]
            row = {
                "representation": rep,
                "assay": assay,
                "split": split,
                "rho_mean": result["rho_mean"],
                "rho_sd": result["rho_sd"],
                "ci_low": result["ci_low"],
                "ci_high": result["ci_high"],
                "headline": assay in HEADLINE,
                "quality": ASSAYS[assay]["quality"],
                "p_perm": np.nan,
            }
            if assay in HEADLINE and split == "grouped":
                row["p_perm"] = permutation_pvalue(X[rep], y, groups, result["rho_mean"])
            rows.append(row)
        print(f"  {rep:9s} {assay:10s} done")

results = pd.DataFrame(rows)
results.to_csv(OUT / "t_stage2_results.csv", index=False, float_format="%.4f")

# Does a learned representation beat the simple features? Grouped CV,
# headline assays, paired bootstrap, Holm-corrected across the 12 tests.
comparisons = []
for rep in ["onehot", "esm2", "ablang2"]:
    for assay in HEADLINE:
        y = df[assay].to_numpy(dtype=float)
        delta, lo, hi, p = paired_bootstrap(
            y, oof[(rep, assay, "grouped")], oof[("baseline", assay, "grouped")]
        )
        comparisons.append(
            {"representation": rep, "assay": assay, "delta_rho": delta, "ci_low": lo, "ci_high": hi, "p": p}
        )
comparisons = pd.DataFrame(comparisons)
comparisons["p_holm"] = holm(comparisons["p"].tolist())
comparisons.to_csv(OUT / "t_stage2_comparisons.csv", index=False, float_format="%.4f")

# Figures.
plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
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

grouped = results[results["split"] == "grouped"].set_index(["representation", "assay"])
fig, axes = plt.subplots(1, 4, figsize=(12, 3.8), sharey=True)
for ax, assay in zip(axes, HEADLINE):
    for i, rep in enumerate(REPRESENTATIONS):
        r = grouped.loc[(rep, assay)]
        ax.errorbar(
            i, r["rho_mean"], yerr=[[r["rho_mean"] - r["ci_low"]], [r["ci_high"] - r["rho_mean"]]],
            fmt="o", markersize=8, color=COLORS[rep], ecolor=COLORS[rep], elinewidth=2, capsize=0,
            markeredgecolor=SURFACE, markeredgewidth=2, label=LABELS[rep],
        )
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set_xticks([])
    ax.set_xlim(-0.6, len(REPRESENTATIONS) - 0.4)
    ax.set_title(ASSAYS[assay]["label"], fontsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
axes[0].set_ylabel("Spearman rho (grouped CV, 95% CI)")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("How well each representation predicts the headline assays", color=INK)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(OUT / "f_stage2_headline.png", dpi=150)
plt.close(fig)

wide = results[results["headline"]].pivot_table(
    index=["representation", "assay"], columns="split", values="rho_mean"
)
fig, axes = plt.subplots(1, 4, figsize=(12, 3.8), sharey=True)
for ax, assay in zip(axes, HEADLINE):
    for rep in REPRESENTATIONS:
        r = wide.loc[(rep, assay)]
        ax.plot([0, 1], [r["random"], r["grouped"]], color=COLORS[rep], linewidth=2, marker="o",
                markersize=8, markeredgecolor=SURFACE, markeredgewidth=2, label=LABELS[rep])
    ax.set_xticks([0, 1], ["Random CV", "Grouped CV"])
    ax.set_xlim(-0.3, 1.3)
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set_title(ASSAYS[assay]["label"], fontsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
axes[0].set_ylabel("Spearman rho (mean of 10 repeats)")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Random vs sequence-identity-grouped cross-validation", color=INK)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(OUT / "f_stage2_random_vs_grouped.png", dpi=150)
plt.close(fig)

print(f"Wrote {len(results)} result rows and {len(comparisons)} comparisons")
