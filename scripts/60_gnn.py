"""Stage 4: a graph neural network on the residue graphs.

Three variants, all on the same grouped folds as every earlier stage:
  gnn-basic            residue identity, chain, CDR flag, predicted error
  gnn-esm2             the same plus per-residue ESM-2 vectors
  gnn-shuffled-edges   gnn-basic features on randomly rewired graphs, the
                       control that asks whether the structure matters

Also writes a summary of every representation from stages 1 to 4.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from able.data import ASSAYS, load_jain
from able.evaluate import holm, paired_bootstrap
from able.gnn import gnn_repeated_cv, shuffle_edges, to_pyg
from able.graphs import residue_graph
from able.structure import check_matches_sequence, read_residues

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
STRUCTURES = ROOT / "cache" / "structures"

HEADLINE = ["hic", "acsins", "psr", "fab_tm"]
VARIANTS = ["gnn-basic", "gnn-esm2", "gnn-shuffled-edges"]
LABELS = {
    "gnn-basic": "GNN (basic features)",
    "gnn-esm2": "GNN (+ ESM-2 per residue)",
    "gnn-shuffled-edges": "GNN (shuffled edges)",
    "best-earlier": "Best from stages 1-3",
}
COLORS = {
    "gnn-basic": "#2a78d6",
    "gnn-esm2": "#eb6834",
    "gnn-shuffled-edges": "#898781",
    "best-earlier": "#1baf7a",
}
INK, MUTED, GRID, BG = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
plt.rcParams.update(
    {
        "figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": MUTED,
        "axes.labelcolor": "#52514e", "axes.titlecolor": INK, "xtick.color": MUTED,
        "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    }
)

df = load_jain(ROOT / "data" / "jain.csv")
names = list(df.index)
groups = pd.read_csv(OUT / "t_stage2_groups.csv", index_col="name").loc[names, "group"].to_numpy()

residue_cache = np.load(ROOT / "cache" / "esm2_650m_residues.npz")
assert list(residue_cache["names"]) == names, "per-residue embedding cache is out of order"
offsets = residue_cache["offsets"]
embeddings = {n: residue_cache["X"][offsets[i] : offsets[i + 1]] for i, n in enumerate(names)}

graphs = {}
for name, row in df.iterrows():
    res = read_residues(STRUCTURES / f"{name}.pdb")
    check_matches_sequence(res, row["vh"], row["vl"], name)
    graphs[name] = residue_graph(res)
print(f"Loaded {len(graphs)} residue graphs")

rows, oof = [], {}
for assay in HEADLINE:
    y = df[assay].to_numpy(dtype=float)
    basic = [to_pyg(graphs[n], y[i]) for i, n in enumerate(names)]
    datasets = {
        "gnn-basic": basic,
        "gnn-esm2": [to_pyg(graphs[n], y[i], node_embedding=embeddings[n]) for i, n in enumerate(names)],
        "gnn-shuffled-edges": [shuffle_edges(d, seed=i) for i, d in enumerate(basic)],
    }
    for variant in VARIANTS:
        result = gnn_repeated_cv(datasets[variant], y, groups)
        oof[(variant, assay)] = result["oof"]
        rows.append(
            {
                "representation": variant, "assay": assay, "split": "grouped",
                "rho_mean": result["rho_mean"], "rho_sd": result["rho_sd"],
                "ci_low": result["ci_low"], "ci_high": result["ci_high"],
                "headline": True, "quality": ASSAYS[assay]["quality"],
            }
        )
        print(f"  {assay:7s} {variant:19s} rho {result['rho_mean']:+.3f}")
results = pd.DataFrame(rows)
results.to_csv(OUT / "t_stage4_results.csv", index=False, float_format="%.4f")

# Every representation from every stage, on the headline assays, grouped CV.
stage2 = pd.read_csv(OUT / "t_stage2_results.csv")
stage3 = pd.read_csv(OUT / "t_stage3_results.csv")
earlier = pd.concat([stage2, stage3])
earlier = earlier[(earlier["split"] == "grouped") & earlier["assay"].isin(HEADLINE)]
earlier = earlier.drop_duplicates(subset=["representation", "assay"])
all_stages = pd.concat([earlier[results.columns.intersection(earlier.columns)], results])
all_stages.to_csv(OUT / "t_all_stages.csv", index=False, float_format="%.4f")

best_earlier = earlier.loc[earlier.groupby("assay")["rho_mean"].idxmax()].set_index("assay")
print("\nBest pre-GNN representation per assay:")
print(best_earlier[["representation", "rho_mean"]].round(3).to_string())

# Comparisons: each GNN variant against gnn-shuffled-edges (does structure
# matter?), and gnn-basic / gnn-esm2 against the best earlier representation.
comparisons = []
for assay in HEADLINE:
    y = df[assay].to_numpy(dtype=float)
    for a, b in (("gnn-basic", "gnn-shuffled-edges"), ("gnn-esm2", "gnn-shuffled-edges")):
        delta, lo, hi, p = paired_bootstrap(y, oof[(a, assay)], oof[(b, assay)])
        comparisons.append({"assay": assay, "a": a, "b": b, "delta_rho": delta, "ci_low": lo, "ci_high": hi, "p": p})
comparisons = pd.DataFrame(comparisons)
comparisons["p_holm"] = holm(comparisons["p"].tolist())
comparisons.to_csv(OUT / "t_stage4_comparisons.csv", index=False, float_format="%.4f")
print("\n" + comparisons.round(4).to_string(index=False))

# Figure 1: the three GNN variants plus the best earlier representation.
fig, axes = plt.subplots(1, 4, figsize=(12, 3.9), sharey=True)
grouped = results.set_index(["representation", "assay"])
for ax, assay in zip(axes, HEADLINE):
    entries = [(v, grouped.loc[(v, assay)]) for v in VARIANTS]
    entries.append(("best-earlier", best_earlier.loc[assay]))
    for i, (key, r) in enumerate(entries):
        ax.errorbar(
            i, r["rho_mean"], yerr=[[r["rho_mean"] - r["ci_low"]], [r["ci_high"] - r["rho_mean"]]],
            fmt="o", markersize=8, color=COLORS[key], ecolor=COLORS[key], elinewidth=2, capsize=0,
            markeredgecolor=BG, markeredgewidth=2,
            label=LABELS[key] if key != "best-earlier" else f"{LABELS[key]}",
        )
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set_xticks([])
    ax.set_xlim(-0.6, 3.4)
    ax.set_title(f"{ASSAYS[assay]['label']}\nbest earlier: {best_earlier.loc[assay, 'representation']}", fontsize=8.5)
    ax.yaxis.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
axes[0].set_ylabel("Spearman rho (grouped CV, 95% CI)")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Does a graph neural network beat the simpler representations?", color=INK)
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(OUT / "f_stage4_headline.png", dpi=150)
plt.close(fig)

# Figure 2: the best representation of each kind, per assay.
# Five distinct categorical slots, one per kind of representation: cycling a
# shorter palette would give two kinds the same colour.
KINDS = {
    "baseline": ("Sequence features", "#898781"),
    "esm2": ("ESM-2", "#eb6834"),
    "ablang2": ("AbLang2", "#1baf7a"),
    "surface": ("Structure surface", "#2a78d6"),
    "gnn-esm2": ("GNN + ESM-2", "#6250d6"),
}
fig, ax = plt.subplots(figsize=(10, 4.2))
width = 0.16
lookup = pd.concat([earlier, results]).set_index(["representation", "assay"])
for k, (key, (label, colour)) in enumerate(KINDS.items()):
    values = [lookup.loc[(key, a), "rho_mean"] if (key, a) in lookup.index else np.nan for a in HEADLINE]
    ax.bar(np.arange(len(HEADLINE)) + k * width, values, width,
           label=label, color=colour, edgecolor=BG, linewidth=1.5)
ax.set_xticks(np.arange(len(HEADLINE)) + 2 * width, [ASSAYS[a]["label"].split(" (")[0] for a in HEADLINE])
ax.axhline(0, color=MUTED, linewidth=1)
ax.set_ylabel("Spearman rho (grouped CV)")
ax.yaxis.grid(True, color=GRID, linewidth=0.5)
ax.set_axisbelow(True)
ax.legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.08))
ax.set_title("Which description of an antibody wins depends on the property", color=INK)
fig.tight_layout()
fig.savefig(OUT / "f_all_stages.png", dpi=150)
plt.close(fig)

print(f"\nWrote {len(results)} stage 4 rows and {len(comparisons)} comparisons")
