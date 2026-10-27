"""Stage 3: what the predicted structures look like, and whether surface helps.

Reads cache/structures/ (from 45_predict_structures.py). Writes per-antibody
structure and graph summaries, surface features, an evaluation of surface
features against the stage 1 sequence features, and three figures.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

from able.data import ASSAYS, load_jain
from able.evaluate import holm, paired_bootstrap, permutation_pvalue, repeated_cv
from able.graphs import residue_graph
from able.numbering import IMGT_REGIONS
from able.structure import check_matches_sequence, read_residues
from able.surface import surface_features

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
STRUCTURES = ROOT / "cache" / "structures"

HEADLINE = ["hic", "acsins", "psr", "fab_tm"]
REPRESENTATIONS = ["baseline", "surface", "baseline+surface"]
LABELS = {"baseline": "Baseline features", "surface": "Surface (structure)", "baseline+surface": "Baseline + surface"}
COLORS = {"baseline": "#898781", "surface": "#2a78d6", "baseline+surface": "#eb6834"}
INK, INK_2, MUTED, GRID, SURFACE_BG = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE_BG,
        "axes.facecolor": SURFACE_BG,
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

df = load_jain(ROOT / "data" / "jain.csv")

# Keep only antibodies whose structure exists and matches its sequence.
residues, missing = {}, []
for name, row in df.iterrows():
    path = STRUCTURES / f"{name}.pdb"
    try:
        res = read_residues(path)
        check_matches_sequence(res, row["vh"], row["vl"], name)
        residues[name] = res
    except Exception:
        missing.append(name)
names = list(residues)
df = df.loc[names]
print(f"Structures used: {len(names)} of {len(names) + len(missing)}")
if missing:
    print(f"  missing or mismatched: {', '.join(missing[:10])}" + (" ..." if len(missing) > 10 else ""))


def region_of(chain: str, position: int) -> str:
    for region, (first, last) in IMGT_REGIONS.items():
        if first <= position <= last:
            return f"{chain}_{region}"
    raise ValueError(position)


# 1. Structure and graph summaries.
summary_rows, region_errors = [], []
for name in names:
    res = residues[name]
    graph = residue_graph(res)
    h3 = (res["chain"] == "H") & res["position"].between(105, 117)
    summary_rows.append(
        {
            "name": name,
            "n_residues": len(res),
            "mean_error": res["error"].mean(),
            "mean_error_cdrh3": res.loc[h3, "error"].mean(),
            "max_error": res["error"].max(),
            "n_edges": graph["edge_index"].shape[1] // 2,
            "mean_degree": graph["edge_index"].shape[1] / len(res),
        }
    )
    regions = [region_of(c, p) for c, p in zip(res["chain"], res["position"])]
    region_errors.append(res.assign(region=regions).groupby("region")["error"].mean())
summary = pd.DataFrame(summary_rows).set_index("name")
summary.to_csv(OUT / "t_structures.csv", float_format="%.4f")

region_order = [f"{c}_{r}" for c in "HL" for r in IMGT_REGIONS]
region_mean = pd.DataFrame(region_errors).mean()[region_order]
print(f"Region with highest mean predicted error: {region_mean.idxmax()} ({region_mean.max():.2f} Å)")
print(
    f"Graphs: {summary['n_residues'].min()}-{summary['n_residues'].max()} residues, "
    f"mean degree {summary['mean_degree'].mean():.1f}, median edges {summary['n_edges'].median():.0f}"
)

# 2. Surface features.
surface = pd.DataFrame({name: surface_features(STRUCTURES / f"{name}.pdb") for name in names}).T
surface.index.name = "name"
surface.join(df[list(ASSAYS)]).to_csv(OUT / "t_surface_features.csv", float_format="%.4f")
rho = spearmanr(surface["sasa_hydrophobic"], df["hic"]).statistic
print(f"Hydrophobic SASA vs HIC: Spearman rho = {rho:.2f}")

# 3. Does surface add to the stage 1 sequence features?
stage1 = pd.read_csv(OUT / "t_features.csv", index_col="name").loc[names].drop(columns=list(ASSAYS))
groups = pd.read_csv(OUT / "t_stage2_groups.csv", index_col="name").loc[names, "group"].to_numpy()
X = {
    "baseline": stage1.to_numpy(dtype=float),
    "surface": surface.to_numpy(dtype=float),
    "baseline+surface": np.hstack([stage1.to_numpy(dtype=float), surface.to_numpy(dtype=float)]),
}

rows, oof = [], {}
for rep in REPRESENTATIONS:
    for assay in ASSAYS:
        y = df[assay].to_numpy(dtype=float)
        for split, split_groups in (("random", None), ("grouped", groups)):
            if rep == "baseline" and not (assay in HEADLINE and split == "grouped"):
                continue  # the baseline is only needed for the comparisons
            result = repeated_cv(X[rep], y, split_groups)
            oof[(rep, assay, split)] = result["oof"]
            row = {
                "representation": rep, "assay": assay, "split": split,
                "rho_mean": result["rho_mean"], "rho_sd": result["rho_sd"],
                "ci_low": result["ci_low"], "ci_high": result["ci_high"],
                "headline": assay in HEADLINE, "quality": ASSAYS[assay]["quality"], "p_perm": np.nan,
            }
            if assay in HEADLINE and split == "grouped":
                row["p_perm"] = permutation_pvalue(X[rep], y, groups, result["rho_mean"])
            rows.append(row)
    print(f"  {rep} done")
results = pd.DataFrame(rows)
results.to_csv(OUT / "t_stage3_results.csv", index=False, float_format="%.4f")

comparisons = []
for rep in ["surface", "baseline+surface"]:
    for assay in HEADLINE:
        y = df[assay].to_numpy(dtype=float)
        delta, lo, hi, p = paired_bootstrap(y, oof[(rep, assay, "grouped")], oof[("baseline", assay, "grouped")])
        comparisons.append({"representation": rep, "assay": assay, "delta_rho": delta, "ci_low": lo, "ci_high": hi, "p": p})
comparisons = pd.DataFrame(comparisons)
comparisons["p_holm"] = holm(comparisons["p"].tolist())
comparisons.to_csv(OUT / "t_stage3_comparisons.csv", index=False, float_format="%.4f")

# 4. Figures.
fig, ax = plt.subplots(figsize=(9, 3.6))
colors = ["#eb6834" if "CDR" in r else MUTED for r in region_order]
ax.bar(range(len(region_order)), region_mean.values, color=colors, edgecolor=SURFACE_BG, linewidth=2)
ax.set_xticks(range(len(region_order)), [r.replace("_", " ") for r in region_order], rotation=45, ha="right")
ax.set_ylabel("Mean predicted error (Å)")
ax.yaxis.grid(True, color=GRID, linewidth=0.5)
ax.set_axisbelow(True)
ax.set_title(f"Where ABodyBuilder2 is uncertain (mean over {len(names)} antibodies; CDRs in orange)", color=INK)
fig.tight_layout()
fig.savefig(OUT / "f_stage3_error_by_region.png", dpi=150)
plt.close(fig)

example = "trastuzumab" if "trastuzumab" in residues else names[0]
res = residues[example]
dist = cdist(res[["x", "y", "z"]].to_numpy(), res[["x", "y", "z"]].to_numpy())
fig, ax = plt.subplots(figsize=(6.2, 5.6))
edges = (dist <= 10.0) & ~np.eye(len(dist), dtype=bool)  # the graph has no self-loops
image = ax.imshow(np.where(edges, dist, np.nan), cmap="Blues_r", vmin=0, vmax=10, interpolation="nearest")
n_heavy = int((res["chain"] == "H").sum())
for i, (c, p) in enumerate(zip(res["chain"], res["position"])):
    if any(a <= p <= b for a, b in ((27, 38), (56, 65), (105, 117))):
        ax.add_patch(plt.Rectangle((i - 0.5, -6), 1, 4, color="#eb6834", clip_on=False, linewidth=0))
        ax.add_patch(plt.Rectangle((-6, i - 0.5), 4, 1, color="#eb6834", clip_on=False, linewidth=0))
ax.axhline(n_heavy - 0.5, color=MUTED, linewidth=0.8)
ax.axvline(n_heavy - 0.5, color=MUTED, linewidth=0.8)
ax.set_xlim(-6.5, len(res) - 0.5)
ax.set_ylim(len(res) - 0.5, -6.5)
ax.set_xticks([n_heavy / 2, n_heavy + (len(res) - n_heavy) / 2], ["Heavy chain", "Light chain"])
ax.set_yticks([n_heavy / 2, n_heavy + (len(res) - n_heavy) / 2], ["Heavy", "Light"])
ax.tick_params(length=0)
for spine in ax.spines.values():
    spine.set_visible(False)
fig.colorbar(image, ax=ax, shrink=0.75, label="Cα-Cα distance (Å), edges only")
ax.set_title(f"{example}: residue graph edges (≤ 10 Å); CDRs marked in orange", color=INK, fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "f_stage3_contact_map.png", dpi=150)
plt.close(fig)

grouped = results[results["split"] == "grouped"].set_index(["representation", "assay"])
fig, axes = plt.subplots(1, 4, figsize=(11, 3.8), sharey=True)
for ax, assay in zip(axes, HEADLINE):
    for i, rep in enumerate(REPRESENTATIONS):
        r = grouped.loc[(rep, assay)]
        ax.errorbar(
            i, r["rho_mean"], yerr=[[r["rho_mean"] - r["ci_low"]], [r["ci_high"] - r["rho_mean"]]],
            fmt="o", markersize=8, color=COLORS[rep], ecolor=COLORS[rep], elinewidth=2, capsize=0,
            markeredgecolor=SURFACE_BG, markeredgewidth=2, label=LABELS[rep],
        )
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set_xticks([])
    ax.set_xlim(-0.6, len(REPRESENTATIONS) - 0.4)
    ax.set_title(ASSAYS[assay]["label"], fontsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
axes[0].set_ylabel("Spearman rho (grouped CV, 95% CI)")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Do structure-based surface features add to sequence features?", color=INK)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(OUT / "f_stage3_headline.png", dpi=150)
plt.close(fig)

print(f"Wrote {len(results)} result rows and {len(comparisons)} comparisons")
