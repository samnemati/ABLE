# ABLE

**Antibody Behavior Learning & Evaluation.** Is this antibody able to become a drug? Predicting how
a therapeutic antibody will behave in manufacturing and in the body from its variable-domain
sequences.

Predicting antibody developability (expression, stability, hydrophobicity, self-interaction and
polyreactivity) from sequence and predicted structure, on 137 clinical-stage therapeutic
antibodies. The aim is a careful comparison of simple sequence features, protein language model
embeddings and graph neural networks on structure, evaluated with splits that keep similar
antibodies out of training and test at the same time.

## Why I built this

My research has been in applying machine learning to biological data from the brain: graph
neural networks on functional and structural connectomes, intracranial neural signals, and
lesion-style analyses of attention heads in vision-language models. I had not worked with protein
data, and I built this project to learn it from the ground up and to test whether the habits I rely
on carry over to a new kind of molecule: a public dataset with documented provenance, a simple
baseline that any complex model has to beat, and an evaluation that does not allow near-identical
samples to sit on both sides of a train/test split.

Antibody developability is a good place to do that. The datasets are small because every
measurement requires producing and testing a real molecule, several of the assays are noisy or
censored, and whether a learned representation adds anything beyond charge and loop length is
still an open question. Those are the same conditions I have worked under with clinical cohorts of
a few hundred people, where the honest answer is often that the simple model is hard to beat.

## Data

Jain et al. (2017) measured 137 antibodies that reached clinical trials or approval on 12
biophysical assays. Each antibody comes with its heavy-chain (VH) and light-chain (VL)
variable-region sequence. The pipeline downloads a checksummed copy; source, license, contents
and assay caveats are in [PROVENANCE.md](PROVENANCE.md).

## Method so far

1. **Load** ([able/data.py](able/data.py)): download, verify SHA-256, rename the 12 assays to
   short names, and record for each assay whether higher values are better and whether it is a
   clean continuous readout. SGAC-SINS (ordinal, censored at both ends) and CSI-BLI (most values
   at a floor near zero) are flagged.
2. **Number** ([able/numbering.py](able/numbering.py)): number every VH and VL with ANARCI in
   the IMGT scheme, which places all antibodies on one coordinate system, then cut each chain into
   four framework regions and three complementarity-determining regions (CDRs, the loops that
   bind the target). IMGT boundaries: CDR1 27-38, CDR2 56-65, CDR3 105-117.
3. **Baseline features** ([able/features.py](able/features.py)): chain lengths, net charge at
   pH 7.4 (whole variable region and CDRs only), Kyte-Doolittle hydrophobicity (variable region,
   CDRs, CDR-H3), the six CDR lengths, and counts of chemical liability motifs inside the CDRs
   (N-glycosylation, deamidation, isomerization, methionine and tryptophan oxidation, cysteine).
   These are the descriptors the learned representations in later stages have to beat.

All 137 antibodies number cleanly, with both chains assigned to the expected chain type.

## First look at the data

**The assays.** Most developability readouts are strongly skewed: the majority of clinical
antibodies behave well, and a tail behaves badly. That tail is what a model needs to catch.

![Assay distributions](outputs/f_assay_distributions.png)

**How the assays relate.** The assays fall into groups that measure related properties. HIC and
SMAC, both hydrophobicity measures, correlate strongly (Spearman rho = 0.86), as Jain et al.
reported. The polyreactivity and self-interaction assays (ELISA, BVP ELISA, PSR, CSI-BLI, AC-SINS,
CIC) form a second block. Expression titer and Fab melting temperature stand mostly apart.

![Assay correlations](outputs/f_assay_correlations.png)

**CDR-H3 length.** The third heavy-chain CDR is the most variable part of an antibody. Here it
ranges from 5 to 23 residues (IMGT), with a median of 12.

![CDR-H3 length](outputs/f_cdrh3_length.png)

**Simple features already carry signal.** Net positive charge of the variable region goes with
polyreactivity and self-interaction (PSR rho = 0.37, AC-SINS rho = 0.34, ELISA rho = 0.30) and
with shorter hydrophobic retention (HIC rho = -0.31). Longer CDR-H3 loops go with longer
hydrophobic retention (SMAC rho = 0.35, HIC rho = 0.27). These correlations are exploratory and
not corrected for the 216 feature-assay pairs tested; the full table is in
[`outputs/t_features_vs_assays.csv`](outputs/t_features_vs_assays.csv). CDR-L2 is three residues
long in every antibody here, so it is left out.

![Features vs assays](outputs/f_features_vs_assays.png)

## Stage 2: protein language models

**Question.** Do learned representations of the sequence predict developability better than the
simple features from stage 1, once the evaluation stops close relatives from sitting in both
training and test?

**Four representations of each antibody** ([able/encodings.py](able/encodings.py),
[able/embeddings.py](able/embeddings.py)):

| Representation | What it is | Dimensions |
|---|---|---|
| Baseline features | Stage 1: charge, hydrophobicity, CDR lengths, liability counts | 19 |
| One-hot (IMGT) | Each residue at each IMGT address, aligned across antibodies, gaps encoded | 5,292 |
| ESM-2 650M | General protein language model; VH and VL embedded separately, mean-pooled, joined | 2,560 |
| AbLang2 | Antibody language model trained on paired heavy and light chains | 480 |

ESM-2 embeddings were computed on an Apple GPU (MPS) and cached; recomputing on a CPU can change
them in the last decimal places.

**Model and evaluation** ([able/evaluate.py](able/evaluate.py)):
- Ridge regression on standardized features. The scaler and the regularization strength are both
  fit inside each training fold, so the cross-validation is nested.
- 5-fold cross-validation repeated 10 times. The score is the mean over repeats of the Spearman
  rho between out-of-fold predictions and measurements.
- 95% confidence intervals bootstrap antibodies and recompute that same mean rho. The predictions
  stay fixed during resampling, so the intervals capture sampling noise in the evaluation set,
  not the full cost of refitting on new data.
- Permutation p-values (200 label shuffles) for the headline assays.
- Each learned representation is compared with the baseline features by paired bootstrap on the
  same mean rho, Holm-corrected across the 12 comparisons. With 2,000 resamples, a raw p-value
  cannot be resolved below 0.0005, so it is reported as at least that.
- The headline assays were fixed before any results, one per property family: HIC
  (hydrophobicity), AC-SINS (self-interaction), PSR (polyreactivity) and Fab Tm (stability).

**Splits that respect sequence similarity.** Pairwise sequence identity is computed over
IMGT-aligned positions of both chains. Antibodies are grouped by single-linkage clustering at an
identity of 0.8, which guarantees that no two antibodies in different groups are 80% identical or
more. The maximum identity across groups is 0.799. Grouped cross-validation keeps each group on
one side of every split. This gives 61 groups. The largest holds 39 antibodies, because single
linkage chains relatives of relatives together, and 48 antibodies form groups of their own.

The identity analysis was itself informative. The CDRs of these clinical antibodies are nearly
all distinct (median pairwise CDR identity 0.31), while their frameworks are shared (median
full-Fv identity 0.60). Close relatives here are mostly antibodies built on similar germline
frameworks, not copies of the same binder.

![Headline results](outputs/f_stage2_headline.png)

**Results (grouped CV).**
- **Stability: a clear gain from language models.** The simple features do not predict Fab
  melting temperature (rho = -0.03, 95% CI -0.17 to 0.10). AbLang2 reaches rho = 0.47 (0.34 to
  0.60) and ESM-2 0.31 (0.15 to 0.45). Both improvements over the baseline survive Holm correction
  (AbLang2 +0.51, ESM-2 +0.34, adjusted p = 0.006 for each, the smallest value this test can
  give). Stability depends on how the whole domain packs, which charge and loop length do not
  capture.
- **Hydrophobicity, self-interaction, polyreactivity: no gain that survives correction.**
  - AbLang2 scores rho = 0.41 against 0.24 for the baseline on AC-SINS (difference +0.17, raw
    p = 0.044, adjusted p = 0.40).
  - ESM-2 scores 0.32 against 0.21 on PSR.
  - On HIC all four representations sit between 0.15 and 0.23.

  These properties are largely surface charge and hydrophobicity, which the simple features
  already describe.
- **One-hot is the weakest learned representation.** Its best headline result is Fab Tm
  (rho = 0.16). With about 110 training antibodies and 5,292 sparse columns, ridge has little to
  learn from.

Fourteen of the 16 headline models beat shuffled labels (permutation p of 0.05 or less). The two
exceptions are the baseline features on Fab Tm (p = 0.28) and one-hot on PSR (p = 0.065). One-hot
on HIC sits at the boundary (p = 0.05, CI -0.002 to 0.28). With 200 shuffles the smallest possible
p is 0.005. The full 12-assay table is in
[`outputs/t_stage2_results.csv`](outputs/t_stage2_results.csv) and is exploratory. It includes the
flagged SGAC-SINS and CSI-BLI rows. The paired comparisons are in
[`outputs/t_stage2_comparisons.csv`](outputs/t_stage2_comparisons.csv).

![Random vs grouped](outputs/f_stage2_random_vs_grouped.png)

**Random vs grouped cross-validation.** The assay most inflated by random splits is HIC. Under
grouping, every representation drops:

| Representation | Random | Grouped |
|---|---|---|
| ESM-2 | 0.31 | 0.19 |
| AbLang2 | 0.36 | 0.23 |
| One-hot | 0.28 | 0.15 |
| Baseline | 0.21 | 0.15 |

Hydrophobicity seems to run partly in framework families, so a random split lets a model score
by recognizing the family. One-hot on Fab Tm also drops, from 0.26 to 0.16. Most other headline
scores change by a few hundredths, comparable to the spread between repeats. The stability result
holds up under grouping (AbLang2 0.49 to 0.47, ESM-2 0.32 to 0.31).

**Limitations.**
- 137 antibodies, one lab, and one linear model.
- The confidence intervals are wide, and a difference of 0.1 in rho falls inside them for most
  assays.
- The largest identity group holds 39 antibodies, so the grouped folds are unequal in size.
- A nonlinear model, or fine-tuning the language models, could change the picture. Doing either
  safely would need far more than 137 antibodies.

## Stage 3: structure

**Question.** Does the 3D shape of an antibody tell us anything about developability that its
sequence does not?

**Predicted structures** ([able/structure.py](able/structure.py)). There are no experimental
structures for most of these antibodies, so each Fv is predicted with ABodyBuilder2 (ImmuneBuilder;
Abanades et al., 2023). It is an antibody-specific deep learning model built on the AlphaFold2
structure module. It runs an ensemble of four models, and the spread between them is its
per-residue error estimate. The model closest to the ensemble average is kept, and OpenMM refines
it to remove steric clashes. All 137
antibodies were predicted with no failures, in 11 minutes on a laptop. Each structure is checked
against its input sequence residue by residue, and its IMGT numbering matches stage 1 exactly.

![Predicted error by region](outputs/f_stage3_error_by_region.png)

**Where the predictions are uncertain.** CDR-H3 has the highest mean predicted error of the 14
regions: 0.69 Å, against a mean of 0.28 Å over all residues. Per antibody, the CDR-H3 mean error
ranges up to 3.27 Å. That is the expected pattern, because CDR-H3 is the most variable loop and the
hardest to model. These are predicted structures, and their least reliable part is the part that
matters most for binding.

**Residue graphs** ([able/graphs.py](able/graphs.py)). Each structure becomes a graph with one node
per residue (at its Cα atom), and an edge wherever two Cα atoms are within 10 Å.
- Node features: amino acid, chain, framework or CDR, and predicted error.
- Edge feature: distance.

Graphs have 220 to 238 nodes, about 2,070 undirected edges and a mean degree of 18.2. Distances do
not change when a structure is rotated or moved, so the graph does not depend on how the molecule
happens to sit in the file. These graphs are the input for the graph neural network in stage 4.

![Contact map](outputs/f_stage3_contact_map.png)

**Surface features** ([able/surface.py](able/surface.py)). Solvent-accessible surface area
(Shrake-Rupley) is computed per residue, and each residue is called exposed when its area is at
least 25% of its theoretical maximum (Tien et al., 2013). Nine features summarize it:
- total, hydrophobic, positive (K, R) and negative (D, E) surface area
- the hydrophobic fraction of the surface
- hydrophobic and positive surface area within the CDRs
- the exposed net charge (count of exposed K and R minus exposed D and E)
- the number of exposed hydrophobic residues in CDR-H3

The area features sum over all residues; buried residues contribute almost nothing. The 25%
threshold is used only for the two count features.

They are evaluated with the stage 2 engine and the same identity groups. Each comparison is
against the stage 1 sequence features, Holm-corrected across 8 comparisons.

![Stage 3 headline](outputs/f_stage3_headline.png)

**Results (grouped CV).**
- **HIC: surface features predict it far better than anything so far.** All nine surface features
  together reach rho = 0.53 (95% CI 0.42 to 0.64), against 0.15 for the stage 1 sequence features
  (difference +0.39, adjusted p = 0.004) and 0.23 for AbLang2, the best stage 2 representation on
  HIC (difference +0.31, adjusted p = 0.002 in a separate paired test). SMAC, the other
  hydrophobicity-related assay, follows (rho = 0.39).
- **The signal is in charge as much as in hydrophobic surface.** An ablation on HIC
  ([`outputs/t_stage3_ablation.csv`](outputs/t_stage3_ablation.csv)) shows this:

  | Features | HIC rho (grouped CV) |
  |---|---|
  | Hydrophobic surface features alone | 0.18 |
  | Charge surface features alone | 0.39 |
  | All nine surface features | 0.53 |
  | Sequence-only counts of K+R and D+E | 0.31 |
  | Stage 1 features plus those counts | 0.26 |

  The strongest single feature is positive surface area, which goes with shorter HIC retention
  (rho = -0.43). The stage 1 features held net charge but not the number of charged residues.
  Simply counting them gets HIC to 0.31 without any structure. The structure-based charge features
  are not significantly better than those counts (+0.08, p = 0.09). The full surface set is
  (+0.27 over stage 1 plus counts, adjusted p = 0.002). The gain from structure comes from
  combining where charged and hydrophobic residues sit on the surface, not from hydrophobic
  exposure alone.
- **Self-interaction: structure alone is worse.** On AC-SINS, surface features score rho = -0.02,
  significantly below the sequence features (difference -0.26, adjusted p = 0.012). Combining the
  two recovers most of the sequence-feature performance (0.19).
- **Combining helps stability a little.** Sequence plus surface features improve Fab Tm over
  sequence features alone (+0.14, adjusted p = 0.012). At rho = 0.11, this is still far below the
  language models in stage 2 (AbLang2, 0.47).
- **Combining is not always better.** On HIC, sequence plus surface (0.41) scores below surface
  alone (0.53). With about 110 training antibodies, extra weakly related features likely cost ridge
  more than they add.

Random and grouped cross-validation give similar results for the surface features (HIC 0.55
random, 0.53 grouped). With only nine features, ridge has little room to memorize sequence
families. The full tables are in
[`outputs/t_stage3_results.csv`](outputs/t_stage3_results.csv),
[`outputs/t_stage3_comparisons.csv`](outputs/t_stage3_comparisons.csv) and
[`outputs/t_stage3_ablation_tests.csv`](outputs/t_stage3_ablation_tests.csv).

**Limitations.**
- The structures are predictions, least reliable in CDR-H3.
- They come from a single prediction method, and only for the Fv, not the full IgG.
- SASA on a static model ignores flexibility.
- The exposure threshold (25%) and the hydrophobic residue set are conventional choices that
  were not tuned.


## Stage 4: graph neural network

**Question.** Does a model that reads the whole residue graph beat the simpler representations that
summarize it, and does the 3D wiring carry information beyond the residues themselves?

**The model** ([able/gnn.py](able/gnn.py)). Two `GraphConv` layers (hidden 64, batch normalization,
ReLU, dropout 0.3), global mean pooling over residues, then a small linear head. Edge weight is
`exp(-d / 4 Å)`, so a residue's closest neighbours count for more than contacts at the edge of the
10 Å cutoff. This is deliberately the same shape of model used on brain connectomes in GraphBRAIN,
so the comparison is about the data rather than about who tuned harder. The architecture and
hyperparameters were fixed in advance and never tuned against the test folds.

**Four variants, two matched pairs:**

| Variant | Node features | Graph |
|---|---|---|
| `gnn-basic` | residue identity, chain, CDR flag, predicted error (23) | real |
| `gnn-basic-shuffled` | the same (23) | rewired |
| `gnn-esm2` | the same plus the per-residue ESM-2 vector (1303) | real |
| `gnn-esm2-shuffled` | the same (1303) | rewired |

Each real graph has a rewired twin with **identical node features**, so the only thing that differs
within a pair is the wiring. Rewiring uses repeated double-edge swaps, which preserve every
residue's number of contacts exactly: the configuration-model null used for connectomes, not a
plain random graph. That matters because `GraphConv` sums over neighbours, so a node whose degree
changed would see a different signal scale for a reason unrelated to wiring. About 90% of edges are
replaced, and the control is redrawn for every repeat.

**Training.** Grouped 5-fold cross-validation on the same identity groups as stages 2 and 3,
5 repeats. Inside each training fold, a further grouped 20% split is held out for early stopping
(patience 20, at most 200 epochs; models stopped after 6 to 23 epochs on average, well short of the
cap). Feature standardization and target z-scoring use training-fold statistics only. **Early
stopping never sees the reported fold**, a deliberate fix of a methodological flaw in the earlier
GraphBRAIN work, where the best epoch was chosen on the same fold that was then reported.

![Stage 4 headline](outputs/f_stage4_headline.png)

**Results (grouped CV, 16 comparisons, Holm-corrected together).**

- **The 3D wiring adds almost nothing.** Of the eight real-against-rewired tests, exactly one
  survives correction: `gnn-esm2` on PSR (+0.09, adjusted p = 0.044). The rest are small and not
  significant, and on AC-SINS the rewired graph with basic features actually scored *higher*
  (-0.10, raw p = 0.023, adjusted p = 0.21). A model given the same residues in a randomly rewired
  arrangement does about as well as one given the real structure.
- **The GNN loses to the simpler methods, usually by a lot.** Six of the eight GNN-against-best-earlier
  tests are significantly worse after correction: on HIC, `gnn-esm2` scores 0.29 against 0.53 for
  surface features (-0.25, adjusted p = 0.008); on Fab Tm, 0.23 against 0.47 for AbLang2
  (-0.25, adjusted p = 0.044). The only GNN variant that matches anything is `gnn-esm2` on PSR
  (0.31 against ESM-2's 0.32, a difference of -0.01, not significant).
- **Node features do all the work.** `gnn-esm2` beats `gnn-basic` on every assay (0.29 against 0.21
  on HIC, 0.28 against 0.07 on AC-SINS, 0.31 against 0.07 on PSR, 0.23 against 0.15 on Fab Tm), and
  `gnn-esm2-shuffled` beats `gnn-basic` too. What the model knows about each residue matters far
  more than which residues sit next to it.

**Why this is the expected answer here, and what it is not.** A GNN has thousands of parameters and
137 training graphs. Ridge regression on a frozen pretrained embedding has one hyperparameter and
inherits what a language model learned from tens of millions of sequences. The same pattern is
documented for brain connectomes: Han et al. (2025, "Rethinking Functional Brain Connectome
Analysis: Do Graph Deep Learning Models Help?", arXiv:2501.17207) find that graph deep learning
often fails to outperform simple baselines there, and a 2025 NeuroGraph follow-up
(arXiv:2503.15902) found GNN accuracy on connectome benchmarks held up even with all edges removed,
which is close to what the rewired control shows here.

This result does **not** say that antibody structure is uninformative. Stage 3 showed the opposite:
surface features computed from the very same structures are the best predictor of HIC by a wide
margin. What it says is narrower and more useful: **at this sample size, with this architecture, a
GNN cannot extract from the contact graph what a few well-chosen structural summaries capture
directly.** What would change the answer: more labelled molecules, pretraining the GNN on a large
unlabelled structure set, or fine-tuning a structure-aware foundation model instead of training
from scratch.

## What the four stages say together

![All stages](outputs/f_all_stages.png)

| Property | Best representation | rho | Why it makes sense |
|---|---|---|---|
| Hydrophobicity (HIC) | Structure surface features | 0.53 | Needs to know which residues are exposed, and how charge and hydrophobicity are arranged |
| Self-interaction (AC-SINS) | AbLang2 embedding | 0.41 | Whole-molecule charge and surface patterns |
| Stability (Fab Tm) | AbLang2 embedding | 0.47 | Folding depends on long-range sequence context |
| Polyreactivity (PSR) | ESM-2 embedding | 0.32 | Several representations reach 0.28 to 0.32; mostly charge |

**No single description of an antibody wins everywhere**, and the simplest adequate representation
usually wins at this sample size. Two findings survive every control: structure-derived surface
features for hydrophobicity, and antibody language model embeddings for stability. The most
expensive model in the project, the graph neural network, wins nothing.

The thread running through all four stages is methodological: identity-grouped splits, simple
baselines reported beside complex models, an ablation when a result had an appealing story
(stage 3), a matched control when a model could be exploiting something other than the intended
signal (stage 4), and correction for multiple comparisons throughout. Everything is in
[`outputs/t_all_stages.csv`](outputs/t_all_stages.csv).

## Status

All four stages are done.

1. ~~Sequences: numbering, CDRs, baseline features, exploration.~~
2. ~~Protein language models (ESM-2, AbLang2) with ridge regression, random vs grouped
   cross-validation.~~
3. ~~Predicted 3D structures, residue graphs and surface features.~~
4. ~~Graph neural network on the residue graphs, with a shuffled-edge control.~~

An optional later part would add an active learning simulation on mutational scanning data: how
many measurements are needed if the model chooses which variants to test next.

## How to run

```bash
brew install hmmer            # or: sudo apt-get install hmmer
conda create -n able python=3.12
conda activate able
pip install -r requirements.txt
pip install -e .

# Only for predicting structures (scripts/45_predict_structures.py):
conda install -c conda-forge pytorch openmm pdbfixer   # one OpenMP runtime for all three
pip install ImmuneBuilder

pytest                        # unit tests, no network needed
bash run_all.sh               # data, numbering, features, embeddings, evaluation, structures
```
