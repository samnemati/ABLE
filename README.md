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

## Status

Stage 1 of 4 is done. Planned next:

1. ~~Sequences: numbering, CDRs, baseline features, exploration.~~
2. Protein language model embeddings (ESM-2) with ridge regression, comparing random and
   sequence-similarity-grouped cross-validation.
3. Predicted 3D structures turned into residue graphs.
4. Graph neural network on structure, sequence and structure combined, full evaluation.

A later part adds an active learning simulation on mutational scanning data.

## How to run

```bash
brew install hmmer            # or: sudo apt-get install hmmer
conda create -n able python=3.12
conda activate able
pip install -r requirements.txt
pip install -e .

pytest                        # unit tests, no network needed
bash run_all.sh               # download data, number, compute features, draw figures
```
