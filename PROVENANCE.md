# Data provenance

## Source study

Jain, T., Sun, T., Durand, S., Hall, A., Houston, N. R., Nett, J. H., Sharkey, B., Bobrowicz, B.,
Caffry, I., Yu, Y., Cao, Y., Lynaugh, H., Brown, M., Baruah, H., Gray, L. T., Krauland, E. M.,
Xu, Y., Vásquez, M., & Wittrup, K. D. (2017). Biophysical properties of the clinical-stage antibody
landscape. *Proceedings of the National Academy of Sciences, 114*(5), 944-949.
https://doi.org/10.1073/pnas.1616408114

The study produced 137 antibodies that had reached clinical trials or approval and ran each
through the same 12 biophysical assays. It is the standard public
benchmark for antibody developability.

## File used here

| | |
|---|---|
| Source | Hugging Face dataset `hugging-science/jain-developability-clean`, file `jain.csv` |
| URL | https://huggingface.co/datasets/hugging-science/jain-developability-clean/resolve/main/jain.csv |
| License | MIT (as declared by the dataset) |
| SHA-256 | `67105123c4cf79358aeec43ffb264e660e10bd0d7a72625782669b3c141b0109` |
| Retrieved | 2026-10-27 |

The PNAS supplementary tables are the primary source, but the PNAS site blocks scripted downloads,
so the pipeline uses this redistribution instead. `scripts/10_download.py` fetches it and refuses
to continue if the checksum does not match. The file is not committed to this repository.

## Contents

- 137 antibodies, no duplicate names.
- Heavy-chain (VH) and light-chain (VL) variable-region sequences for every antibody.
- Light chains: 124 kappa, 13 lambda.
- All 12 assays present for every antibody; no missing values.

## Assay caveats

Two assays are not clean continuous measurements, and later stages treat them accordingly.

**SGAC-SINS** reports the ammonium sulfate concentration at which an antibody starts to
self-associate, read in 100 mM steps. Only nine distinct values occur, and the scale is capped at
both ends:

| Value (mM) | 0 | 300 | 400 | 500 | 600 | 700 | 800 | 900 | 1000 |
|---|---|---|---|---|---|---|---|---|---|
| Antibodies | 21 | 5 | 2 | 7 | 11 | 23 | 27 | 27 | 14 |

It is ordinal and censored, so it is excluded from the headline regression results.

**CSI-BLI** (cross-interaction measured by biolayer interferometry) sits at a floor: 79 of 137
antibodies have values between -0.03 and -0.01 nm, and only 10 exceed 0.1 nm. It is
also excluded from the headline results.
