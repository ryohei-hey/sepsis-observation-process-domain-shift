# Analysis plan

Study: *Observation-process features are associated with larger domain shift in sepsis mortality
prediction: a cross-database evaluation using MIMIC-IV and eICU-CRD*

This document records the analysis plan for the study. No formal protocol was registered in a
prediction-model registry. The elements in **Part A** were specified before the analysis was run
and are the primary analysis; the elements in **Part B** were added during peer review at the
reviewers' request and are reported as secondary or sensitivity analyses.

---

## Part A — Prespecified (primary analysis)

### A1. Design
Retrospective cohort study. Model development in MIMIC-IV; external validation in eICU-CRD, with no
model updating or recalibration on the external data, so that raw transportability is quantified.

### A2. Population
Adult ICU stays meeting Sepsis-3 criteria; each patient's first sepsis ICU admission; age ≥ 18
years; ICU length of stay ≥ 24 hours (the models are therefore 24-hour landmark models).

### A3. Outcome
In-hospital mortality, taken from the discharge-disposition field in both databases.

### A4. Predictors
Restricted to variables aligned with the APACHE III framework and available in both databases, so
that specifications differ only in how the physiologic time series is summarised and whether
observation-process features are added.

### A5. The seven specifications (prespecified)
Physiologic summary strategy × presence of the measurement-count block:

| Model | Physiologic summary | Measurement counts |
|---|---|---|
| 1 | APACHE III score only | no |
| 2 | Most recent (latest) values | no |
| 3 | Most recent (latest) values | **yes** |
| 4 | Minimum / maximum | no |
| 5 | Minimum / maximum | **yes** |
| 6 | Within-window variability (range) | no |
| 7 | Within-window variability (range) | **yes** |

The three physiologic strategies are alternative representations of the same time series and are
not strictly nested. The paired contrasts **3 vs 2, 5 vs 4, and 7 vs 6** were prespecified as the
means of isolating the incremental contribution of the observation-process features.

### A6. Algorithms
L2-penalised logistic regression and tuned XGBoost, each fit to all seven specifications.
Preprocessing (imputation, scaling, one-hot encoding) fit on the training fold only and applied
unchanged to the internal-test and external sets.

### A7. Data partitioning
Stratified 60/40 split of the MIMIC-IV cohort (development / internal validation), fixed random
seed 42. eICU-CRD used in full as the external set.

### A8. Performance measures
- Discrimination: AUROC and AUPRC (primary), with nonparametric percentile bootstrap 95% CIs.
- Calibration: calibration curves, Brier score, calibration slope and calibration intercept.
- Domain shift: Δ = external − internal for each measure.
- Apparent optimism: training minus internal-validation AUROC.

### A9. Subgroups
Descriptive performance by race and ethnicity.

---

## Part B — Added during peer review (secondary / sensitivity)

Each item names the reviewer comment that prompted it. Scripts are in `revision_analyses/`.

| # | Analysis | Prompted by | Script |
|---|---|---|---|
| B1 | True calibration-in-the-large via an offset model; calibration slope by unpenalised maximum likelihood; intercept-only versus slope-plus-intercept recalibration | R3 Major 1, R3 Major 2, R1 Comment 5 | `wp_a_calibration.py` |
| B2 | Paired difference-in-differences for the incremental effect of counts (common bootstrap resample indices across the two models in each contrast; B = 2000; two-sided α = 0.05) | R3 Major 3 | `wp_b_paired_inference.py` |
| B3 | Prevalence-adjusted AUPRC (normalised AUPRC and an equalised-prevalence subsample) | R4 Major 4 | `wp_b_paired_inference.py` |
| B4 | Placebo-count control (permuted and random-noise count blocks of equal dimensionality) and a nested ablation ladder anchored on latest values | R1 Comment 4, R3 Major 4 | `wp_c_placebo_ablation.py` |
| B5 | Count re-encoding (binary measured/not-measured, log counts, winsorised counts) and vital-only versus laboratory-only count blocks | R1 Comment 3, R3 Minor 6 | `wp_d_count_encoding.py` |
| B6 | Common-preprocessing sensitivity (logistic regression refit with median imputation to match the XGBoost pipeline) | R1 Comment 6, R4 Minor 3 | `wp_e_common_preproc.py` |
| B7 | FiO2 and A-a gradient excluded from every specification | R4 Major 2 | `wp_f_fio2_excluded.py` |
| B8 | Subgroup calibration (calibration-in-the-large and slope) by race and ethnicity at five levels, sex, age band, hospital teaching status, and region | R3 Major 6, R3 Minor 5, R4 Major 3 | `wp_h_subgroup_calibration.py` |
| B9 | Reverse and bidirectional validation (development in eICU-CRD, validation in MIMIC-IV) | R3 Major 5, R4 Major 1 | `wp_i_reverse.py` |
| B10 | Era-restricted sensitivity (MIMIC-IV development limited to the eICU-contemporaneous `anchor_year_group` bin) | R4 Major 1 | `wp_k_era_restricted.py` |
| B11 | Counts of ICU stays excluded by the 24-hour criterion, with mortality among the excluded | R1 Comment 1, R3 Minor 12 | `sql/wp_j_*.sql` |

---

## Data correction during revision

While investigating Reviewer 4's Major Comment 2 on the cross-database FiO2 difference, an error was
found in the MIMIC-IV FiO2 extraction: the chartevents query included `itemid 220277`, which records
pulse oximetry (SpO2), alongside the correct `itemid 223835` (inspired oxygen fraction). The query
was corrected to use `223835` alone
(`notebooks/cohort_extraction/03_mimic_apache3_dataset_creation.ipynb`), the analysis dataset was
regenerated, and **every analysis in Parts A and B was re-run**. All results reported in the revised
manuscript and Supporting Information come from the corrected data.

The correction is confined to FiO2-derived features. The cohorts (30,218 and 31,403), the outcome
rates (16.3% and 13.9%), the 60/40 split (18,130 and 12,088), and the alveolar-arterial gradient
(derived from blood-gas rather than chartevents data) are unchanged.

---

## Reproduction order

```
notebooks/cohort_extraction/   01 → 02 → 03   (MIMIC-IV)      # BigQuery, requires credentialing
notebooks/cohort_extraction/   04 → 05 → 06   (eICU-CRD)
notebooks/analysis/01_data_preparation.ipynb                  # long-format pickle
notebooks/analysis/02_table1.ipynb                            # baseline table
notebooks/analysis/03_derivation_and_external_validation.ipynb# wide parquet, primary results, figures
revision_analyses/build_cache.py                              # cached predictions for B1–B3, B8
revision_analyses/wp_*.py                                     # secondary analyses (any order)
revision_analyses/make_figures.py                             # response-letter figures
```

Dependency versions are pinned in `requirements.txt` (Python 3.14.2).
