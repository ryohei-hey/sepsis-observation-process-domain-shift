# Observation-process features and domain shift in sepsis

Code and workflow materials for the manuscript:

"Observation-process features are associated with larger domain shift in sepsis mortality prediction: a cross-database evaluation using MIMIC-IV and eICU-CRD"

## Repository purpose

This repository contains the cohort-extraction notebooks, analysis notebooks, and manuscript-supporting materials used for the study.

## Repository structure

- `notebooks/cohort_extraction/`: cohort construction notebooks for MIMIC-IV and eICU-CRD
- `notebooks/analysis/`: analysis, validation, and table-generation notebooks
- `revision_analyses/`: scripts for the secondary and sensitivity analyses added during peer review (calibration correction, paired difference-in-differences, placebo-count control, count re-encoding, common preprocessing, FiO2 exclusion, subgroup calibration, reverse/bidirectional validation, era-restricted sensitivity)
- `ANALYSIS_PLAN.md`: prespecified primary analysis and the peer-review additions, with the script that implements each
- `docs/`: supplementary workflow notes and publication prep notes
- `config/`: local path templates for users running the code in their own environment
- `data/`: placeholder directory for local data files only; no raw data are distributed here
- `output/`: placeholder directory for local generated outputs only

## Data access

Raw patient-level data are not distributed in this repository.

The study used:

- MIMIC-IV v3.1: https://physionet.org/content/mimiciv/
- eICU Collaborative Research Database: https://physionet.org/content/eicu-crd/

Access to these databases requires appropriate credentialing and agreement with PhysioNet data use requirements.

## Reproducibility overview

1. Obtain approved access to MIMIC-IV and eICU-CRD through PhysioNet.
2. Set up a local Python environment with the packages listed in `requirements.txt`. Versions are pinned to those used to produce the reported results (Python 3.14.2).
3. Copy `config/paths_template.yml` to a local configuration file and update paths for your environment.
4. Run the notebooks in `notebooks/cohort_extraction/` to create the analytic cohorts.
5. Run the notebooks in `notebooks/analysis/` to prepare datasets, fit models, and generate manuscript tables and figures.
6. Run the scripts in `revision_analyses/` for the secondary and sensitivity analyses. Start with `build_cache.py`, which caches the primary predictions consumed by several of the others.

`ANALYSIS_PLAN.md` gives the full reproduction order and maps every analysis to the script that implements it.

## Data correction (2026-07-28)

During peer review an error was identified in the MIMIC-IV FiO2 extraction: the chartevents query
included `itemid 220277`, which records pulse oximetry (SpO2), alongside the correct `itemid 223835`
(inspired oxygen fraction). Because SpO2 values cluster near 95–100%, this inflated the MIMIC-IV
FiO2 values and FiO2 measurement counts.

The query in `notebooks/cohort_extraction/03_mimic_apache3_dataset_creation.ipynb` was corrected to
use `223835` alone, the analysis dataset was regenerated, and every analysis was re-run. All results
in the revised manuscript come from the corrected data. The correction is confined to FiO2-derived
features: the cohorts, outcome rates, development split, and the alveolar-arterial gradient (derived
from blood-gas rather than chartevents data) are unchanged.

## Current status before public release

This staging copy was prepared for public release. Notebook paths were converted to repository-relative paths for sharing, and local data files are excluded from version control.

See:

- `PUBLICATION_CHECKLIST.md`

## Suggested citation

If you use this repository, please cite the associated manuscript when available.
