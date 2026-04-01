# Observation-process features and domain shift in sepsis

Code and workflow materials for the manuscript:

"Observation-process features are associated with larger domain shift in sepsis mortality prediction: a cross-database evaluation using MIMIC-IV and eICU-CRD"

## Repository purpose

This repository contains the cohort-extraction notebooks, analysis notebooks, and manuscript-supporting materials used for the study.

## Repository structure

- `notebooks/cohort_extraction/`: cohort construction notebooks for MIMIC-IV and eICU-CRD
- `notebooks/analysis/`: analysis, validation, and table-generation notebooks
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
2. Set up a local Python environment with the packages listed in `requirements.txt`.
3. Copy `config/paths_template.yml` to a local configuration file and update paths for your environment.
4. Run the notebooks in `notebooks/cohort_extraction/` to create the analytic cohorts.
5. Run the notebooks in `notebooks/analysis/` to prepare datasets, fit models, and generate manuscript tables and figures.

## Current status before public release

This staging copy was prepared for public release. Notebook paths were converted to repository-relative paths for sharing, and local data files are excluded from version control.

See:

- `PUBLICATION_CHECKLIST.md`
- `docs/repository_prep_notes.md`

## Suggested citation

If you use this repository, please cite the associated manuscript when available.
