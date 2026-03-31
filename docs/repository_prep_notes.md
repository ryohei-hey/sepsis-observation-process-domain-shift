# Repository preparation notes

## Known cleanup items

The following notebook files currently contain local absolute paths from the original working environment and should be edited before public release:

- `notebooks/analysis/01_data_preparation.ipynb`
- `notebooks/analysis/03_delivation_and_external_validation.ipynb`

These paths currently point to local directories under the original Dropbox workspace. They should be replaced with:

- relative paths inside this repository, or
- a local configuration file based on `config/paths_template.yml`

## Recommended cleanup approach

1. Move all local input and output path definitions to one configuration cell at the top of each notebook.
2. Replace absolute paths with repository-relative paths.
3. Remove notebook outputs that print local filesystem locations.
4. Re-run only the minimum cells needed to confirm the notebooks still execute.

## Suggested public repository contents

Include:

- code used for cohort extraction
- code used for analysis and validation
- instructions for reproducing figures and tables
- supporting workflow notes that do not expose sensitive information

Do not include:

- credential files
- raw data extracts
- temporary outputs
- reviewer-only documents
- local machine paths or environment-specific secrets
