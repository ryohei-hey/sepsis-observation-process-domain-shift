# Publication checklist for GitHub release

- Review all notebooks for hard-coded local paths such as `C:\Users\...` and replace them with relative or config-based paths.
- Confirm that no raw patient-level data files are present in `data/`, `output/`, or notebook outputs.
- Clear notebook outputs if they contain local paths or environment-specific messages.
- Confirm that any required credentials are loaded from a local environment and are not embedded in notebooks.
- Check that filenames and folder names are suitable for public sharing.
- Verify that the `README.md` accurately describes the release contents.
- Create the GitHub repository as `Public`.
- Push only the contents of this `code_repository` folder, not the full Dropbox research directory.
