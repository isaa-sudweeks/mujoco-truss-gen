# Releasing

PyPI publishing is automated through the GitHub Actions workflow in
`.github/workflows/publish.yml`. The workflow runs when a GitHub Release is
published, and it can also be started manually from the GitHub Actions tab.

For small test releases, use a pre-release tag such as `0.1.0a1`. For bug fixes
or backwards-compatible changes, use a patch release tag such as `0.1.1`. For
new features or breaking changes, use a minor or major release tag such as
`0.2.0` or `1.0.0`.

## Automated Release Workflow

The workflow:

- Installs the package test dependencies.
- Runs `python -m pytest`.
- Builds the source distribution and wheel with `python -m build`.
- Checks the built distributions with `python -m twine check dist/*`.
- Publishes the distributions to PyPI.

The publish job uses PyPI Trusted Publishing, so the repository does not need a
long-lived PyPI API token in GitHub Secrets. PyPI must be configured to trust
this GitHub Actions workflow before the first automated release. For the
existing `mujoco-truss-gen` PyPI project, add a GitHub Actions trusted publisher
with these settings:

- PyPI project name: `mujoco-truss-gen`
- GitHub repository owner: `isaa-sudweeks`
- GitHub repository name: `mujoco-truss-gen`
- Workflow filename: `publish.yml`
- GitHub environment name: `pypi`

To publish a new version:

1. Update `version` in `pyproject.toml`.
2. Commit and push the change.
3. Create and publish a GitHub Release for that commit.
4. Confirm the `Publish to PyPI` workflow passes.

PyPI versions are immutable. If a release workflow fails after uploading a
version, fix the issue, bump `version` again, and publish a new release.

## Manual Release Checklist

1. Update `version` in `pyproject.toml`.
2. Run `python -m pytest`.
3. Run `python -m ruff check .`.
4. Run `python -m ruff format --check .`.
5. Build distributions with `python -m build`.
6. Upload with `python -m twine upload dist/*`.
7. Verify installation in a clean environment with
   `python -m pip install mujoco-truss-gen`.

Users update to the newest published package with:

```bash
python -m pip install --upgrade mujoco-truss-gen
```
