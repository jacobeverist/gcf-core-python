# Release Checklist

This document outlines the steps to release a new version of gnomics to PyPI.

## Pre-Release Checks

- [ ] All tests pass (`uv run pytest -v`)
- [ ] Type checking passes (`uv run mypy python/gnomics --strict`)
- [ ] Code is formatted (`uv run ruff format --check .`)
- [ ] Linting passes (`uv run ruff check .`)
- [ ] Documentation is up-to-date
- [ ] CHANGELOG.md is updated with release notes
- [ ] Version number updated in `pyproject.toml`

## Local Build Test

```bash
# Clean previous builds
rm -rf target/wheels dist build

# Build wheel
uv run maturin build --release

# Test wheel installation in clean environment
python -m venv test_env
source test_env/bin/activate  # or `test_env\Scripts\activate` on Windows
pip install target/wheels/gnomics-*.whl
python -c "from gnomics import BitArray; print('Import successful')"
deactivate
rm -rf test_env
```

## Release Process

### Option 1: GitHub Release (Recommended)

1. **Create and push a version tag**
   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```

2. **Create GitHub Release**
   - Go to https://github.com/jacobeverist/gcf-core-python/releases
   - Click "Draft a new release"
   - Select the tag you just created
   - Add release notes
   - Publish release

3. **Automated Process**
   - GitHub Actions will automatically:
     - Build wheels for Linux, macOS, and Windows
     - Build source distribution
     - Publish to PyPI (if configured with trusted publishing)

### Option 2: Manual Release

1. **Build wheels locally**
   ```bash
   # Build for current platform
   uv run maturin build --release

   # Or use docker for manylinux wheels
   docker run --rm -v $(pwd):/io ghcr.io/pyo3/maturin build --release
   ```

2. **Build source distribution**
   ```bash
   uv run maturin sdist
   ```

3. **Upload to PyPI**
   ```bash
   pip install twine
   twine upload target/wheels/*
   ```

## Post-Release

- [ ] Verify package on PyPI: https://pypi.org/project/gnomics/
- [ ] Test installation from PyPI: `pip install gnomics`
- [ ] Update documentation if needed
- [ ] Announce release (if applicable)

## Version Numbering

This project uses Semantic Versioning (SemVer):
- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

Current version: 0.1.0 (Alpha)

## PyPI Trusted Publishing Setup

To enable automated PyPI publishing:

1. Go to PyPI project settings
2. Add GitHub Actions as a trusted publisher
3. Configure:
   - Repository: `jacobeverist/gcf-core-python`
   - Workflow: `release.yml`
   - Environment: `pypi`

## Troubleshooting

### Build Failures

- Ensure Rust toolchain is up to date: `rustup update`
- Clean build artifacts: `cargo clean && rm -rf target`
- Check Cargo.toml dependencies

### Upload Failures

- Verify PyPI credentials
- Check package name availability
- Ensure version number hasn't been used

### Test Failures

- Run tests locally before release
- Check CI logs for platform-specific issues
- Verify all dependencies are compatible
