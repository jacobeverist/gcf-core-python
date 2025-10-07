"""Basic smoke tests to verify the package is working."""

import gnomics


def test_import() -> None:
    """Test that the package can be imported."""
    assert gnomics is not None


def test_version() -> None:
    """Test that the version is available."""
    version = gnomics.__version__
    assert version is not None
    assert isinstance(version, str)
    assert len(version) > 0
