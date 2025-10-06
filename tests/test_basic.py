"""Basic smoke tests to verify the package is working."""

import gcf_core_python_client


def test_import() -> None:
    """Test that the package can be imported."""
    assert gcf_core_python_client is not None


def test_version() -> None:
    """Test that the version is available."""
    version = gcf_core_python_client.__version__
    assert version is not None
    assert isinstance(version, str)
    assert len(version) > 0
