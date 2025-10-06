# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`gcf-core-python-client` is a Python client library project using Python 3.13+. The project uses `uv` for dependency management and virtual environment handling.

## Goal

This python project is to be a wrapper around the Rust library found in gcf-core-rust located at https://github.com/jacobeverist/gcf-core-rust

The goal is to treat the underlying Rust library as a python package, and to expose the functionality of the Rust library in a pythonic way.



## Development Environment Setup

The project uses `uv` for package management. To set up the development environment:

```bash
# Install dependencies (when they exist)
uv sync

# Activate the virtual environment
source .venv/bin/activate
```

## Common Commands

### Building

```bash
# Build the Rust extension in development mode
uv run maturin develop

# Build release wheels
uv run maturin build --release
```

### Testing

```bash
# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_specific.py

# Run tests with verbose output
uv run pytest -v

# Run tests with coverage
uv run pytest --cov=gcf_core_python_client
```

### Linting and Formatting

```bash
# Check code with ruff
uv run ruff check .

# Format code with ruff
uv run ruff format .

# Type check with mypy
uv run mypy python/
```

## Project Configuration

- **pyproject.toml**: Main project configuration file defining dependencies, build settings, and tool configurations
- **Python version**: Requires Python 3.13+
- **Virtual environment**: Managed by `uv` in `.venv/`

## Architecture Notes

This is a client library project. When developing:

- Follow the typical Python package structure with source code in a dedicated package directory (e.g., `src/gcf_core_python_client/` or `gcf_core_python_client/`)
- Separate client implementation, models/schemas, and utilities into distinct modules
- Include comprehensive type hints for all public APIs
- Write unit tests in a `tests/` directory mirroring the source structure
