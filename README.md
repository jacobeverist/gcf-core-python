# GCF Core Python Client

Python bindings for the [Gnomic Computing Framework (GCF)](https://github.com/jacobeverist/gcf-core-rust) - a high-performance computational neuroscience framework for machine learning based on Hierarchical Temporal Memory (HTM) principles.

## Overview

The Gnomic Computing Framework is a machine learning library inspired by HTM and sparse distributed representations. This Python client provides Pythonic bindings to the underlying Rust implementation, offering high performance with an easy-to-use interface.

## Features

- **High Performance**: Rust-powered core with Python ergonomics
- **Sparse Distributed Representations**: Efficient BitArray with 32x memory compression
- **Encoding**: Scalar and categorical value encoding into SDRs
- **Learning**: Unsupervised (PatternPooler) and supervised (PatternClassifier) learning
- **Temporal Learning**: Context-dependent pattern recognition with anomaly detection
- **Type-Safe**: Complete type stubs for IDE support and mypy compatibility
- **Well-Tested**: 179 tests including unit, integration, and property-based tests

## Installation

```bash
pip install gcf-core-python-client
```

### From Source

```bash
# Clone the repository
git clone https://github.com/jacobeverist/gcf-core-python-client.git
cd gcf-core-python-client

# Install with uv
uv sync
uv run maturin develop

# Or with pip
pip install maturin
maturin develop
```

## Quick Start

### Encoding Scalar Values

```python
from gcf_core_python_client.api import create_scalar_encoder

# Create encoder for values 0-100
encoder = create_scalar_encoder(min_value=0.0, max_value=100.0)

# Encode a value
encoder.set_value(42.5)
encoder.execute(learn_flag=False)
encoded = encoder.output()

print(f"Encoded {encoded.num_set()} active bits")
```

### Pattern Classification

```python
from gcf_core_python_client.api import create_scalar_encoder, create_classifier

# Create encoder and classifier
encoder = create_scalar_encoder(min_value=0.0, max_value=100.0)
classifier = create_classifier(num_labels=3, num_statelets=90, active_statelets=10)

# Initialize
encoder_output_size = encoder.num_s() * encoder.num_as()
classifier.init(num_i=encoder_output_size)

# Train on ranges: 0-33 → label 0, 34-66 → label 1, 67-100 → label 2
training_data = [
    (10.0, 0), (20.0, 0), (30.0, 0),
    (40.0, 1), (50.0, 1), (60.0, 1),
    (70.0, 2), (80.0, 2), (90.0, 2),
]

for value, label in training_data:
    encoder.set_value(value)
    encoder.execute(learn_flag=False)
    encoded = encoder.output()

    classifier.set_label(label)
    classifier.execute(encoded, learn_flag=True)

# Predict
encoder.set_value(25.0)
encoder.execute(learn_flag=False)
classifier.compute(encoder.output())

print(f"Predicted label: {classifier.get_predicted_label()}")
print(f"Probabilities: {classifier.get_probabilities()}")
```

## Core Components

### BitArray
High-performance sparse binary array with 32x compression.

```python
from gcf_core_python_client import BitArray

ba = BitArray(1000)
ba.set_acts([10, 20, 30, 40, 50])  # Set active indices
print(f"Active bits: {ba.num_set()}")
```

### Encoders

**ScalarTransformer**: Encodes continuous values with semantic similarity
```python
from gcf_core_python_client.api import create_scalar_encoder

encoder = create_scalar_encoder(min_value=0.0, max_value=100.0, num_segments=20)
```

**DiscreteTransformer**: Encodes categorical values with zero overlap
```python
from gcf_core_python_client.api import create_category_encoder

encoder = create_category_encoder(num_categories=10)
```

### Learning Blocks

**PatternPooler**: Unsupervised learning via competitive winner-take-all
```python
from gcf_core_python_client.api import create_pooler

pooler = create_pooler(num_statelets=200, active_statelets=20)
```

**PatternClassifier**: Supervised multi-class classification
```python
from gcf_core_python_client.api import create_classifier

classifier = create_classifier(num_labels=5, num_statelets=150, active_statelets=15)
```

**ContextLearner**: Temporal pattern recognition with anomaly detection
```python
from gcf_core_python_client.api import create_temporal_learner

learner = create_temporal_learner(num_columns=100)
```

## Testing

Run the comprehensive test suite (179 tests):

```bash
uv run pytest -v
```

Includes:
- Unit tests for all components
- Integration tests for pipelines
- Property-based tests using Hypothesis

## Type Checking

Full mypy compatibility with complete type stubs:

```bash
uv run mypy python/gcf_core_python_client --strict
```

## Development

This project uses:
- **uv** for Python dependency management
- **maturin** for building Rust extensions
- **pytest** for testing
- **ruff** for linting and formatting
- **mypy** for type checking

### Setup

```bash
# Install dependencies
uv sync

# Build the Rust extension
uv run maturin develop

# Run tests
uv run pytest

# Type check
uv run mypy python/gcf_core_python_client --strict
```

## Project Structure

```
gcf-core-python-client/
├── src/              # Rust source code (PyO3 bindings)
├── python/           # Python package code
│   └── gcf_core_python_client/
├── tests/            # Python tests
├── Cargo.toml        # Rust dependencies
└── pyproject.toml    # Python project configuration
```

## License

MIT

## Links

- [GCF Rust Library](https://github.com/jacobeverist/gcf-core-rust)
- [Documentation](https://github.com/jacobeverist/gcf-core-python-client)
