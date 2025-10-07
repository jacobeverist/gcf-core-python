# Contributing to GCF Core Python Client

Thank you for your interest in contributing to the GCF Core Python Client!

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/jacobeverist/gcf-core-python.git
   cd gcf-core-python
   ```

2. **Install dependencies**
   ```bash
   uv sync
   ```

3. **Build the extension**
   ```bash
   uv run maturin develop
   ```

4. **Run tests**
   ```bash
   uv run pytest -v
   ```

## Development Workflow

### Making Changes

1. Create a new branch for your feature or bugfix
   ```bash
   git checkout -b feature/my-feature
   ```

2. Make your changes in the appropriate files:
   - **Rust bindings**: `src/*.rs`
   - **Python code**: `python/gnomics/*.py`
   - **Tests**: `tests/test_*.py`
   - **Type stubs**: `python/gnomics/core.pyi`

3. Rebuild after Rust changes
   ```bash
   uv run maturin develop
   ```

### Testing

Run the full test suite:
```bash
uv run pytest -v
```

Run specific test files:
```bash
uv run pytest tests/test_bitarray.py -v
```

Run with hypothesis statistics:
```bash
uv run pytest tests/test_properties.py --hypothesis-show-statistics
```

### Type Checking

```bash
uv run mypy python/gnomics --strict
```

### Code Quality

Format code:
```bash
uv run ruff format .
```

Lint code:
```bash
uv run ruff check .
```

Fix lint issues:
```bash
uv run ruff check --fix .
```

## Pull Request Process

1. **Update tests**: Add tests for new functionality
2. **Update type stubs**: If adding new methods to Rust classes
3. **Update documentation**: Update README.md if needed
4. **Run all checks**:
   ```bash
   uv run pytest -v
   uv run mypy python/gnomics --strict
   uv run ruff format --check .
   uv run ruff check .
   ```

5. **Submit PR**: Create a pull request with a clear description

## Adding New Rust Bindings

When wrapping a new Rust class:

1. **Create Rust wrapper** in `src/`:
   ```rust
   #[pyclass(name = "MyClass", module = "gnomics", unsendable)]
   pub struct PyMyClass {
       inner: RustMyClass,
   }
   ```

2. **Register in `src/lib.rs`**:
   ```rust
   mod my_class;
   use my_class::PyMyClass;

   m.add_class::<PyMyClass>()?;
   ```

3. **Add type stubs** in `python/gnomics/core.pyi`

4. **Export** in `python/gnomics/__init__.py`

5. **Write tests** in `tests/test_my_class.py`

6. **Add factory function** (optional) in `python/gnomics/api.py`

## Testing Guidelines

- **Unit tests**: Test individual components in isolation
- **Integration tests**: Test pipelines and component interactions
- **Property-based tests**: Use Hypothesis for testing invariants
- **Type tests**: Ensure mypy passes in strict mode

## Documentation

- Add docstrings to all public methods
- Update README.md for significant features
- Include code examples in docstrings
- Keep type stubs synchronized with implementations

## Code Style

- **Python**: Follow PEP 8, enforced by Ruff
- **Rust**: Follow standard Rust conventions
- **Line length**: 100 characters max
- **Type hints**: Required for all Python code

## Questions?

Feel free to open an issue for questions or discussions!

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
