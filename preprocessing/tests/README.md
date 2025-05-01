# Preprocessing Module Tests

This directory contains unit tests for the preprocessing utilities in the FNO Battery project.

## Test Coverage

Tests are included for the following modules:

- `utils.py`: Data preprocessing utilities including:
  - `create_sequences`: Creating time series sequences from data
  - `create_multi_input_sequences`: Creating sequences for multiple inputs
  - `scale_data`: Scaling data with different scaler types
  - `scale_data_with_existing`: Applying previously fit scalers
  - `split_data`: Splitting data into train/validation/test sets

## Running Tests

### Option 1: Using pytest directly

From the preprocessing directory, run:

```bash
python -m pytest tests/test_utils.py -v
```

### Option 2: Using the run_tests.py script

From the preprocessing directory, run:

```bash
python tests/run_tests.py
```

## Adding New Tests

When adding new tests:

1. Create test files with the naming convention `test_*.py`
2. Group related tests in classes to organize them
3. Follow the existing test structure
4. Add fixtures to `conftest.py` if appropriate

## Testing Guidelines

- Each test function should test a single behavior or functionality
- Use descriptive test names that indicate what's being tested
- Include edge cases and error conditions
- Keep tests independent from each other
- Use fixtures for reusable test data 