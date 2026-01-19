# LLM Tests Documentation

This directory contains tests that require actual LLM API calls. These tests are **automatically skipped** unless explicitly enabled.

## Test Files

- `test_group_filters.py` - Tests for parameter extraction with categorical features
- `test_whatif_improvements.py` - Tests for what-if parameter modification tool
- `performance_benchmark.py` - Performance comparison between LLM models

## Running LLM Tests

### Prerequisites

Set your Gemini API key:
```bash
export GEMINI_API_KEY="your-api-key-here"
```

### Commands

#### Run LLM tests:
```bash
# Enable and run LLM tests
SYNTHEFY_RUN_LLM_TESTS=true pytest src/synthefy_pkg/app/fm_agent/

# Run specific test file
SYNTHEFY_RUN_LLM_TESTS=true pytest src/synthefy_pkg/app/fm_agent/test_group_filters.py

# Run performance benchmark
python src/synthefy_pkg/app/fm_agent/performance_benchmark.py
```

#### Regular testing:
```bash
# Normal test run - LLM tests automatically skipped
pytest src/synthefy_pkg/app/fm_agent/
```

## How It Works

- Tests check for `SYNTHEFY_RUN_LLM_TESTS=true` environment variable
- Without it, tests are automatically skipped with a clear message
- No configuration files needed - just set the environment variable when you want to run them

## Safety Features

- Tests are **automatically skipped** unless `SYNTHEFY_RUN_LLM_TESTS=true`
- Tests skip if `GEMINI_API_KEY` is not set
- Each test makes 1-3 API calls - plan accordingly for costs

## CI/CD Integration

For continuous integration, add LLM tests as a separate job:

```yaml
# Regular tests (fast, no API calls)
- name: Run Unit Tests
  run: pytest

# LLM tests (slower, requires API key)  
- name: Run LLM Integration Tests
  env:
    GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
    SYNTHEFY_RUN_LLM_TESTS: true
  run: pytest -m llm
``` 