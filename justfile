# Show available commands
default:
    @just --list

# Set up development environment
setup:
    uv sync --extra test --extra lint

# Run all checks
check: check-format check-lint check-type

# Run type checking
check-type:
    .venv/bin/ty check llm_ollama

# Verify code formatting
check-format:
    .venv/bin/ruff format --check .

# Run linting checks
check-lint:
    .venv/bin/ruff check .

# Auto-fix linting issues and auto-format code
fix: fix-lint fix-format

# Auto-format code
fix-format:
    .venv/bin/ruff format .

# Auto-fix linting issues
fix-lint:
    .venv/bin/ruff check --fix .

# Run tests with coverage; extra args are passed through, e.g. just test -k auth
test *ARGS:
    .venv/bin/pytest --cov=llm_ollama --cov-report=term-missing {{ARGS}}
