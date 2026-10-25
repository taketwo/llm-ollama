# Show available commands
default:
    @just --list

# Set up development environment
setup:
    uv sync --extra test --extra lint

# Pin ruff, tombi and ty to their latest releases
upgrade-checkers: && setup
    uv remove --optional lint --no-sync ruff tombi ty
    uv add --optional lint --bounds exact --no-sync ruff tombi ty

# Run all checks
check: check-format check-lint check-type

# Run type checking
check-type:
    .venv/bin/ty check llm_ollama

# Verify formatting
check-format: check-format-ruff check-format-tombi

# Verify Python formatting
check-format-ruff:
    .venv/bin/ruff format --check .

# Verify pyproject.toml formatting
check-format-tombi:
    .venv/bin/tombi format --check pyproject.toml

# Run linting checks
check-lint:
    .venv/bin/ruff check .

# Auto-fix linting issues and auto-format code
fix: fix-lint fix-format

# Auto-format code
fix-format: fix-format-ruff fix-format-tombi

# Auto-format Python code
fix-format-ruff:
    .venv/bin/ruff format .

# Auto-format pyproject.toml
fix-format-tombi:
    .venv/bin/tombi format pyproject.toml

# Auto-fix linting issues
fix-lint:
    .venv/bin/ruff check --fix .

# Run tests with coverage; extra args are passed through, e.g. just test -k auth
test *ARGS:
    .venv/bin/pytest --cov=llm_ollama --cov-report=term-missing {{ARGS}}

# Bump the version (major, minor or patch) and commit it as a release
release BUMP:
    uv version --bump {{BUMP}} --no-sync
    git commit -m "Release $(uv version --short)" pyproject.toml
