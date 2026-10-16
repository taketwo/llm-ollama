# llm-ollama - Project guidelines

## Code quality

This project uses ruff with ALL diagnostics enabled (with a few exceptions defined in `pyproject.toml`).

Handling ruff diagnostics:
1. Try to fix diagnostics first — most diagnostics point to real issues or better patterns
2. Don't go crazy — if a diagnostic seems unreasonable for this project, consider ignoring it
3. Ignore hierarchy (in order of preference):
   - Line-level: `# noqa: <code>`
   - File-level: `# ruff: noqa: <code>` at top of file
   - Pattern-based: `lint.per-file-ignores` in `pyproject.toml`
   - Project-wide: `lint.ignore` in `pyproject.toml`
4. ALL ignores require explicit user permission — NEVER add any ignore without asking first
