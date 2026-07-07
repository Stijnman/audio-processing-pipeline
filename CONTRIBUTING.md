# Contributing

Thank you for considering a contribution to the Audio Processing Pipeline. This document explains how to get set up, what the conventions are, and how to submit changes.

---

## Getting Started

Fork the repository, clone your fork, and create a branch:

```bash
git clone https://github.com/<your-username>/audio-processing-pipeline.git
cd audio-processing-pipeline
git checkout -b feature/your-feature-name
```

Install the development dependencies:

```bash
pip install -r requirements.txt
pip install pytest ruff
```

---

## Code Style

This project uses [Ruff](https://github.com/astral-sh/ruff) for linting and formatting. Before committing, run:

```bash
ruff check .
ruff format .
```

Key conventions followed throughout the codebase:

- Type hints on all function signatures
- Docstrings on all public functions and classes
- No bare `except:` — always catch a specific exception type
- Logging via the standard `logging` module, never `print()`
- All file paths handled as `pathlib.Path` objects, not raw strings

---

## Testing

Run the test suite with:

```bash
pytest tests/ -v
```

When adding a new feature, include at least one test that covers the happy path and one that covers a failure case (e.g. missing file, invalid format).

---

## Submitting a Pull Request

Before opening a PR, make sure:

1. All existing tests pass (`pytest tests/ -v`)
2. Ruff reports no errors (`ruff check .`)
3. New functionality is covered by tests
4. `CHANGELOG.md` has an entry under `[Unreleased]` describing the change
5. The PR description explains *what* changed and *why*

PRs are squash-merged. Keep the commit history clean — one logical change per PR is preferred.

---

## Reporting Issues

When reporting a bug, please include:

- The command you ran (with flags)
- The input file format and approximate duration
- The full error message and traceback
- Your OS, Python version, and ffmpeg version (`ffmpeg -version`)

Use the issue templates in `.github/ISSUE_TEMPLATE/` when opening a new issue.

---

## Feature Requests

Feature requests are welcome. Please open an issue using the feature request template and describe the use case clearly before starting implementation — this avoids duplicate work and ensures the feature fits the project's direction.
