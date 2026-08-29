# Contributing to Failo

Thanks for helping out. Failo is small on purpose — the bar for new code is
whether it makes AI calls more reliable without dragging in dependencies.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Checks

All three must pass before a pull request is merged:

```bash
pytest
ruff check .
mypy src
```

The suite must stay fast. Never sleep for real in a test — pass the `sleeper`
argument (see `tests/conftest.py::RecordingSleeper`) and assert on the recorded
delays instead.

## Ground rules

- **No runtime dependencies.** The published package must install with an empty
  dependency list.
- **No provider SDK imports** anywhere in `src/failo/`. Provider support will
  arrive later as optional extras that depend on Failo, not the other way
  around.
- **One retry loop.** `retry.execute_with_retry` is the only place a retry
  decision is made. New entry points delegate to it.
- **Never swallow an error.** No `except Exception: pass`. Every caught
  exception is either classified and re-raised, or recorded in the attempt
  history and acted on.
- **Never retry indefinitely.** Every loop is bounded by `max_attempts`.
- Full type hints and docstrings on public functions.
- Use `time.monotonic()` for latency and `asyncio.sleep()` for delays.

## Adding a classification rule

Do not mutate a global registry. Write a `Callable[[BaseException], AIError | None]`
and compose it:

```python
ErrorClassifier().with_rules(my_rule)
```

Add a test with a fake exception — never a live API call.

## Pull requests

- One focused change per PR.
- Update `CHANGELOG.md` under `[Unreleased]`.
- Tests are required for behavior changes.
