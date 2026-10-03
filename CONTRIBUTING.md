# Contributing

Bug reports and focused improvements to UCRA are welcome. Please include the Python version, the
smallest reproducing example, and the expected versus observed behavior.

Before opening a pull request, run:

```bash
uv sync --extra dev
uv run ruff check src tests examples
uv run pytest -q
```

Do not commit model weights, datasets, generated trajectories, credentials, manuscript sources, or
experiment logs. Changes to the acquisition rule or fusion score should include a unit test and a
clear note that they define a new policy variant rather than silently changing the reference method.
