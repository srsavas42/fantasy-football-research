"""Create the temp root pytest is configured to use, before it needs one.

``pyproject.toml`` sets ``--basetemp=.cache/pytest-tmp``. ``.cache`` is
gitignored, so on a fresh clone that path's parent does not exist, and pytest's
own ``mkdir`` does not create parents — every test taking a ``tmp_path`` fixture
errors with ``FileNotFoundError`` before running. Nine of them do.

This runs at collection, which is early enough for the fixture and avoids
committing a ``.gitkeep`` into a directory whose whole point is being disposable.
"""

from pathlib import Path


def pytest_configure(config):
    basetemp = config.getoption("basetemp", None)
    if basetemp:
        Path(basetemp).mkdir(parents=True, exist_ok=True)
