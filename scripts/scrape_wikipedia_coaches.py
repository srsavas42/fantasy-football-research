"""Compatibility wrapper for the installed ``ffmodel-coaches`` command."""

from ffmodel.data.wikipedia_coaching import main


if __name__ == "__main__":
    raise SystemExit(main())
