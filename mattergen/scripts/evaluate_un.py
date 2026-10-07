#!/usr/bin/env python
"""CLI entry point for evaluating generated CIF structures against a reference set.

The implementation lives in :mod:`mattergen.evaluate_un`; this module only wires
up the command line and the process exit code. It is installed as the
``mattergen-evaluate-un`` console script::

    mattergen-evaluate-un --structures_dir ... --reference_dataset ...

See :mod:`mattergen.evaluate_un.pipeline` for the orchestration and
:mod:`mattergen.evaluate_un.cli` for the available arguments.
"""

from __future__ import annotations

import sys

from mattergen.evaluate_un.pipeline import run


def main() -> int:
    return run()


def _main() -> int:
    try:
        return main()
    except Exception as exc:  # pragma: no cover - simple CLI guard
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(_main())
