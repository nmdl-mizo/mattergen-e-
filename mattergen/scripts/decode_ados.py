#!/usr/bin/env python
"""CLI entry point for decoding the per-atom DOS of generated CIF structures.

The implementation lives in :mod:`mattergen.decode_ados`; this module only wires
up the command line and the process exit code. It is installed as the
``mattergen-decode-ados`` console script::

    mattergen-decode-ados --structures_dir ... --autoencoder_path ...

See :mod:`mattergen.decode_ados.pipeline` for the orchestration and
:mod:`mattergen.decode_ados.cli` for the available arguments.
"""

from __future__ import annotations

import sys

from mattergen.decode_ados.pipeline import run


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
