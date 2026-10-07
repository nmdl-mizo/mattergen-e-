"""Parse the ``_atom_site_`` loop of a CIF file.

pymatgen drops unknown ``_atom_site_*`` columns when it builds a ``Structure``,
so the latent vectors written by MatterGen-e- are not reachable through
``Structure.site_properties`` and have to be read from the raw CIF text.

This module is standard-library only so it can be unit-tested without pymatgen.
"""

from __future__ import annotations

ATOM_SITE_PREFIX = "_atom_site_"

_QUOTES = ("'", '"')


class CifLoopError(ValueError):
    """Raised when the ``_atom_site_`` loop is missing or malformed."""


def _tokenize(line: str) -> list[str]:
    tokens: list[str] = []
    current: list[str] = []
    quote: str | None = None

    for char in line:
        if quote is not None:
            if char == quote:
                quote = None
            else:
                current.append(char)
        elif char in _QUOTES:
            quote = char
        elif char.isspace():
            if current:
                tokens.append("".join(current))
                current = []
        else:
            current.append(char)

    if current:
        tokens.append("".join(current))
    return tokens


def _is_data_terminator(line: str) -> bool:
    return not line or line.startswith(("_", "#", "loop_", "data_"))


def parse_atom_site_loop(cif_content: str) -> dict[str, list[str]]:
    """Return the ``_atom_site_`` columns of the first matching loop.

    Keys are column names with the ``_atom_site_`` prefix stripped; values are
    the raw, still unconverted cell contents, one per site.
    """
    lines = cif_content.splitlines()
    index = 0

    while index < len(lines):
        if lines[index].strip() != "loop_":
            index += 1
            continue

        index += 1
        headers: list[str] = []
        while index < len(lines) and lines[index].strip().startswith("_"):
            headers.append(lines[index].strip())
            index += 1

        if not any(header.startswith(ATOM_SITE_PREFIX) for header in headers):
            continue

        tokens: list[str] = []
        while index < len(lines) and not _is_data_terminator(lines[index].strip()):
            tokens.extend(_tokenize(lines[index]))
            index += 1

        return _build_columns(headers, tokens)

    raise CifLoopError("No _atom_site_ loop found in the CIF content.")


def _build_columns(headers: list[str], tokens: list[str]) -> dict[str, list[str]]:
    if not tokens:
        raise CifLoopError("The _atom_site_ loop contains no data rows.")
    if len(tokens) % len(headers) != 0:
        raise CifLoopError(
            f"The _atom_site_ loop holds {len(tokens)} values, which is not a "
            f"multiple of its {len(headers)} columns."
        )

    # Rows are recovered by chunking the token stream rather than by reading one
    # line per row: pymatgen wraps a row across several lines once it carries
    # enough columns, which a line-oriented parser would silently mis-split.
    n_rows = len(tokens) // len(headers)
    columns: dict[str, list[str]] = {}
    for position, header in enumerate(headers):
        if not header.startswith(ATOM_SITE_PREFIX):
            continue
        name = header[len(ATOM_SITE_PREFIX) :]
        columns[name] = [tokens[row * len(headers) + position] for row in range(n_rows)]
    return columns
