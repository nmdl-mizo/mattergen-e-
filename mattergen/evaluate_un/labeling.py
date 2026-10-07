"""Compute the ``structure`` summary label for a generated structure.

The label is a compact, human-readable summary built from two independent
boolean signals, concatenated in a fixed order (``U`` then ``N``):

* **U** ("unique") -- the structure is the *representative* of its duplicate
  group, i.e. it is either not duplicated at all (``is_duplicated`` is ``False``)
  or it is the representative chosen for its group
  (``representative_gen_id == gen_id``).
* **N** ("novel")  -- the structure is *not* present in the reference dataset
  (``in_reference`` is ``False``).

Because the two flags are appended independently in a fixed order, the label is
always exactly one of four values::

    ""    neither unique-representative nor novel
    "U"   unique representative, but already in the reference dataset
    "N"   novel, but a non-representative duplicate
    "UN"  unique representative *and* novel

The ``--u_output_dir`` / ``--n_output_dir`` / ``--un_output_dir`` selections are
derived from these labels: a CIF goes to the U directory when its label contains
``"U"`` (``"U"`` or ``"UN"``), to the N directory when its label contains ``"N"``
(``"N"`` or ``"UN"``), and to the UN directory only when it contains *both* (label
``"UN"``). A ``"UN"`` structure therefore goes to all three directories.
"""

from __future__ import annotations

UNIQUE_LABEL = "U"
NOVEL_LABEL = "N"


def is_unique_representative(
    *,
    is_duplicated: bool,
    representative_gen_id: str,
    gen_id: str,
) -> bool:
    """Return ``True`` if this structure represents its duplicate group.

    A structure is a unique representative when it has no duplicates
    (``is_duplicated`` is ``False``) or when it is the representative chosen for
    its duplicate group (``representative_gen_id == gen_id``).
    """
    return (not is_duplicated) or (representative_gen_id == gen_id)


def compute_structure_label(
    *,
    in_reference: bool,
    is_duplicated: bool,
    representative_gen_id: str,
    gen_id: str,
) -> str:
    """Build the ``structure`` column label (one of ``""``, ``"U"``, ``"N"``, ``"UN"``).

    See the module docstring for the meaning of each value.
    """
    label = ""
    if is_unique_representative(
        is_duplicated=is_duplicated,
        representative_gen_id=representative_gen_id,
        gen_id=gen_id,
    ):
        label += UNIQUE_LABEL
    if not in_reference:
        label += NOVEL_LABEL
    return label


def compute_structure_labels(
    results: list[dict],
    representative_gen_id: dict[int, str],
    duplicate_gen_ids: dict[int, list[str]],
) -> list[str]:
    """Compute the ``structure`` label for every result, in ``results`` order.

    This mirrors exactly what :func:`mattergen.evaluate_un.output.save_csv` writes
    into the ``structure`` column, so any selection derived from these labels
    (e.g. the ``--u_output_dir`` / ``--n_output_dir`` / ``--un_output_dir``
    directories) stays consistent with the CSV.

    Args:
        results: per-structure result dicts (each with ``index``, ``gen_id`` and
            ``in_reference``).
        representative_gen_id: idx -> representative gen_id (see
            :func:`mattergen.evaluate_un.metadata.build_duplicate_metadata`).
        duplicate_gen_ids: idx -> list of duplicate gen_ids.

    Returns:
        A list of labels aligned with ``results`` (``labels[i]`` is the label for
        ``results[i]``).
    """
    labels: list[str] = []
    for r in results:
        idx = r["index"]
        labels.append(
            compute_structure_label(
                in_reference=bool(r["in_reference"]),
                is_duplicated=bool(duplicate_gen_ids.get(idx, [])),
                representative_gen_id=representative_gen_id.get(idx, r["id"]),
                gen_id=r["id"],
            )
        )
    return labels
