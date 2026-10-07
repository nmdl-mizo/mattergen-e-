"""Check whether generated CIF structures are contained in a reference dataset.

Installed as the ``mattergen-evaluate-un`` console script (see
:mod:`mattergen.scripts.evaluate_un`). The logic is split into small, focused
modules so the orchestration stays readable:

* :mod:`mattergen.evaluate_un.cli`          -- command-line argument parsing
* :mod:`mattergen.evaluate_un.data_loading` -- reading CIFs and the reference dataset
* :mod:`mattergen.evaluate_un.membership`   -- novelty / uniqueness evaluation
* :mod:`mattergen.evaluate_un.duplicates`   -- grouping duplicate generated structures
* :mod:`mattergen.evaluate_un.metadata`     -- per-structure duplicate metadata
* :mod:`mattergen.evaluate_un.labeling`     -- the ``structure`` summary label (U / N)
* :mod:`mattergen.evaluate_un.output`       -- writing the CSV and copying unique+novel CIFs
* :mod:`mattergen.evaluate_un.pipeline`     -- ties everything together (``run``)

Only :mod:`mattergen.evaluate_un.metadata`, :mod:`mattergen.evaluate_un.labeling`
and :mod:`mattergen.evaluate_un.output` are dependency-light (standard library
only); the remaining modules pull in pymatgen / mattergen and are therefore slow
to import. Keep this ``__init__`` free of heavy imports so the light-weight
modules (and their unit tests) stay fast.
"""
