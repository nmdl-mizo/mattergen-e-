"""Decode per-atom DOS from the latent vectors carried by generated CIF files.

Installed as the ``mattergen-decode-ados`` console script (see
:mod:`mattergen.scripts.decode_ados`). The logic is split into focused modules:

* :mod:`mattergen.decode_ados.cli`           -- command-line argument parsing
* :mod:`mattergen.decode_ados.cif_site_loop` -- raw ``_atom_site_`` loop parsing
* :mod:`mattergen.decode_ados.cif_latents`   -- reading latent vectors from a CIF directory
* :mod:`mattergen.decode_ados.records`       -- the identity of one decoded atom
* :mod:`mattergen.decode_ados.autoencoder`   -- the convolutional autoencoder definition
* :mod:`mattergen.decode_ados.decoding`      -- checkpoint loading and batched decoding
* :mod:`mattergen.decode_ados.diagnostics`   -- flagging spectra that cannot be physical
* :mod:`mattergen.decode_ados.output`        -- writing the ``.npz`` archive
* :mod:`mattergen.decode_ados.pipeline`      -- ties everything together (``run``)

:mod:`mattergen.decode_ados.cif_site_loop`, :mod:`mattergen.decode_ados.records`,
:mod:`mattergen.decode_ados.diagnostics` and :mod:`mattergen.decode_ados.output` are
dependency-light (standard library plus numpy) so their unit tests stay fast;
the remaining modules pull in pymatgen / torch. Keep this ``__init__`` free of
heavy imports.
"""
