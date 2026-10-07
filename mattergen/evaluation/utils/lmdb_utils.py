# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import pickle
from typing import Any

import lmdb  # type: ignore [import]


def lmdb_open(db_path: str | os.PathLike, readonly: bool = False) -> lmdb.Environment:
    if readonly:
        return lmdb.open(
            str(db_path),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1,
        )
    else:
        return lmdb.open(
            str(db_path),
            map_size=1099511627776 * 2,
            subdir=False,
            meminit=False,
            map_async=True,
        )


def lmdb_read_metadata(db_path: str | os.PathLike, key: str, default=None) -> Any:
    with lmdb_open(db_path, readonly=True) as db:
        with db.begin() as txn:
            result = lmdb_get(txn, key, default=default)
    return result


def lmdb_put(txn: lmdb.Transaction, key: str, value: Any) -> bool:
    """
    Stores a record in a database.

    Args:
        txn: LMDB transaction (use env.begin())
        key: key of the data to be stored.
        value: value of the data to be stored (needs to be picklable).

    Returns:
        True if it was written.
    """
    return txn.put(
        key.encode("ascii"),
        pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL),
    )


class LmdbNotFoundError(Exception):
    pass


def lmdb_get(
    txn: lmdb.Transaction, key: str, default: Any = None, raise_if_missing: bool = True
) -> Any:
    """
    Fetches a record from a database.

    Args:
        txn: LMDB transaction (use env.begin())
        key: key of the data to be fetched.
        default: default value to be used if the record doesn't exist.
        raise_if_missing: raise LmdbNotFoundError if the record doesn't exist
            and no default value was given.

    Returns:
        the value of the retrieved data.
    """
    value = txn.get(key.encode("ascii"))
    if value is None:
        if default is None and raise_if_missing:
            raise LmdbNotFoundError(
                f"Key {key} not found in database but default was not provided."
            )
        return default
    return pickle.loads(value)
