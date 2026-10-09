"""Immutable raw ingest manifests and versioned streaming Parquet transitions."""

import hashlib
import hmac
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .schema import SCHEMA_VERSION, Transition, game_split


def anonymize_id(identifier: str, secret: bytes) -> str:
    if len(secret) < 16:
        raise ValueError("Use a dataset secret of at least 16 bytes")
    return hmac.new(secret, identifier.encode(), hashlib.sha256).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_transitions(path: Path, transitions, split_seed: int = 0) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(
        [
            ("episode_id", pa.string()),
            ("decision_id", pa.int64()),
            ("seat", pa.int8()),
            ("split", pa.string()),
            ("payload", pa.large_string()),
        ],
        metadata={b"schema_version": SCHEMA_VERSION.encode()},
    )
    count, rows = 0, []
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with pq.ParquetWriter(temporary, schema, compression="zstd") as writer:
            for item in transitions:
                item.validate()
                rows.append(
                    {
                        "episode_id": item.episode_id,
                        "decision_id": item.decision_id,
                        "seat": item.seat,
                        "split": game_split(item.episode_id, split_seed),
                        "payload": json.dumps(item.to_dict(), separators=(",", ":")),
                    }
                )
                count += 1
                if len(rows) == 1024:
                    writer.write_table(pa.Table.from_pylist(rows, schema=schema))
                    rows = []
            if rows:
                writer.write_table(pa.Table.from_pylist(rows, schema=schema))
        if count == 0:
            raise ValueError("No transitions produced; refusing an empty training dataset")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "transitions": count,
        "sha256": file_hash(path),
        "split_seed": split_seed,
    }
    path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return count


def read_transitions(path: Path, split: str | None = None) -> list[Transition]:
    file = pq.ParquetFile(path)
    if file.schema_arrow.metadata.get(b"schema_version") != SCHEMA_VERSION.encode():
        raise ValueError("Dataset schema version mismatch")
    records = []
    for batch in file.iter_batches(batch_size=1024):
        for row in batch.to_pylist():
            if split is None or row["split"] == split:
                records.append(Transition.from_dict(json.loads(row["payload"])))
    if not records:
        raise ValueError(f"No samples in requested split: {split}")
    return records
