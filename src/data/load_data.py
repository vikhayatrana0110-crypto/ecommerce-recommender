"""Streaming readers for the Amazon Reviews 2023 archives.

Both plain `.jsonl` and gzip-compressed `.jsonl.gz` are accepted. The files
are far larger than memory and carry heavy nested fields we never use
(`text`, `images`, `description`).
Parsing line by line and keeping only the needed keys avoids materialising
those columns at all, and lets a malformed record be skipped rather than
aborting a multi-gigabyte read.
"""
import gzip
import json
import zlib

import pandas as pd

from src.utils.helpers import get_logger, resolve_path

logger = get_logger(__name__)

REVIEW_FIELDS = ("user_id", "parent_asin", "rating", "timestamp")
METADATA_FIELDS = ("parent_asin", "title", "categories", "main_category")

# A handful of unparseable lines is noise; a burst of them means we have run
# into a corrupt region and everything past it is untrustworthy.
CONSECUTIVE_BAD_LIMIT = 50


def _stream_records(file_path, fields, log_every=1000000):
    """Yields dicts of `fields` from a gzip JSONL file.

    Both archives in this project are truncated, so the reader treats a decode
    failure or a burst of unparseable lines as the end of usable data and stops
    cleanly instead of raising — callers still get every record that precedes
    the damage.
    """
    bad = consecutive_bad = 0
    line_no = 0
    truncated = None

    opener = gzip.open if file_path.suffix == ".gz" else open

    try:
        with opener(file_path, "rt", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line_no += 1
                try:
                    record = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    bad += 1
                    consecutive_bad += 1
                    if consecutive_bad >= CONSECUTIVE_BAD_LIMIT:
                        truncated = (
                            f"{consecutive_bad} consecutive unparseable lines "
                            f"at line {line_no:,}"
                        )
                        break
                    continue

                consecutive_bad = 0
                if log_every and line_no % log_every == 0:
                    logger.info("...read %s lines from %s", f"{line_no:,}", file_path.name)

                yield {f: record.get(f) for f in fields}
    except (EOFError, OSError, zlib.error) as exc:
        truncated = f"{type(exc).__name__} at line {line_no:,}: {exc}"

    if truncated:
        logger.warning(
            "%s is corrupt/truncated - stopped early (%s). "
            "Using the %s records read before the damage.",
            file_path.name, truncated, f"{line_no - bad:,}",
        )
    elif bad:
        logger.warning("Skipped %d malformed line(s) in %s", bad, file_path.name)


def load_reviews(file_path, max_records=100000, chunk_size=None):
    """Streams up to `max_records` reviews into a DataFrame.

    `chunk_size` is accepted for config compatibility but unused: records are
    read one line at a time and the stream stops at `max_records`.
    """
    file_path = resolve_path(file_path)
    rows = []
    for record in _stream_records(file_path, REVIEW_FIELDS):
        rows.append(record)
        if len(rows) >= max_records:
            break

    if not rows:
        raise ValueError(f"No review records read from {file_path}")

    df = pd.DataFrame(rows)
    missing = [c for c in REVIEW_FIELDS if df[c].isna().all()]
    if missing:
        raise KeyError(f"{file_path.name} has no values for expected field(s) {missing}")

    logger.info("Loaded %s raw reviews from %s", f"{len(df):,}", file_path.name)
    return df.rename(columns={"parent_asin": "item_id"})


def load_metadata(file_path, filter_items=None, chunk_size=None, max_chunks=None,
                  max_lines=None):
    """Loads product metadata for the requested ASINs only.

    Stops as soon as every requested ASIN is resolved. `max_lines` bounds the
    scan so a handful of ASINs absent from the archive can't force a read of
    the whole file.
    """
    file_path = resolve_path(file_path)
    filter_set = set(filter_items) if filter_items is not None else None
    if filter_set is not None and not filter_set:
        logger.warning("No items requested; skipping metadata read")
        return pd.DataFrame(columns=["item_id", "title", "categories", "main_category"])

    rows, resolved = [], set()
    for line_no, record in enumerate(_stream_records(file_path, METADATA_FIELDS), start=1):
        asin = record.get("parent_asin")
        if filter_set is not None:
            if asin not in filter_set or asin in resolved:
                if max_lines and line_no >= max_lines:
                    break
                continue
            resolved.add(asin)

        rows.append(record)

        if filter_set is not None and len(resolved) >= len(filter_set):
            logger.info("Resolved all %s items after %s lines",
                        f"{len(filter_set):,}", f"{line_no:,}")
            break
        if max_lines and line_no >= max_lines:
            logger.warning(
                "Metadata scan hit the %s-line budget; %s/%s items resolved",
                f"{max_lines:,}", f"{len(resolved):,}",
                f"{len(filter_set):,}" if filter_set else "?",
            )
            break

    if not rows:
        logger.warning("No metadata matched the requested items")
        return pd.DataFrame(columns=["item_id", "title", "categories", "main_category"])

    df = pd.DataFrame(rows).drop_duplicates(subset="parent_asin")
    logger.info("Loaded metadata for %s items", f"{len(df):,}")
    return df.rename(columns={"parent_asin": "item_id"})
