import hashlib
import json

import pandas as pd

from src.utils.helpers import get_logger, resolve_path

logger = get_logger(__name__)


def cache_key(*parts):
    """Builds a short, stable digest from the config values a frame depends on."""
    payload = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def source_fingerprint(path):
    """Identifies a source archive by size and mtime.

    Config values alone are not enough: replacing an archive on disk leaves
    every config value unchanged, so a cache keyed only on config would serve a
    frame built from the previous file.
    """
    resolved = resolve_path(path)
    if not resolved.exists():
        return (str(path), None, None)
    stat = resolved.stat()
    return (str(path), stat.st_size, int(stat.st_mtime))


def cached_frame(name, builder, cache_dir, key_parts, enabled=True):
    """Returns a DataFrame from the Parquet cache, building and storing it on miss.

    The cache filename embeds a digest of `key_parts`, so changing any config
    value that feeds the frame produces a new file instead of a stale hit.
    """
    if not enabled:
        return builder()

    cache_dir = resolve_path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{name}-{cache_key(*key_parts)}.parquet"

    if path.exists():
        df = pd.read_parquet(path)
        logger.info("Cache hit: %s (%s rows)", path.name, f"{len(df):,}")
        return df

    logger.info("Cache miss: building %s", path.name)
    df = builder()
    df.to_parquet(path, index=False)
    logger.info("Cached %s rows to %s", f"{len(df):,}", path.name)
    return df
