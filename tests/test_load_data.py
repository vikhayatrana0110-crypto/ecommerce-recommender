"""Loader tests run against tiny fixtures written to tmp_path.

Nothing here touches the multi-gigabyte archives in data/raw.
"""
import gzip
import json

import pytest

from src.data.load_data import load_metadata, load_reviews


def write_jsonl_gz(path, records):
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")
    return path


def review(user, asin, rating=5.0, ts=1):
    return {
        "user_id": user, "parent_asin": asin, "rating": rating, "timestamp": ts,
        "title": "t", "text": "body", "helpful_vote": 0,
    }


def test_load_reviews_selects_and_renames_columns(tmp_path):
    path = write_jsonl_gz(tmp_path / "r.jsonl.gz", [review("U1", "A"), review("U2", "B")])
    df = load_reviews(path)
    assert list(df.columns) == ["user_id", "item_id", "rating", "timestamp"]
    assert len(df) == 2


def test_load_reviews_keeps_timestamp_for_the_temporal_split(tmp_path):
    path = write_jsonl_gz(tmp_path / "r.jsonl.gz", [review("U1", "A", ts=1700000000)])
    assert load_reviews(path)["timestamp"].iloc[0] == 1700000000


def test_load_reviews_stops_at_max_records(tmp_path):
    path = write_jsonl_gz(
        tmp_path / "r.jsonl.gz", [review(f"U{i}", "A") for i in range(50)]
    )
    assert len(load_reviews(path, max_records=10)) == 10


def test_load_reviews_rejects_a_file_missing_expected_fields(tmp_path):
    path = write_jsonl_gz(tmp_path / "r.jsonl.gz", [{"foo": 1, "bar": 2}])
    with pytest.raises(KeyError, match="user_id"):
        load_reviews(path)


def test_load_reviews_errors_on_an_empty_file(tmp_path):
    path = write_jsonl_gz(tmp_path / "r.jsonl.gz", [])
    with pytest.raises(ValueError):
        load_reviews(path)


def test_load_reviews_skips_a_malformed_line(tmp_path):
    path = tmp_path / "r.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(json.dumps(review("U1", "A")) + "\n")
        fh.write("{not json at all\n")
        fh.write(json.dumps(review("U2", "B")) + "\n")
    assert len(load_reviews(path)) == 2


def test_load_reviews_survives_a_truncated_archive(tmp_path):
    """The real archives are truncated; we keep the readable prefix."""
    good = tmp_path / "good.gz"
    write_jsonl_gz(good, [review(f"U{i}", "A") for i in range(200)])
    truncated = tmp_path / "trunc.jsonl.gz"
    data = good.read_bytes()
    truncated.write_bytes(data[: int(len(data) * 0.6)])

    df = load_reviews(truncated, max_records=1000)
    assert len(df) > 0          # recovered what precedes the damage
    assert len(df) < 200        # but not the whole file


def meta(asin, title="Product"):
    return {
        "parent_asin": asin, "title": title, "categories": ["Electronics"],
        "main_category": "All Electronics", "description": ["long"], "images": [],
    }


def test_load_metadata_filters_to_requested_items(tmp_path):
    path = write_jsonl_gz(
        tmp_path / "m.jsonl.gz", [meta("A"), meta("B"), meta("C")]
    )
    df = load_metadata(path, filter_items=["A", "C"])
    assert set(df["item_id"]) == {"A", "C"}
    assert "title" in df.columns


def test_load_metadata_stops_once_every_item_is_resolved(tmp_path):
    """Early exit is what keeps this off a full 1.2 GB scan."""
    records = [meta("A")] + [meta(f"X{i}") for i in range(500)]
    path = write_jsonl_gz(tmp_path / "m.jsonl.gz", records)
    df = load_metadata(path, filter_items=["A"])
    assert len(df) == 1


def test_load_metadata_respects_the_line_budget(tmp_path):
    """An unresolvable ASIN must not trigger an unbounded scan."""
    path = write_jsonl_gz(
        tmp_path / "m.jsonl.gz", [meta(f"X{i}") for i in range(500)]
    )
    df = load_metadata(path, filter_items=["MISSING"], max_lines=10)
    assert df.empty


def test_load_metadata_deduplicates_repeated_asins(tmp_path):
    path = write_jsonl_gz(
        tmp_path / "m.jsonl.gz", [meta("A", "first"), meta("A", "second"), meta("B")]
    )
    df = load_metadata(path, filter_items=["A", "B"])
    assert len(df) == 2


def test_load_metadata_returns_empty_frame_when_nothing_requested(tmp_path):
    path = write_jsonl_gz(tmp_path / "m.jsonl.gz", [meta("A")])
    df = load_metadata(path, filter_items=[])
    assert df.empty
    assert "item_id" in df.columns
