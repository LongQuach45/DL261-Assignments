#!/usr/bin/env python3
"""Prepare an 18-pair preliminary review locally, or verify its committed record.

Requires Python 3, GitHub CLI, and ImageMagick for --prepare.
Raw inputs/previews must stay in the repository's ignored .local directory.
No models are loaded or trained; semantic judgments are recorded separately.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from urllib.request import Request, urlopen
from urllib.parse import urlparse

sys.dont_write_bytecode = True
from audit_metadata import CLASSES, METADATA_SHA256, REPOSITORY, REVISION

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[2]


def sample():
    with (ROOT / "manifest.csv").open() as stream:
        manifest = list(csv.DictReader(stream))
    return [r for c in CLASSES for r in [r for r in manifest if r["category"] == c and r["split"] == "train"][:3]]


def prepare(cache):
    if not cache.is_relative_to(REPO_ROOT / ".local"):
        raise ValueError("Use an ignored .local cache directory; raw inputs must not be published")
    cache.mkdir(parents=True, exist_ok=True)
    source_path = cache / "nytimes_dataset.json"
    if not source_path.exists():
        endpoint = f"repos/{REPOSITORY}/contents/nytimes_dataset.json?ref={REVISION}"
        raw = subprocess.check_output(["gh", "api", endpoint, "-H", "Accept: application/vnd.github.raw+json"])
        assert hashlib.sha256(raw).hexdigest() == METADATA_SHA256
        source_path.write_bytes(raw)
    raw = source_path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == METADATA_SHA256
    source = json.loads(raw)

    def inspect(item):
        index, selected = item
        record = source[int(selected["source_row_index"])]
        assert record["image_id"] == selected["image_id"]
        url = record["image_url"]
        assert urlparse(url).hostname == "static01.nyt.com"
        path = cache / f"{index:02d}-{record['image_id']}.image"
        if not path.exists():
            with urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=30) as response:
                assert response.status == 200
                payload = response.read(20_000_001)
                assert len(payload) <= 20_000_000, "Unexpectedly large source image"
                path.write_bytes(payload)
        subprocess.run(["magick", "-regard-warnings", str(path), "null:"], check=True, capture_output=True)
        image_format, width, height = subprocess.check_output(
            ["magick", "identify", "-format", "%m %w %h", str(path)], text=True).split()
        return dict(sample_index=index, image_id=record["image_id"], source_row_index=selected["source_row_index"],
                    category=selected["category"], split="train", headline=record["headline"],
                    image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), image_format=image_format,
                    width=int(width), height=int(height), decoded=True, local_path=str(path))

    with ThreadPoolExecutor(max_workers=3) as pool:
        observations = list(pool.map(inspect, enumerate(sample(), start=1)))
    local_record = dict(checked_at_utc=datetime.now(timezone.utc).isoformat(), source_commit=REVISION,
                        metadata_sha256=METADATA_SHA256, selection="First three hash-ranked training records per category",
                        observations=observations)
    (cache / "observations.json").write_text(json.dumps(local_record, indent=2) + "\n")
    for start in (0, 9):
        subprocess.run(["magick", "montage", *[r["local_path"] for r in observations[start:start + 9]],
                        "-thumbnail", "360x240", "-tile", "3x3", "-geometry", "360x240+8+8",
                        "-background", "white", str(cache / f"contact-{start // 9 + 1}.png")], check=True)
    for record in observations:
        print(f"{record['sample_index']:02d} [{record['category']}] {record['headline']}")
    print(f"Prepared {len(observations)} decoded pairs; observations and contact sheets are local only: {cache}")


def verify(cache):
    with (ROOT / "pair-review.csv").open() as stream:
        review = list(csv.DictReader(stream))
    expected = sample()
    assert len(review) == len(expected) == 18
    allowed = {"directly_supportive", "contextually_related", "unclear", "apparently_mismatched"}
    for reviewed, selected in zip(review, expected):
        for key in ("image_id", "source_row_index", "category", "split"):
            assert reviewed[key] == selected[key], f"Sample mismatch: {key}"
        assert reviewed["semantic_assessment"] in allowed
        assert reviewed["decoded"] == "true" and int(reviewed["width"]) > 0 and int(reviewed["height"]) > 0
        assert len(reviewed["image_sha256"]) == 64
        if cache:
            local = cache / f"{int(reviewed['sample_index']):02d}-{reviewed['image_id']}.image"
            assert hashlib.sha256(local.read_bytes()).hexdigest() == reviewed["image_sha256"]
    print("Verified 18 reviewed training pairs, three per category; sample IDs match the provisional manifest.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="Fetch/cache sample images and prepare local review sheets")
    parser.add_argument("--cache-dir", type=Path, help="Ignored .local cache; also enables byte checks during verification")
    args = parser.parse_args()
    cache = args.cache_dir.resolve() if args.cache_dir else None
    if args.prepare:
        if not cache:
            parser.error("--prepare requires --cache-dir")
        prepare(cache)
    else:
        verify(cache)
