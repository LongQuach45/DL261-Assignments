#!/usr/bin/env python3
"""Reproduce N24News M1 evidence using Python 3 and GitHub CLI.

Default: fetch pinned metadata and compare deterministic evidence with committed files.
--write regenerates evidence. --probe-images additionally checks six HTTP HEAD responses.
No image bodies, article bodies, model weights, or training results are downloaded.
"""
import argparse
import collections
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import statistics
import subprocess
import unicodedata
from urllib.parse import urlparse
from urllib.request import Request, urlopen

REPOSITORY = "billywzh717/N24News"
REVISION = "047a2bc3f4df0ea738b847a4ac3a6bd6aa2e4106"
METADATA_SHA256 = "cc7a3422e948a84fffcfe2f9f8c6162ec076a45f712188a6e31be382a9150e9c"
CLASSES = ("Food", "Health", "Science", "Sports", "Technology", "Travel")
ROOT = Path(__file__).resolve().parent


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_headline(value):
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def csv_text(fields, records):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(records)
    return stream.getvalue()


def json_text(value):
    return json.dumps(value, indent=2) + "\n"


def generate():
    endpoint = f"repos/{REPOSITORY}/contents/nytimes_dataset.json?ref={REVISION}"
    raw = subprocess.check_output(["gh", "api", endpoint, "-H", "Accept: application/vnd.github.raw+json"])
    actual_hash = hashlib.sha256(raw).hexdigest()
    assert actual_hash == METADATA_SHA256, "Pinned source metadata checksum differs"
    source = json.loads(raw)
    assert len(source) == 61218
    counts = collections.Counter(r["section"] for r in source)
    assert len(counts) == 24
    rows = [(i, r) for i, r in enumerate(source) if r["section"] in CLASSES]
    assert len(rows) == 18000
    assert all(counts[c] == 3000 for c in CLASSES)
    required = ("headline", "image_url", "article_url", "image_id")
    assert all(isinstance(r.get(k), str) and r[k].strip() for _, r in rows for k in required)
    parents = list(range(len(rows)))

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    seen = {}
    for i, (_, record) in enumerate(rows):
        keys = (("article", record["article_url"]), ("image", record["image_url"]),
                ("headline", normalize_headline(record["headline"])))
        for key in keys:
            if key in seen:
                parents[root(i)] = root(seen[key])
            else:
                seen[key] = i
    components = collections.defaultdict(list)
    for i, row in enumerate(rows):
        components[root(i)].append(row)

    rank = lambda row: (digest("42|" + row[1]["article_url"]), row[1]["image_id"])
    candidates = []
    groups = []
    exclusions = []
    conflicting_components = 0
    for component in components.values():
        group_id = digest("|".join(sorted(record["image_id"] for _, record in component)))
        conflict = len({record["section"] for _, record in component}) != 1
        conflicting_components += int(conflict)
        representative = min(component, key=rank)
        for source_row, record in sorted(component):
            status = "excluded_conflicting_label" if conflict else ("candidate" if source_row == representative[0] else "excluded_duplicate")
            entry = dict(source_row_index=source_row, image_id=record["image_id"], category=record["section"],
                         group_id=group_id, status=status)
            groups.append(entry)
            if status.startswith("excluded"):
                exclusions.append(dict(entry))
            else:
                candidates.append((source_row, record, group_id))
    by_class = {c: sorted((r for r in candidates if r[1]["section"] == c), key=lambda r: rank(r[:2])) for c in CLASSES}
    expected_candidates = dict(Food=2616, Health=2997, Science=2998, Sports=2984, Technology=2977, Travel=2960)
    assert {c: len(v) for c, v in by_class.items()} == expected_candidates
    assert len(candidates) == 17532
    manifest = []
    probe_records = []
    for category in CLASSES:
        assert len(by_class[category]) >= 1250
        probe_records.append(by_class[category][0][1])
        for position, (source_row, record, group_id) in enumerate(by_class[category][:1250]):
            split = "train" if position < 1000 else ("validation" if position < 1125 else "test")
            manifest.append(dict(source_row_index=source_row, image_id=record["image_id"], category=category,
                                 group_id=group_id, split=split, status="selected_pending_media",
                                 ranking_sha256=rank((source_row, record))[0],
                                 article_url_sha256=digest(record["article_url"]),
                                 image_url_sha256=digest(record["image_url"]),
                                 headline_normalized_sha256=digest(normalize_headline(record["headline"]))))
    selected = {r["image_id"]: r["split"] for r in manifest}
    for entry in groups:
        if entry["image_id"] in selected:
            entry["status"] = "selected_pending_media"
        elif entry["status"] == "candidate":
            entry["status"] = "reserve"
    assert len(manifest) == len({r["image_id"] for r in manifest}) == 7500
    for key in ("group_id", "article_url_sha256", "image_url_sha256", "headline_normalized_sha256"):
        assert len({r[key] for r in manifest}) == len(manifest), f"Duplicate selected {key}"
    splits = [dict(split=s, paired_samples=sum(r["split"] == s for r in manifest),
                   **{c: sum(r["split"] == s and r["category"] == c for r in manifest) for c in CLASSES})
              for s in ("train", "validation", "test")]
    assert [r["paired_samples"] for r in splits] == [6000, 750, 750]
    lengths = [len(r["headline"].split()) for _, r in rows]
    audit = {
        "dataset": "N24News", "task": "Image–text news classification",
        "source_repository": f"https://github.com/{REPOSITORY}", "source_commit": REVISION,
        "metadata_file": "nytimes_dataset.json", "metadata_sha256": actual_hash,
        "source_records": len(source), "source_categories": len(counts),
        "source_class_counts": dict(sorted(counts.items())),
        "selected_categories": list(CLASSES), "selected_category_records": len(rows),
        "empty_selected_fields": {k: sum(not r[k].strip() for _, r in rows) for k in source[0]},
        "distinct_selected_image_urls": len({r["image_url"] for _, r in rows}),
        "distinct_selected_headlines": len({r["headline"] for _, r in rows}),
        "headline_word_lengths": dict(min=min(lengths), median=statistics.median(lengths), max=max(lengths)),
        "duplicate_policy": "Connected components by exact article URL, exact image URL, or NFKC/casefold/whitespace-normalized headline.",
        "conflicting_label_components": conflicting_components,
        "conflicting_label_records": sum(r["status"] == "excluded_conflicting_label" for r in exclusions),
        "duplicate_records_removed": sum(r["status"] == "excluded_duplicate" for r in exclusions),
        "candidate_representatives": len(candidates), "candidate_counts": {c: len(by_class[c]) for c in CLASSES},
        "seed": 42, "selection_policy": "Lowest SHA-256 of 42|article_url per non-conflicting component; rank within category by the same hash (image_id breaks ties); first 1250 per category.",
        "split_policy": "Within each selected category's hash order: first 1000 train, next 125 validation, final 125 test.",
        "metadata_selected_pairs": len(manifest), "class_counts": {c: 1250 for c in CLASSES}, "splits": splits,
        "status": "Provisional metadata selection; image integrity, usage permissions, and instructor approval pending.",
        "license": "No standalone dataset license declared in the inspected official repository; paper licensing does not establish rights to underlying news content.",
        "evidence_scope": "Pinned metadata only; HTTP HEAD probes are separately dated observations, not image decoding or pair-quality verification.",
        "pending_checks": ["usage permission confirmation", "instructor approval", "student review", "all selected image availability and decoding", "exact/perceptual image duplicates", "semantic alignment and repeated-event review", "token lengths and truncation"],
    }
    class_rows = [dict(category=c, source_pairs=counts[c], candidate_representatives=len(by_class[c]),
                       metadata_selected_pairs=1250, train=1000, validation=125, test=125) for c in CLASSES]
    groups.sort(key=lambda r: r["source_row_index"])
    exclusions.sort(key=lambda r: r["source_row_index"])
    evidence = {
        "metadata-audit.json": json_text(audit),
        "class-counts.csv": csv_text(list(class_rows[0]), class_rows),
        "split-counts.csv": csv_text(list(splits[0]), splits),
        "exclusions.csv": csv_text(list(groups[0]), exclusions),
        "groups.csv": csv_text(list(groups[0]), groups),
        "manifest.csv": csv_text(list(manifest[0]), manifest),
    }
    return evidence, probe_records


def probe_image(record):
    url = record["image_url"]
    assert urlparse(url).hostname == "static01.nyt.com", "Unexpected source image host"
    result = dict(category=record["section"], image_id=record["image_id"], image_url_sha256=digest(url), method="HEAD")
    try:
        with urlopen(Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"}), timeout=20) as response:
            result.update(status=response.status, content_type=response.headers.get("Content-Type"))
    except Exception as exc:
        result.update(status=None, error=str(exc))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Regenerate deterministic evidence files")
    parser.add_argument("--probe-images", action="store_true", help="Check one source-image HEAD response per category; --write saves dated observations")
    args = parser.parse_args()
    evidence, probe_records = generate()
    for name, content in evidence.items():
        target = ROOT / name
        if args.write:
            target.write_text(content, encoding="utf-8")
        else:
            assert target.read_text(encoding="utf-8") == content, f"Evidence differs: {name}"
    print(f"{'Generated' if args.write else 'Verified'} {len(evidence)} evidence files: 7,500 metadata-selected pairs; splits 6,000/750/750.")
    if args.probe_images:
        with ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(probe_image, probe_records))
        observations = dict(checked_at_utc=datetime.now(timezone.utc).isoformat(), source_commit=REVISION,
                            scope="Six HTTP HEAD observations only; no image bodies decoded.", probes=results)
        if args.write:
            (ROOT / "image-probes.json").write_text(json_text(observations), encoding="utf-8")
        print(json_text(observations), end="")
