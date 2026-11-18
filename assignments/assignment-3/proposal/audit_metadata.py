#!/usr/bin/env python3
"""Reproduce the M1 metadata audit; requires Python 3 and authenticated GitHub CLI.

python3 audit_metadata.py --write   # regenerate evidence
python3 audit_metadata.py          # compare source metadata with committed evidence
No audio/video is downloaded. No training is performed.
"""
import argparse
import collections
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import subprocess

REPOSITORY = "CheyneyComputerScience/CREMA-D"
REVISION = "1658cd342dff90010aa843eaeebd53610a08b1dc"
EXCLUSIONS = {
    "1076_MTI_NEU_XX": "Author reports very short video with no audio.",
    "1076_MTI_SAD_XX": "Author reports error picture, missing audio, very short video.",
    "1064_TIE_SAD_XX": "Author reports video with no duration.",
    "1064_IEO_DIS_MD": "Author reports one-minute compilation, not the named utterance.",
}
LABELS = {"ANG": "anger", "DIS": "disgust", "FEA": "fear", "HAP": "happy", "NEU": "neutral", "SAD": "sad"}
ROOT = Path(__file__).resolve().parent


def fetch(endpoint, raw=False):
    args = ["gh", "api", f"repos/{REPOSITORY}/{endpoint}"]
    if raw:
        args += ["-H", "Accept: application/vnd.github.raw+json"]
    return subprocess.check_output(args)


def csv_text(fields, records):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(records)
    return stream.getvalue()


def generate():
    tree_bytes = fetch(f"git/trees/{REVISION}?recursive=1")
    tree = json.loads(tree_bytes)
    assert not tree["truncated"], "Incomplete upstream tree"
    paths = {x["path"] for x in tree["tree"] if x["type"] == "blob"}
    audio = {Path(p).stem for p in paths if p.startswith("AudioWAV/") and p.endswith(".wav")}
    video = {Path(p).stem for p in paths if p.startswith("VideoFlash/") and p.endswith(".flv")}
    pairs = sorted(audio & video)
    assert len(audio) == len(video) == len(pairs) == 7442
    assert set(EXCLUSIONS) <= set(pairs)
    actors = sorted({p.split("_")[0] for p in pairs})
    assert len(actors) == 91
    random.Random(42).shuffle(actors)
    groups = {"train": sorted(actors[:73]), "validation": sorted(actors[73:82]), "test": sorted(actors[82:])}
    actor_split = {actor: split for split, ids in groups.items() for actor in ids}
    assert len(actor_split) == sum(map(len, groups.values())) == 91
    manifest = []
    for recording in pairs:
        actor, sentence, emotion, intensity = recording.split("_")
        assert emotion in LABELS
        manifest.append(dict(recording_id=recording, actor_id=actor, sentence_code=sentence,
                             emotion_code=emotion, label=LABELS[emotion], intensity_code=intensity,
                             split=actor_split[actor], status="excluded" if recording in EXCLUSIONS else "eligible",
                             audio_path=f"AudioWAV/{recording}.wav", video_path=f"VideoFlash/{recording}.flv"))
    eligible = [r for r in manifest if r["status"] == "eligible"]
    votes_bytes = fetch(f"contents/processedResults/summaryTable.csv?ref={REVISION}", raw=True)
    votes = list(csv.DictReader(io.StringIO(votes_bytes.decode("utf-8-sig"))))
    assert len(votes) == len({r["FileName"] for r in votes}) == 7442
    assert {r["FileName"] for r in votes} == set(pairs)
    votes = [r for r in votes if r["FileName"] not in EXCLUSIONS]
    singles = [r for r in votes if r["VoiceVote"] in set("ADFHNS") and r["FaceVote"] in set("ADFHNS")]
    class_counts = collections.Counter(r["emotion_code"] for r in eligible)
    splits = []
    for split, ids in groups.items():
        records = [r for r in eligible if r["split"] == split]
        counts = collections.Counter(r["emotion_code"] for r in records)
        assert set(counts) == set(LABELS)
        splits.append(dict(split=split, actors=len(ids), paired_samples=len(records), **{c: counts[c] for c in LABELS}))
    assert [r["paired_samples"] for r in splits] == [5965, 735, 738]
    quality = {
        "eligible_annotation_rows": len(votes),
        "empty_votes": {c: sum(not r[c].strip() for r in votes) for c in ("VoiceVote", "FaceVote", "MultiModalVote")},
        "single_valued_voice_face_pairs": len(singles),
        "voice_face_disagreements": sum(r["VoiceVote"] != r["FaceVote"] for r in singles),
        "tied_or_other_voice_face_pairs": len(votes) - len(singles),
    }
    audit = {
        "source_repository": f"https://github.com/{REPOSITORY}", "source_commit": REVISION,
        "annotation_file": "processedResults/summaryTable.csv",
        "annotation_sha256": hashlib.sha256(votes_bytes).hexdigest(),
        "evidence_scope": "Upstream file paths and annotations only; media bytes not downloaded or decoded.",
        "original_audio_files": len(audio), "original_video_files": len(video), "matched_recording_ids": len(pairs),
        "audio_without_video_path": len(audio - video), "video_without_audio_path": len(video - audio),
        "actors": len(actors), "excluded_known_defects": len(EXCLUSIONS), "eligible_pairs": len(eligible),
        "seed": 42, "actor_assignment": "Sort all actor IDs, random.Random(42).shuffle, then take 73/9/9 actors; retain assignment after exclusions.",
        "class_counts": dict(sorted(class_counts.items())), "splits": splits, "pair_quality": quality,
        "pending_media_checks": ["LFS payload retrieval", "decoding", "silence", "duration", "synchronization", "content duplicates", "representative playback"],
    }
    evidence = {
        "metadata-audit.json": json.dumps(audit, indent=2) + "\n",
        "actors.json": json.dumps(groups, indent=2) + "\n",
        "class-counts.csv": csv_text(["emotion_code", "label", "paired_samples"], [dict(emotion_code=c, label=LABELS[c], paired_samples=class_counts[c]) for c in LABELS]),
        "split-counts.csv": csv_text(["split", "actors", "paired_samples", *LABELS], splits),
        "exclusions.csv": csv_text(["recording_id", "reason"], [dict(recording_id=k, reason=v) for k, v in sorted(EXCLUSIONS.items())]),
        "manifest.csv": csv_text(list(manifest[0]), manifest),
    }
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Regenerate proposal evidence files")
    args = parser.parse_args()
    evidence = generate()
    for name, content in evidence.items():
        target = ROOT / name
        if args.write:
            target.write_text(content, encoding="utf-8")
        else:
            assert target.read_text(encoding="utf-8") == content, f"Evidence differs: {name}"
    print(f"{'Generated' if args.write else 'Verified'} {len(evidence)} evidence files: 7,438 eligible pairs; actor-disjoint splits 5,965/735/738.")
