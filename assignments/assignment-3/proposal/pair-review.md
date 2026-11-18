# Preliminary image–headline pair review

**Reviewed:** 18 November 2026 · **Reviewer:** OpenAI Codex visual inspection; student verification pending

## Sample and procedure

Select the first three hash-ranked **training** records in each of the six categories from the committed manifest, giving 18 pairs. No validation/test images were inspected and no model was trained. Retrieve each image using its pinned source metadata, decode the full file with ImageMagick, record its dimensions/format/SHA-256, and compare the visible content with its original headline. Raw headlines, images, and contact sheets remain in ignored `.local/n24news-pair-review/`.

The [per-pair record](pair-review.csv) contains source indices, IDs, byte hashes, technical checks, and paraphrased judgments. The [summary](pair-review-summary.json) records source revision, retrieval date, totals, reviewer, and limitations. IDs and hashes allow another reviewer to inspect the same source pairs without publishing copyrighted images/headlines here.

## Results

All 18 source images were retrieved and decoded successfully. They have 18 distinct byte hashes: 17 JPEG and one PNG, with widths 986–1,050 and heights 516–550 pixels. Do not assume the image format from a URL/file suffix.

| Category | Directly supportive | Contextually related | Unclear | Apparently mismatched |
|---|---:|---:|---:|---:|
| Food | 2 | 1 | 0 | 0 |
| Health | 2 | 1 | 0 | 0 |
| Science | 2 | 1 | 0 | 0 |
| Sports | 2 | 1 | 0 | 0 |
| Technology | 1 | 1 | 1 | 0 |
| Travel | 0 | 3 | 0 | 0 |
| **Total** | **9** | **8** | **1** | **0** |

**Rubric:** directly supportive means a recognizable principal subject matches the headline; contextually related means the broad topic is compatible but the particular claim, person, or place is not established visually; unclear means the visible content is insufficient to judge the association; apparently mismatched means an obvious contradiction or unrelated subject. These are AI-assisted judgments, not authenticated semantic ground truth or an estimate of the full dataset's mismatch rate.

The unclear Technology pair is image ID `191cbc58-22f0-5274-af48-65246b4194e0`: a generic office portrait does not visibly establish the headline's AI/workplace subject. Keep it flagged for student review, rather than declaring it corrupt or silently excluding it. The travel examples show why a scenic image can support a category without verifying the named location or article claim.

Publisher/year overlays are visible in **18/18** sampled images; some images contain additional embedded text. Such artifacts could become shortcuts. Inspect their visibility after the exact prescribed resize/center crop and define any uniform treatment before finalizing the protocol. This review does not establish that a trained model relies on them. No OCR feature or overlay-removal step has been added to the experiment.

## Reproduction and limits

With Python 3, GitHub CLI, and ImageMagick installed, the preparation command is:

```bash
python3 assignments/assignment-3/proposal/review_pairs.py --prepare --cache-dir .local/n24news-pair-review
```

This creates ignored local source files and contact sheets. It makes no permission grant; further retrieval/use remains subject to applicable permissions. To verify committed IDs, classifications, dimensions, and local file hashes without making network requests:

```bash
python3 assignments/assignment-3/proposal/review_pairs.py --cache-dir .local/n24news-pair-review
```

Without `--cache-dir`, verification checks the committed record against the manifest but cannot recheck image bytes or semantic judgments. Independent student review, all 7,500 images' validation, perceptual duplicates, repeated events, and population-level pair quality remain pending. Current [usage-permission findings](usage-review.md) do not clear main experiments.
