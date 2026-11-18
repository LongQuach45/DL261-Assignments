# M1 evidence and handbook mapping

Review the [Dataset Proposal](dataset.md). **Task:** image–text news classification. **Status:** Pending instructor approval.

## Reproduce the audit

Requirements: Python 3 standard library, GitHub CLI (`gh`), and GitHub API access. Authenticate with `gh auth login` if needed. From the repository root:

```bash
python3 assignments/assignment-3/proposal/audit_metadata.py
```

The script fetches `nytimes_dataset.json` at source commit `047a2bc3f4df0ea738b847a4ac3a6bd6aa2e4106`, verifies its SHA-256, recomputes duplicate components and representative selection, and compares six deterministic evidence files with the committed versions. `--write` regenerates those files. Network/authentication or checksum failures stop the audit.

`--probe-images` additionally makes one HTTP HEAD request per category; with `--write`, it saves separately dated observations. Probe responses can change and are not part of deterministic metadata comparison. No image bodies or article bodies are downloaded; no model is trained.

| Evidence | Contents |
|---|---|
| [metadata-audit.json](metadata-audit.json) | Pinned source, checksum, full/category counts, missing fields, text lengths, deduplication and split policies, verification limits |
| [manifest.csv](manifest.csv) | 7,500 provisional selected pairs: source-row index, image ID, category, component ID, split, status and input hashes |
| [groups.csv](groups.csv) | Decisions for all 18,000 source records: selected, reserve, duplicate exclusion, or conflicting-label exclusion |
| [class-counts.csv](class-counts.csv) | Source/candidate/selected counts and split sizes per category |
| [split-counts.csv](split-counts.csv) | Balanced metadata-selected partition counts |
| [exclusions.csv](exclusions.csv) | Filtered source identifiers, component IDs, and reason/status codes |
| [image-probes.json](image-probes.json) | Six dated HEAD observations; availability evidence only |
| [pair-review.md](pair-review.md), [pair-review.csv](pair-review.csv), [summary](pair-review-summary.json) | Actual decoding and AI-assisted visual inspection of 18 deterministic training pairs; student review pending |
| [usage-review.md](usage-review.md) | Official publisher-term findings; permission not cleared; request draft not sent |

Source row indices are zero-based positions in the pinned JSON array. Group IDs hash sorted member image IDs. Hashes identify URLs/headlines without republishing them. Source URLs are compared exactly; headline grouping uses Unicode NFKC, case folding, and whitespace collapsing.

Known duplicate components are collapsed to one representative; conflicting-label components are excluded. Class selection then takes the lowest 1,250 SHA-256 ranks per category, assigning 1,000/125/125 to train/validation/test. Ties use image ID. Final media validation can change eligibility and requires a recorded manifest revision before training.

## Requirement mapping

Handbook: `handbook-ene-v2.pdf`, revision 14 September 2026, retained locally.

| Handbook requirement | Proposal location / evidence |
|---|---|
| §7.6 M1 modalities, pair validity, split unit, metrics, baselines | Sections 1–6 |
| §26 explicitly listed task | Section 1: image–text classification |
| §27 dataset name/source/version/license | Section 1 and usage review: source/version identified; standalone license absent and prior written consent not supplied; clearance remains unresolved |
| §27 task/input/output, paired samples and annotations | Sections 1–2 |
| §27 preliminary pair quality and distributions | Section 2; metadata/count audit plus decoded 18-pair review with explicit semantic judgments; independent student verification pending |
| §27 split/leakage unit, subset rules | Section 3; component ledger, exclusions, ranked manifest |
| §27 metrics, baselines/models, compute | Sections 5–7 |
| §28.1 genuine pairing; missing/noisy/mismatched inputs | Sections 1–4: original association, missing fields, duplicate groups, decoded sample with one unclear pair; full-dataset validation pending |
| §28.1 keep repeated records of an entity together | Section 3: article/duplicate component; additional identified event groups before freezing final splits |
| §28.2 at least five classes and about 5,000 pairs | Six classes; 7,500 metadata-selected pairs, subject to image validation |
| §29 unimodal A/B, simple/improved fusion, representations/loss/training/ablation | Sections 4–6: planned M2/M3 methods |
| §§30–31 comparison, support/conflict, qualitative/errors, compute, multimodal EDA | Sections 2, 6–7 |
| §2 shared/assignment webpages and available links | Public landing/A3 pages, proposal/evidence/README/disclosure; later deliverables pending |
| §5 AI disclosure fields and locations | Proposal disclosure, source AI log, root log, public summaries |
| Approval before main experiments | Section 8: instructor approval and separate usage-permission prerequisite remain outstanding |

## Outstanding prerequisites

The group must review the proposal, submit through the instructor's designated channel, and obtain/record approval or conditions. The [publisher-term review](usage-review.md) found a prior-written-consent requirement for listed ML/AI uses and no applicable grant. Permission is **not cleared**; obtain applicable permission or replace the dataset. Instructor approval is separate. No request has been sent and repository checks cannot create a grant.

An 18-pair preliminary review now provides actual decoding and semantic observations, with a reproducible sample and hashes; the remaining selected images and population-level duplicates/events still require checks. Publisher/year overlays are flagged for preprocessing review. Metadata-selected counts remain provisional. No model results or M2/M3 deliverables are claimed. The handbook does not require a proposal PDF; the final report PDF is a separate later deliverable.
