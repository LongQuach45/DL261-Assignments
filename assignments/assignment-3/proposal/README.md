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

Source row indices are zero-based positions in the pinned JSON array. Group IDs hash sorted member image IDs. Hashes identify URLs/headlines without republishing them. Source URLs are compared exactly; headline grouping uses Unicode NFKC, case folding, and whitespace collapsing.

Known duplicate components are collapsed to one representative; conflicting-label components are excluded. Class selection then takes the lowest 1,250 SHA-256 ranks per category, assigning 1,000/125/125 to train/validation/test. Ties use image ID. Final media validation can change eligibility and requires a recorded manifest revision before training.

## Requirement mapping

Handbook: `handbook-ene-v2.pdf`, revision 14 September 2026, retained locally.

| Handbook requirement | Proposal location / evidence |
|---|---|
| §7.6 M1 modalities, pair validity, split unit, metrics, baselines | Sections 1–6 |
| §26 explicitly listed task | Section 1: image–text classification |
| §27 dataset name/source/version/license | Section 1: pinned N24News source; standalone license not declared, permissions unresolved |
| §27 task/input/output, paired samples and annotations | Sections 1–2 |
| §27 preliminary pair quality and distributions | Section 2; metadata audit, class counts, dated availability probes; decoding/alignment pending |
| §27 split/leakage unit, subset rules | Section 3; component ledger, exclusions, ranked manifest |
| §27 metrics, baselines/models, compute | Sections 5–7 |
| §28.1 genuine pairing; missing/noisy/mismatched inputs | Sections 1–4: original article–image association, missing fields, duplicate groups, pending media/pair audit |
| §28.1 keep repeated records of an entity together | Section 3: article/duplicate component; additional identified event groups before freezing final splits |
| §28.2 at least five classes and about 5,000 pairs | Six classes; 7,500 metadata-selected pairs, subject to image validation |
| §29 unimodal A/B, simple/improved fusion, representations/loss/training/ablation | Sections 4–6: planned M2/M3 methods |
| §§30–31 comparison, support/conflict, qualitative/errors, compute, multimodal EDA | Sections 2, 6–7 |
| §2 shared/assignment webpages and available links | Public landing/A3 pages, proposal/evidence/README/disclosure; later deliverables pending |
| §5 AI disclosure fields and locations | Proposal disclosure, source AI log, root log, public summaries |
| Approval before main experiments | Section 8: instructor approval and separate usage-permission prerequisite remain outstanding |

## Outstanding prerequisites

The group must review the proposal, confirm applicable news-content usage permissions, submit through the instructor's designated channel, and obtain/record approval or conditions. The inspected official repository does not declare a standalone dataset license; the paper's license does not establish rights to underlying news content. Repository checks cannot resolve these external actions.

All selected images still require availability, decoding, duplicate-content, and semantic/event checks. Metadata-selected counts are provisional. No model results or M2/M3 deliverables are claimed. The handbook does not require a proposal PDF; the final report PDF is a separate later deliverable.
