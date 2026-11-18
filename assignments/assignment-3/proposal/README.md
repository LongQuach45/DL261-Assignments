# M1 evidence and handbook mapping

Review the [Dataset Proposal](dataset.md). Status: **Pending instructor approval**.

## Reproduce the audit

Requirements: Python 3 standard library, GitHub CLI (`gh`), and access to GitHub's API. Authenticate with `gh auth login` if necessary. From the repository root:

```bash
python3 assignments/assignment-3/proposal/audit_metadata.py
```

This fetches the complete upstream tree and `processedResults/summaryTable.csv` at commit `1658cd342dff90010aa843eaeebd53610a08b1dc`. It verifies path/annotation coverage, creates actor-disjoint assignments, removes the four author-documented defective pairs, and compares all six generated evidence files with their committed versions. `--write` regenerates the files. Network/authentication failures stop the audit; they do not validate the evidence.

| Evidence | Contents |
|---|---|
| [metadata-audit.json](metadata-audit.json) | Source revision, annotation SHA-256, counts, split policy, quality statistics, verification limits |
| [actors.json](actors.json) | Exact train/validation/test actor lists |
| [manifest.csv](manifest.csv) | All 7,442 recording IDs, paths, labels, split, and eligibility; four excluded records retained for traceability |
| [class-counts.csv](class-counts.csv) | Eligible per-class counts |
| [split-counts.csv](split-counts.csv) | Eligible counts per partition and class |
| [exclusions.csv](exclusions.csv) | Known defective recording IDs and author-documented reasons |

Filename pairing establishes metadata correspondence only. Raw audio/video bytes were not downloaded or played. Duration, silence, synchronization, decoding, duplicate content, and representative pair inspection remain pending. Crowd-vote disagreement is not a measured media-mismatch rate. The evidence has no training results.

## Requirement mapping

Handbook: `handbook-ene-v2.pdf`, revision 14 September 2026. The source handbook is retained locally, not republished here.

| Handbook requirement | Proposal location / evidence |
|---|---|
| §7.6 M1 modalities, valid pairs, split unit, metrics, baselines | Sections 1–6 |
| §27 name, source, version, license | Section 1; pinned source and license links |
| §27 task, modalities, input/output, paired counts, annotations | Sections 1–2 |
| §27 preliminary pair-quality/distribution analysis | Section 2; audit JSON and class CSV |
| §27 split, leakage unit, metrics, models, compute, subset rules | Sections 3, 5–7 |
| §28.1 genuine related modalities, missing/noisy/mismatched pairs | Sections 1–4; matched recording IDs and pending media audit |
| §28.1 all records of one entity in the same split | Section 3; actor lists and manifest |
| §28.2 at least five classes and about 5,000 pairs | Section 2: six classes, 7,438 eligible metadata pairs |
| §29 unimodal A/B, simple and improved fusion, losses, representations, training, ablation | Sections 4–6: planned for M2/M3 |
| §§30–31 metrics, support/conflict, qualitative/errors, compute, multimodal EDA | Sections 2, 6–7: preliminary evidence and planned experiments |
| §2 shared/assignment pages and linked deliverables as available | Public landing/A3 pages; M1 proposal and disclosure links; later deliverables pending |
| §5 AI disclosure and mandatory fields | Proposal disclosure; `../source/AI_USAGE.md`; course log and public pages |
| Approval before main experiments | Section 8: pending instructor approval; no experiments performed |

## Remaining actions

The group must review the proposal, submit it through the instructor's designated channel, and obtain/record approval or conditions. Submission and approval cannot be established by repository checks. The handbook does not prescribe a proposal PDF; its mandatory final PDF report is a separate later deliverable. Do not mark M2/M3 requirements complete on the basis of this M1 plan.
