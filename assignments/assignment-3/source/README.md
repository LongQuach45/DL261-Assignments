# Assignment 3 — Multimodal Deep Learning

**Group:** MobyDick · **Milestone:** M1 proposal available · **Approval:** Pending instructor approval

Proposed task: **image–text news classification** using N24News article headlines and their associated images. The six categories are Food, Health, Science, Sports, Technology, and Travel.

The provisional metadata selection contains 7,500 pairs: 6,000 training, 750 validation, and 750 test, with balanced classes. Duplicate article/image/headline components are filtered before deterministic selection. An 18-pair preliminary review is complete; full image validation and student verification remain pending. **Training-use permission is not cleared.**

- [Dataset Proposal](../proposal/dataset.md)
- [Preliminary evidence and handbook mapping](../proposal/README.md)
- [Public assignment page](https://longquach45.github.io/DL261-Assignments/assignments/a3.html)
- [AI Usage Disclosure](AI_USAGE.md)
- [Preliminary visual pair review](../proposal/pair-review.md)
- [Usage-permission findings and unsent request draft](../proposal/usage-review.md)

## Reproduce M1 evidence

Install Python 3 and GitHub CLI, authenticate with `gh auth login` if needed, then run from the repository root:

```bash
python3 assignments/assignment-3/proposal/audit_metadata.py
```

This verifies pinned public metadata against committed evidence; it does not download image bodies or train models. See the evidence notes for regeneration and optional HEAD probes.

## Implementation and final deliverables

Main experiments await instructor approval, applicable written permission, and full image/pair validation. If permission will not be obtained, replace N24News before main experiments. Training/evaluation code, locked dependencies, actual hardware details, checkpoints or reconstruction instructions, report, slides, and YouTube video are pending. No training commands or measured model results are available at M1.
