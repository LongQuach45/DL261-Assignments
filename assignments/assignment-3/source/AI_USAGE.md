# AI Usage Disclosure — Assignment 3 (CO3133)

## Record 1: M1 research, metadata audit, and Dataset Proposal

- **Tool name and version/model:** OpenAI Codex; exact model/version not recorded.
- **Member who used it:** Quach Kien Long.
- **Time or development stage:** M1 proposal preparation, 18 November 2026.
- **Purpose:** Research a suitable multimodal dataset; check handbook requirements and source licenses; design actor-disjoint splits, baselines, fusion and ablation experiments; prepare the Markdown proposal and metadata evidence; update public proposal/disclosure links.
- **Affected sections:** Assignment 3 proposal and evidence, M1 source README, course-level AI disclosure, deliverable configuration, and landing/Assignment 3/AI webpages. Assignment 2's optional proposal PDF and its active links were removed with user approval; its approved Markdown record remains.
- **Representative prompt:** “Complete M1 as a concise Markdown proposal for audio–visual emotion classification using CREMA-D, supported by verified source metadata, preliminary distribution analysis, an experiment plan, and AI disclosure.” The user selected CREMA-D and approved the detailed implementation plan.
- **AI contribution:** Source research, proposal drafting, experiment design, a Python metadata-audit script, and documentation/webpage edits. No model was trained; no raw media was decoded or independently inspected.
- **Editing and automated verification:** Compared proposal fields with handbook §§7.6 and 27–31; checked the upstream repository tree and annotation schema at commit `1658cd342dff90010aa843eaeebd53610a08b1dc`; computed class counts, author-documented exclusions, actor lists, splits, and vote disagreement from source metadata. The script checks complete coverage, disjoint actors, class representation, and committed evidence consistency. Website type/build/content checks are recorded with the publication commit and deployment.
- **Student verification:** Pending. Automated checks do not establish that a student has reviewed or approved all content. Media integrity, pilot compute estimates, and model performance are not verified by this M1 metadata audit.
- **Member responsible for final verification:** Quach Kien Long; final group review and instructor approval remain pending.
- **Verification sources:** Course handbook revision 14 September 2026; [pinned CREMA-D documentation](https://github.com/CheyneyComputerScience/CREMA-D/blob/1658cd342dff90010aa843eaeebd53610a08b1dc/README.md), license, repository tree and `processedResults/summaryTable.csv`; Cao et al. (2014), doi:10.1109/TAFFC.2014.2336244; official torchvision ResNet-18 documentation; He et al. (2016); Arevalo et al. (2017), arXiv:1702.01992; committed proposal evidence and website validation checks.

Only the AI use documented above is asserted. Further tools, member contributions, prompts, and verification outcomes should be appended when they occur.
