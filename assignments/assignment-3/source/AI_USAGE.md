# AI Usage Disclosure — Assignment 3 (CO3133)

## Record 1: M1 research, metadata audit, and Dataset Proposal

**Historical record — superseded.** This entry records the original CREMA-D proposal published in commit `a09a9c1`. The user subsequently selected the handbook's explicitly listed image–text classification task and N24News; Record 2 describes the replacement. Original files and evidence remain available in Git history. Verification statements in this entry refer to that earlier scope.

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

## Record 2: User-directed switch to N24News image–text classification

- **Tool name and version/model:** OpenAI Codex; exact model/version not recorded.
- **Member who used it:** Quach Kien Long.
- **Time or development stage:** M1 proposal revision, 18 November 2026.
- **Purpose:** Replace the audio–visual task with image–text classification explicitly listed in handbook §26. Research N24News, define a manageable six-category subset, audit duplicates and splits, and update proposal/public records.
- **Affected sections:** Dataset Proposal, requirement mapping, metadata audit and evidence, Assignment 3 README, root and assignment AI disclosures, deliverable metadata, landing/A3/AI pages, and handbook checks.
- **Representative prompt:** “Replace the CREMA-D proposal with N24News image–text classification … six categories—Food, Health, Science, Sports, Technology, and Travel—with a target of 7,500 pairs.” The user selected news classification and the bounded subset, then approved the implementation plan.
- **AI contribution:** Source research; duplicate-component and deterministic-selection design; metadata-audit scripting; headline/image encoder and fusion planning; Markdown and website updates. No image bodies were decoded, model trained, or experimental metric generated.
- **Editing and automated verification:** Checked the handbook and official N24News paper/repository; verified the pinned JSON SHA-256; recomputed source/category counts, missing fields, headline lengths, conflicting-label exclusions, duplicate components, 7,500-pair selection, and disjoint splits. Recorded six dated image HEAD responses separately from deterministic evidence. Compared regenerated evidence with committed files; validated webpage build/content and links during publication.
- **Student verification:** Pending. Metadata checks and sample HTTP availability do not establish full image integrity, semantic pairing, event independence, content permissions, or student approval. No dataset license was inferred from paper or mirror licensing.
- **Member responsible for final verification:** Quach Kien Long; final group review, usage-permission confirmation, and instructor approval remain pending.
- **Verification sources:** Handbook §§7.6 and 26–31; [N24News LREC 2022 paper](https://aclanthology.org/2022.lrec-1.729/); [official source snapshot](https://github.com/billywzh717/N24News/tree/047a2bc3f4df0ea738b847a4ac3a6bd6aa2e4106), `nytimes_dataset.json`, and README; official torchvision ResNet-18 and DistilBERT model/tokenizer documentation; He et al. (2016), Sanh et al. (2019), and Arevalo et al. (2017); committed audit aggregates/manifests and website checks.

## Record 3: Publisher-term investigation and preliminary visual pair review

- **Tool name and version/model:** OpenAI Codex; exact model/version not recorded. ImageMagick supplied image decoding and local contact-sheet preparation.
- **Member who used it:** Quach Kien Long.
- **Time or development stage:** M1 requirements follow-up, 18 November 2026.
- **Purpose:** Investigate the unresolved dataset usage terms and perform actual image–headline inspection beyond metadata/HEAD checks.
- **Affected sections:** Proposal pair-quality, formats, limitations, licensing status, requirement mapping, source README, visual-review evidence/script, usage-review draft, public A3/AI summaries, and compliance checks.
- **Representative prompt:** “Address the 2 points, then push.” The user subsequently confirmed that no written permission is held and none will be sought.
- **AI contribution:** Read the official publisher terms and repository records; prepared an unsent permission-request draft; retrieved and fully decoded the first three selected training images per category; visually compared all 18 with their source headlines; recorded technical checks, hashes, cautious semantic judgments, and source-image overlays.
- **Editing and automated verification:** The pinned source checksum and manifest identify each pair. ImageMagick decoded all 18; 17 were JPEG and one PNG, with distinct byte hashes. Codex viewed local contact sheets alongside the source headlines and recorded nine directly supportive, eight contextually related, and one unclear pair. Source-specific people, places, and article claims were not independently authenticated. Raw images/headlines/contact sheets remain ignored; only identifiers, hashes, aggregates, and paraphrased observations are published. The official NYT terms §§2.1/4.1 were read successfully; no training-use grant was found or supplied. No request was sent and no model was trained.
- **Student verification:** Pending. AI-assisted semantic judgments are not ground-truth labels, and the small sample cannot establish the full dataset's mismatch rate. Record 2 describes the earlier metadata-only stage; actual decoding/visual review occurred in this record.
- **Member responsible for final verification:** Quach Kien Long.
- **Verification sources:** [Official NYT Terms of Service](https://help.nytimes.com/115014893428-Terms-of-Service), [NYT Licensing contact](https://nytlicensing.com/contact/), pinned N24News source/manifest, ignored local review files, and committed `pair-review.csv`, `pair-review-summary.json`, `pair-review.md`, `review_pairs.py`, and `usage-review.md`.

## Record 4: Consolidate M1 public documents

- **Tool/model:** OpenAI Codex; exact model/version not recorded.
- **Member / stage:** Quach Kien Long; M1 documentation cleanup, 18 November 2026.
- **Purpose / prompt:** User requested the same compact structure as Assignment 2: keep the Dataset Proposal, source README, and AI disclosure; move supporting material into the ignored local workspace.
- **Affected sections:** Public proposal/README, website resource links and data loading, compliance checks, and relocation of audit/review files.
- **Editing and verification:** Preserved byte-identical supporting files under `.local/DL261-A3-local/proposal-support/`, including a copy of the prior proposal. Consolidated preliminary statistics, visual judgments, selection rules, sources, and unresolved permissions into `proposal/dataset.md`. Removed active public evidence links and build dependencies on relocated files. Website checks use the committed proposal, so clean-checkout builds do not need private files.
- **Historical records:** File references in Records 1–3 describe artifacts at those stages. Supporting files now remain local and in earlier Git commits; no new public links to them are provided. Original disclosure content is retained.
- **Responsible student reviewer:** Quach Kien Long; student verification remains pending. No experiment, new permission grant, or dataset change occurred.
- **Verification sources:** Prior committed documents, retained local audit/review material, current proposal, link checks, Astro build, and handbook-content checks.
