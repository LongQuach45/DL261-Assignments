# Assignment 3 Dataset Proposal

**Course:** CO3133 — Deep Learning and Its Applications, Semester 261<br>
**Institution:** Ho Chi Minh City University of Technology – VNU-HCM, Faculty of Computer Science and Engineering<br>
**Instructor:** Le Thanh Sach · **Group:** MobyDick<br>
**Milestone:** M1 Proposal · **Due:** 18 November 2026, 23:59 GMT+7<br>
**Status:** Pending instructor approval

| Member | Student ID |
|---|---|
| Le Quoc Binh | 2352117 |
| Quach Kien Long | 2352698 |
| Le Thanh Lam | 2252421 |
| To Nha Vy | 2353360 |

## 1. Dataset and task

| Field | Specification |
|---|---|
| Dataset | N24News [1], six-category subset |
| Source | [Official repository](https://github.com/billywzh717/N24News) and its `nytimes_dataset.json` |
| Version | Snapshot [`047a2bc3f4df0ea738b847a4ac3a6bd6aa2e4106`](https://github.com/billywzh717/N24News/tree/047a2bc3f4df0ea738b847a4ac3a6bd6aa2e4106), dated 29 August 2023; paper published at LREC 2022 |
| Metadata SHA-256 | `cc7a3422e948a84fffcfe2f9f8c6162ec076a45f712188a6e31be382a9150e9c` |
| License / usage status | No standalone dataset license is declared in the inspected official repository. The paper's CC-BY-NC license and third-party mirror labels do not establish permission for the underlying news text/images. Confirm applicable usage permissions before main experiments, separately from instructor approval. |
| Task | Image–text classification, explicitly listed in handbook §26 |
| Input | One original article headline and its associated RGB news image |
| Output | One label: Food, Health, Science, Sports, Technology, or Travel |
| Prediction unit | One article–image pair |
| Annotations | Original `section` label; metadata also contains headline, abstract, caption, image/article URLs, and image ID |
| Formats | UTF-8 JSON metadata; source-image URL references. All six sampled image responses reported JPEG content types. |
| Public artifacts | Article/source-row identifiers, hashes, grouping/selection decisions, aggregate statistics, and audit code; no raw headlines or images |

N24News pairs each news article with its associated image, rather than combining independent text and image datasets. The proposed classifier uses the **headline only** as text. Abstracts, captions, article bodies, URLs, filenames, and category metadata are not model inputs. No article-body scraping or generated descriptions are needed.

The task asks whether an image helps classify short or ambiguous headlines and when the modalities provide conflicting evidence. Source section labels describe editorial categories, not necessarily every topic visible in an image. Results will apply to this six-category subset of one English-language publisher, not all news.

## 2. Preliminary distribution and pair quality

The [audit](metadata-audit.json) reads pinned source metadata. Its published checks do not establish image decoding or semantic pair quality.

| Finding | Verified result |
|---|---:|
| Full metadata records / categories | 61,218 / 24 |
| Records across six selected categories | 18,000 |
| Records per selected category before filtering | 3,000 |
| Empty selected headlines / image URLs | 0 / 0 |
| Distinct selected image URLs | 17,923 |
| Distinct selected raw headlines | 17,607 |
| Headline word length, min / median / max | 1 / 9 / 19 |
| Single-category representatives after duplicate filtering | 17,532 |
| Metadata-selected target | **7,500 pairs** |

Headline lengths above are whitespace-delimited word counts across the 18,000 source records, not tokenizer lengths. Nine source caption fields are empty; captions are not used. No empty image URL means no missing URL reference, not proof that the image is retrievable.

| Category | Source records | Candidate representatives | Selected target |
|---|---:|---:|---:|
| Food | 3,000 | 2,616 | 1,250 |
| Health | 3,000 | 2,997 | 1,250 |
| Science | 3,000 | 2,998 | 1,250 |
| Sports | 3,000 | 2,984 | 1,250 |
| Technology | 3,000 | 2,977 | 1,250 |
| Travel | 3,000 | 2,960 | 1,250 |
| **Total** | **18,000** | **17,532** | **7,500** |

The metadata-selected subset is balanced and exceeds the handbook default of at least five classes and about 5,000 paired samples. Duplicate filtering excludes one conflicting-label component containing two records and removes 466 redundant records from single-category components.

[Six dated HTTP HEAD probes](image-probes.json), one deterministic representative per category, returned HTTP 200 with `image/jpeg`. These observations establish response availability for those six URLs at the recorded time only. They do not establish decoding, semantic alignment, or availability of all selected images.

**Pending image/pair audit:** retrieve selected images after usage-permission confirmation; record download failures, invalid/corrupt files, dimensions/aspect ratios, exact/perceptual duplicates, representative semantic alignment, and repeated-event evidence. Inspect informative, loosely related, and potentially mismatched image–headline pairs. Measure the actual missing-image rate. No complete media audit or experimental result is claimed at M1.

## 3. Subset, split, and leakage prevention

Categories are fixed before experimentation. Selection uses the following deterministic protocol, reproduced by [audit_metadata.py](audit_metadata.py):

1. Form connected components over the 18,000 source records using **exact article URL**, **exact image URL**, or a headline normalized with Unicode NFKC, case folding, and whitespace collapsing.
2. Exclude every component containing conflicting category labels. Within each remaining component, keep the record with the smallest SHA-256 of `"42|" + article_url`; break hash ties by image ID.
3. For each category, sort representatives by that same hash and select the first 1,250.
4. Assign the first 1,000 to training, the next 125 to validation, and the last 125 to test.

| Partition | Pairs per category | Total pairs | Use |
|---|---:|---:|---|
| Training | 1,000 | 6,000 | Fit classification/fusion heads |
| Validation | 125 | 750 | Hyperparameters and checkpoint selection |
| Test | 125 | 750 | Final evaluation after selection |
| **Total** | **1,250** | **7,500** | |

This is a custom balanced split, not the paper's original experimental split. Seed `42` is embedded in the deterministic hash ranking. The [manifest](manifest.csv) records source-row index (zero-based), image ID, category, component ID, partition, hashes, and `selected_pending_media` status. The [group ledger](groups.csv) preserves selection/reserve/exclusion decisions for all 18,000 records; [exclusions](exclusions.csv) retain filtered identifiers and reasons as status codes.

The leakage-prevention unit is the article plus its duplicate component. No selected pair shares an exact article URL, image URL, or normalized headline with another selected pair. Derived images and cached features inherit their article's split. Group any additional confirmed duplicates or identified repeated-event groups within one partition. Metadata matching alone cannot identify all repeated events, syndicated wording, or visually similar images.

Before training, apply objective media exclusions and additional grouping; rerun deterministic selection from the remaining candidate pool and freeze a versioned final manifest. Preserve revision reasons and measured counts. If there are insufficient candidates or a material protocol change, revise the proposal and seek instructor review. Do not alter splits after observing model performance. Smaller subsets are debugging-only; record their size, seed, and purpose.

Use no test data for hyperparameters, early stopping, learned statistics, or checkpoint selection. Published pretrained normalization constants are fixed. Final images, cached features, and checkpoints remain outside Git.

## 4. Preprocessing and alignment

```text
Same article record → headline + associated image → duplicate-group selection/split
                    → tokenizer + RGB preprocessing → paired data loader
                    → text/image encoders → unimodal head or fusion
                    → cross-entropy → optimization → category → evaluation
```

- **Images:** convert to RGB; resize shorter side to 256; center-crop to `224 × 224`; apply ImageNet normalization. Batch shape `(B, 3, 224, 224)`.
- **Headlines:** use the `distilbert/distilbert-base-uncased` tokenizer [3], maximum 64 tokens, padding, truncation, and attention masks. Input IDs and mask shapes are `(B, L)`, with `L ≤ 64`. Measure token lengths and the truncation rate during preparation.
- **Alignment:** resolve both inputs from the same pinned source row/image ID. Store IDs through batching and feature caching; never randomly pair inputs.
- **Controls:** deterministic preprocessing and no augmentation in the core comparison. Missing, unreadable, or invalid raw modalities exclude the pair from the primary benchmark. Do not substitute another article's image or fabricate text.

## 5. Models and training plan

Use PyTorch training primitives, torchvision's ImageNet ResNet-18 [2], and Hugging Face Transformers' DistilBERT [3]. The group will implement heads, pooling, projections, fusion, and evaluation. Keep both pretrained encoders frozen in evaluation mode, including BatchNorm buffers and dropout; train only heads/projections/gates.

| Configuration | Planned representation and head |
|---|---|
| Image only | ResNet-18 pooled 512-dimensional features and six-class linear head |
| Text only | DistilBERT attention-mask-aware mean-pooled 768-dimensional features and six-class linear head |
| Simple fusion | Concatenate the 512 + 768 features; 256-unit ReLU MLP and six-class classifier |
| Improved fusion | Project each modality to 256 dimensions; learn a sigmoid gate over the projected pair; classify the gated representation [4] |
| Gate ablation | Same projection/classifier design as improved fusion; replace the learned gate with equal fixed weights |

The improved representation is `h = g * h_image + (1 - g) * h_text`, with a 256-dimensional learned gate. The ablation fixes `g = 0.5` and retrains the remaining head components. Report capacity differences; adaptive gating is a planned adaptation, not a claimed reproduction of the original GMU experiments.

Train each unimodal head separately with cross-entropy. Then train each fusion head independently on the same frozen encoder features and source category labels. Each configuration's overall loss is its supervised cross-entropy; no auxiliary or joint encoder loss is used. Backbone fine-tuning is outside the core scope.

Initial settings: AdamW, learning rate `1e-3`, weight decay `1e-4`, batch size 32, and up to 20 epochs per configuration. Reset Python, NumPy, CPU/CUDA and data-loader seeds to `42` before each configuration. Select highest validation macro-F1, breaking ties by lower validation loss and then earlier epoch. Fix split, pretrained weights, preprocessing, seed, optimizer, epoch budget, and checkpoint rule across fusion comparisons; document validation-selected changes.

## 6. Experiments, evaluation, and error analysis

**Primary metrics:** accuracy and macro-F1. Select by validation macro-F1; evaluate selected configurations once on test after model selection. Report per-class precision/recall/F1, confusion matrices, and learning curves.

| Experiment | Hypothesis / changed factor | Fixed factors | Decision criterion |
|---|---|---|---|
| Image vs. text | Visual and headline inputs differ in predictive strength; modality and encoder change | Split, labels, seed, metrics, checkpoint rule | Validation macro-F1 and final test metrics with compute cost; not an isolated architecture test |
| Unimodal vs. concatenation | Combining features may help ambiguous headlines; available features/head change | Same frozen encoders, manifests, preprocessing, evaluation | Validation macro-F1; report whether test results agree |
| Concatenation vs. learned gate | Adaptive weighting may help conflicting pairs; fusion design changes | Encoder features, split, seed, optimizer, budget, checkpoint rule | Validation macro-F1, final test metrics and compute; acknowledge capacity differences |
| Learned vs. fixed gate | Input-dependent weighting contributes beyond equal weighting; remove adaptive gate | Projection/classifier design, encoder features, split, seed, schedule | Validation macro-F1, final test metrics, parameter/time differences |

Plan support examples where both modalities agree or one correct modality rescues the other, and conflict cases where unimodal predictions disagree. Examine short/ambiguous headlines, loosely related illustrative images, Science/Technology confusion, overlapping editorial categories, and repeated-event effects. Report counts and representative errors; gate values alone do not prove a causal explanation. Fusion need not outperform either modality.

Diagnostic image/text removal will zero the removed projected representation and bypass its gate contribution, retaining the available modality. Label this an inference intervention, not a missing-modality-trained model or replacement for independent unimodal baselines. With both modalities absent, abstain.

Measure total/trainable parameters, feature-extraction time, head-training time, warmed-up model-only and end-to-end inference latency, and peak allocated GPU memory. End-to-end latency includes image preprocessing, tokenization, both encoders, and the head; report batch size, hardware, timing boundaries, and synchronization.

## 7. Compute and reproducibility

| Item | Initial estimate / required record |
|---|---|
| Hardware | One GPU with approximately 8 GB memory; actual availability not yet confirmed |
| Storage | 10 GB free for selected images, metadata, weights, caches, and checkpoints; measure actual usage after preparation |
| Budget | Five configurations, up to 20 epochs each; batch size 32 |
| Time | Estimated 1–3 GPU-hours including feature extraction; validate with a pilot after approval and usage-permission confirmation |
| Memory | Frozen encoders, deterministic cached features, microbatch extraction if needed |
| Environment | Record actual Python, PyTorch, torchvision, Transformers, tokenizer and dependency versions; pin pretrained model revisions/checksums |
| Traceability | Configs, seeds, final manifest, image hashes, exclusions, logs/run IDs, checkpoint hashes, and repository commit/tag |

To reproduce the preliminary metadata evidence, install Python 3 and GitHub CLI, authenticate with `gh auth login` if needed, and run from the repository root:

```bash
python3 assignments/assignment-3/proposal/audit_metadata.py
```

This fetches only the pinned JSON and verifies deterministic evidence files. `--write` regenerates evidence; `--probe-images` adds six HTTP HEAD observations, saved separately when combined with `--write`. Historical probe observations are not expected to be invariant. See [audit notes and handbook mapping](README.md).

## 8. Approval, limitations, and milestone boundaries

**Pending instructor approval.** Student review, applicable usage-permission confirmation, and the full image/pair audit remain outstanding. Instructor approval does not replace content usage permissions. Main experiments begin only after **Approved** or **Approved with conditions**, applicable permissions are confirmed, and data checks are completed.

The frozen encoders, one seed, single English-language publisher, selected topics, and headline-only input limit generalization. Images may illustrate rather than directly describe a headline; metadata deduplication cannot guarantee event independence or semantic agreement. No model result, completed media audit, submission, or approval is claimed.

- **M1 — 18 November 2026:** Dataset Proposal, preliminary metadata evidence, split/model/compute plan, and AI disclosure for group/instructor review.
- **M2 — 25 November 2026:** image-only, text-only, and simple fusion implementation and quantitative results after prerequisites.
- **M3 — 2 December 2026:** improved fusion, ablation, qualitative/error/compute analysis, final report, slides, video, assignment page, and checkpoints or reconstruction instructions.

All deadlines are 23:59 GMT+7. The handbook requires the proposal record and approval status, not a formatted proposal PDF. Final deliverables remain pending.

## AI Usage Disclosure

OpenAI Codex assisted with N24News research, metadata auditing, proposal/experiment planning, and repository/webpage updates. The earlier audio–visual proposal was superseded at the user's request. Automated source checks do not establish student verification. Quach Kien Long requested the work; final student review remains pending. See the [detailed Assignment 3 record](../source/AI_USAGE.md).

## References

1. Wang, Z., Shan, X., Zhang, X., and Yang, J. (2022). *N24News: A New Dataset for Multimodal News Classification*. LREC, 6768–6775. [Paper and citation](https://aclanthology.org/2022.lrec-1.729/); [pinned source](https://github.com/billywzh717/N24News/tree/047a2bc3f4df0ea738b847a4ac3a6bd6aa2e4106).
2. He, K., Zhang, X., Ren, S., and Sun, J. (2016). *Deep Residual Learning for Image Recognition*. CVPR. [Paper](https://arxiv.org/abs/1512.03385); [torchvision ResNet-18 weights and preprocessing](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html).
3. Sanh, V., Debut, L., Chaumond, J., and Wolf, T. (2019). *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter*. [Paper](https://arxiv.org/abs/1910.01108); [official model/tokenizer card](https://huggingface.co/distilbert/distilbert-base-uncased).
4. Arevalo, J., Solorio, T., Montes-y-Gomez, M., and Gonzalez, F. A. (2017). *Gated Multimodal Units for Information Fusion*. [arXiv:1702.01992](https://arxiv.org/abs/1702.01992).
