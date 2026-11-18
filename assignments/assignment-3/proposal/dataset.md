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
| Source/version | [Official repository snapshot](https://github.com/billywzh717/N24News/tree/047a2bc3f4df0ea738b847a4ac3a6bd6aa2e4106), commit `047a2bc3f4df0ea738b847a4ac3a6bd6aa2e4106`, dated 29 August 2023; LREC 2022 paper |
| Metadata | `nytimes_dataset.json`; SHA-256 `cc7a3422e948a84fffcfe2f9f8c6162ec076a45f712188a6e31be382a9150e9c` |
| License / usage status | **Training permission is not cleared.** No standalone dataset license is declared in the inspected official repository. [Publisher terms §§2.1 and 4.1](https://help.nytimes.com/115014893428-Terms-of-Service) require prior written consent for the listed ML/AI development uses; no grant has been supplied. Academic use, the paper's license, and mirror labels do not establish this permission. |
| Task | Image–text classification, explicitly listed in handbook §26 |
| Input/output | One original article headline plus its associated RGB image → Food, Health, Science, Sports, Technology, or Travel |
| Prediction unit | One article–image pair |
| Annotations | Original `section` category; metadata also provides headline, abstract, caption, URLs, and image ID |
| Formats | UTF-8 JSON and source-image URLs; decoded sample includes JPEG and PNG |

The task studies whether images help classify short or ambiguous headlines and when modalities conflict. Both inputs come from the same source article; independent records are never randomly paired. Use **headline-only** text: no generated descriptions, article-body scraping, captions, abstracts, URLs, filenames, or category metadata as model features.

If applicable permission will not be obtained, replace N24News before main experiments. Instructor approval is a separate prerequisite.

## 2. Preliminary distribution and pair quality

The metadata audit uses the pinned source above. Counts are provisional until full image validation.

| Finding | Result |
|---|---|
| Full metadata records / categories | 61,218 / 24 |
| Records across selected categories | 18,000 |
| Empty selected headlines / image URLs | 0 / 0 |
| Distinct image URLs / raw headlines | 17,923 / 17,607 |
| Headline words, min / median / max | 1 / 9 / 19 |
| Candidate representatives after filtering | 17,532 |
| Metadata-selected pairs / categories | 7,500 / 6 |
| Train / validation / test targets | 6,000 / 750 / 750 |
| Decoded sample pairs | 18 |
| Visual sample assessments | 9 directly supportive; 8 contextually related; 1 unclear; 0 apparent mismatches |

| Category | Source records | Candidate representatives | Selected target |
|---|---:|---:|---:|
| Food | 3,000 | 2,616 | 1,250 |
| Health | 3,000 | 2,997 | 1,250 |
| Science | 3,000 | 2,998 | 1,250 |
| Sports | 3,000 | 2,984 | 1,250 |
| Technology | 3,000 | 2,977 | 1,250 |
| Travel | 3,000 | 2,960 | 1,250 |
| **Total** | **18,000** | **17,532** | **7,500** |

The target is balanced and exceeds the default of at least five classes and about 5,000 pairs. Filtering removes one conflicting-label component containing two records and 466 redundant records. Word counts use whitespace across the 18,000 source records; tokenizer lengths remain to be measured. Nine caption fields are empty, but captions are unused. Nonempty image URLs do not establish availability.

On 18 November 2026, six image HEAD probes returned HTTP 200. A subsequent review retrieved and fully decoded the **first three hash-ranked training pairs per category**: 17 JPEG and one PNG, 18 distinct byte hashes, widths 986–1,050 and heights 516–550 pixels. No validation/test images were reviewed.

Codex visually compared all 18 images with their headlines. “Directly supportive” means the principal subject is recognizable; “contextually related” means the broad topic is compatible but the specific claim/person/place is not established; “unclear” means insufficient visual evidence. The unclear Technology pair is a generic office portrait accompanying an AI/workplace headline. It remains flagged for student review, not silently excluded. These AI-assisted judgments are not ground truth or a population mismatch-rate estimate; student verification remains pending.

Publisher/year overlays are visible in all 18 images, and some contain additional text. Inspect artifact visibility after prescribed preprocessing and define any uniform treatment before freezing the protocol. No OCR feature or overlay-removal step has been added.

**Full checks still planned:** image availability/decoding, missing rates, dimensions/aspect ratios, exact/perceptual duplicates, semantic alignment, and repeated events. Supporting scripts, manifests, per-pair observations, hashes, and raw review inputs are retained in the ignored local workspace. No full-dataset integrity result or model result is claimed.

## 3. Subset, split, and leakage prevention

Categories are fixed before experimentation. Apply this deterministic protocol:

1. Form connected components using exact article URL, exact image URL, or headlines normalized with Unicode NFKC, case folding, and whitespace collapsing.
2. Exclude conflicting-label components. Keep the smallest SHA-256 of `"42|" + article_url` in each remaining component, breaking hash ties by image ID.
3. Rank representatives within each category by that same hash; select the first 1,250.
4. Assign the first 1,000 to training, next 125 to validation, and last 125 to test.

| Partition | Pairs per category | Total | Purpose |
|---|---:|---:|---|
| Training | 1,000 | 6,000 | Fit heads and learned statistics |
| Validation | 125 | 750 | Hyperparameters and checkpoints |
| Test | 125 | 750 | Final evaluation after selection |
| **Total** | **1,250** | **7,500** | |

This is a custom balanced split, not the paper's original split. Seed `42` is embedded in the hash rank. The leakage unit is the article and its duplicate component. No selected pair shares an exact article URL, image URL, or normalized headline with another selected pair. Derived images/features inherit their article's partition. Keep additional confirmed duplicates and identified repeated-event groups together; metadata alone cannot detect every repeated event.

After objective image exclusions and additional grouping, regenerate selection from remaining candidates and freeze a versioned final manifest **before training**. Record source-row indices, image IDs, component IDs, partitions, hashes, and exclusion reasons. Revise the proposal if scope changes materially or class availability is insufficient. Never alter splits after seeing model performance. Smaller subsets are debugging-only; record size, seed, and purpose.

Do not use test data for tuning, early stopping, learned statistics, or checkpoint selection. Final implementation will publish the split/configuration and reconstruction records needed to trace reported results.

## 4. Preprocessing and alignment

```text
Same article → headline + image → duplicate-group selection/split
             → tokenizer + RGB preprocessing → paired data loader
             → encoders → unimodal head or fusion → cross-entropy
             → optimization → category → evaluation
```

- **Image:** RGB; shorter side resized to 256; center crop 224×224; ImageNet normalization. Batch: `(B, 3, 224, 224)`.
- **Text:** DistilBERT uncased tokenizer [3]; at most 64 tokens; padding, truncation, and attention mask. IDs/mask: `(B, L)`, `L ≤ 64`. Measure token lengths and truncation rate.
- **Alignment:** resolve both modalities from the same pinned source row/image ID; preserve IDs through batching and feature caching.
- **Controls:** deterministic preprocessing, no augmentation in the core comparison. Missing/corrupt modalities exclude the pair; never substitute unrelated data.

## 5. Models and training plan

PyTorch provides training primitives, torchvision provides ImageNet ResNet-18 [2], and Transformers provides DistilBERT [3]. The group will implement pooling, heads, projections, fusion, and evaluation. Both encoders remain frozen in evaluation mode, including BatchNorm buffers/dropout; backbone fine-tuning is outside the core scope.

| Configuration | Planned method |
|---|---|
| Image only | ResNet-18 pooled 512-dimensional features; six-class linear head |
| Text only | Attention-mask-aware mean-pooled 768-dimensional DistilBERT features; six-class linear head |
| Simple fusion | Intermediate fusion: concatenate features; 256-unit ReLU MLP and classifier |
| Improved fusion | Intermediate fusion: project each modality to 256 dimensions; learned sigmoid gate and classifier [4] |
| Gate ablation | Same projection/classifier design with equal fixed weights |

Improved fusion uses `h = g * h_image + (1 - g) * h_text`; the ablation fixes `g = 0.5` and retrains the remaining head. Report parameter/capacity differences. This is an adaptation of gated fusion, not a claim to reproduce the paper's experiments.

Train each unimodal head separately, then each fusion head independently on the same frozen encoder features. Each configuration uses supervised cross-entropy as its overall loss, with no auxiliary joint-encoder loss.

Initial settings: AdamW, learning rate `1e-3`, weight decay `1e-4`, batch size 32, and up to 20 epochs/configuration. Reset Python, NumPy, CPU/CUDA, and data-loader seeds to `42` per configuration. Select highest validation macro-F1, then lower validation loss, then earlier epoch. Hold split, pretrained weights, preprocessing, seed, optimizer, epoch budget, and selection rule fixed across fusion comparisons.

## 6. Experiments and evaluation

**Metrics:** accuracy and macro-F1; select on validation macro-F1 and evaluate selected models once on test. Report per-class precision/recall/F1, confusion matrices, and learning curves.

| Comparison | Hypothesis / changed factor | Fixed factors | Decision criterion |
|---|---|---|---|
| Image vs. text | Modalities differ in predictive strength; modality/encoder change | Split, labels, seed, evaluation, checkpoint rule | Validation macro-F1; test metrics with compute cost; not an isolated architecture test |
| Unimodal vs. concatenation | Combining features may help ambiguous headlines; features/head change | Same encoders, split, preprocessing, evaluation | Validation macro-F1; whether test findings agree |
| Concatenation vs. gate | Adaptive weighting may help conflicts; fusion design changes | Same encoder features, split, seed, optimizer, budget, selection | Validation macro-F1, final test metrics, compute/capacity |
| Learned vs. fixed gate | Adaptive weights contribute beyond equal weighting | Same projections/classifier, encoder features, split, seed, schedule | Validation macro-F1, test metrics, parameter/time differences |

Analyze support cases, rescue cases, conflicting predictions, ambiguous headlines, illustrative images, Science/Technology confusion, editorial overlap, and repeated events. Include counts and representative examples; gate values alone do not prove causation. Fusion need not outperform unimodal baselines.

Diagnostic modality removal zeros the removed representation and bypasses its gate contribution. This is an inference intervention, not a missing-modality-trained model or substitute for independent baselines. Both modalities absent → abstain.

Measure total/trainable parameters, feature-extraction/head-training time, warmed-up model-only and end-to-end latency, and peak allocated GPU memory. End-to-end cost includes preprocessing, tokenization, encoders, and head; document batch size, timing boundaries, hardware, and GPU synchronization.

## 7. Compute and reproducibility

| Item | Initial estimate / record |
|---|---|
| Hardware/storage | Approximately 8 GB GPU memory and 10 GB free storage; confirm availability and actual usage |
| Schedule | Five configurations, ≤20 epochs each, batch size 32 |
| Time | Estimated 1–3 GPU-hours including extraction; pilot after prerequisites |
| Memory | Frozen encoders, deterministic cached features, microbatch extraction if needed |
| Environment | Exact Python, PyTorch, torchvision, Transformers/tokenizer versions and dependency lock; pretrained revisions/checksums |
| Traceability | Configs, seeds, final manifest, image hashes, exclusions, per-epoch logs, run IDs, checkpoint hashes, commit/tag, and reconstruction commands |

Supporting M1 audit material is local; this document preserves its selection rules and findings. Publish final experiment provenance and reconstruction instructions with the later implementation. No runnable training pipeline exists at M1.

## 8. Approval and limitations

**Pending instructor approval. Training permission is not cleared.** Student verification and full-dataset validation remain pending. No permission request was sent or grant supplied. If permission will not be obtained, replace N24News before main experiments. Instructor approval does not substitute for applicable content permissions.

Results will be limited by six topics, one English-language publisher, headline-only text, frozen encoders, one seed, and possible publisher/text shortcuts. The small visual sample does not establish full-dataset quality or event independence. No training, submission, instructor approval, permission grant, or completed licensing compliance is claimed.

**Milestones, each due 23:59 GMT+7:** M1 proposal on 18 November 2026; M2 unimodal/simple-fusion draft on 25 November; M3 improved fusion, ablation, analysis, report, slides, video, webpage, and checkpoints/reconstruction on 2 December. A proposal PDF is not required by the handbook; final deliverables remain pending.

## AI Usage Disclosure

OpenAI Codex assisted with research, metadata analysis, proposal/experiment planning, publisher-term investigation, the 18-pair visual review, and public records. Semantic judgments await student verification. Quach Kien Long requested the work. See the [detailed Assignment 3 disclosure](../source/AI_USAGE.md).

## References

1. Wang, Z., Shan, X., Zhang, X., and Yang, J. (2022). *N24News: A New Dataset for Multimodal News Classification*. LREC, 6768–6775. [Paper](https://aclanthology.org/2022.lrec-1.729/).
2. He, K., Zhang, X., Ren, S., and Sun, J. (2016). *Deep Residual Learning for Image Recognition*. CVPR. [Paper](https://arxiv.org/abs/1512.03385); [ResNet-18 implementation/weights](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html).
3. Sanh, V., Debut, L., Chaumond, J., and Wolf, T. (2019). *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter*. [Paper](https://arxiv.org/abs/1910.01108); [model/tokenizer](https://huggingface.co/distilbert/distilbert-base-uncased).
4. Arevalo, J., Solorio, T., Montes-y-Gomez, M., and Gonzalez, F. A. (2017). *Gated Multimodal Units for Information Fusion*. [Paper](https://arxiv.org/abs/1702.01992).
