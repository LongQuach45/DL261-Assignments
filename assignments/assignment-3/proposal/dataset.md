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
| Dataset | CREMA-D — Crowd-sourced Emotional Multimodal Actors Dataset [1] |
| Source | [Official repository](https://github.com/CheyneyComputerScience/CREMA-D) |
| Version | Repository snapshot [`1658cd342dff90010aa843eaeebd53610a08b1dc`](https://github.com/CheyneyComputerScience/CREMA-D/tree/1658cd342dff90010aa843eaeebd53610a08b1dc), dated 21 March 2025; dataset paper published in 2014 |
| License | Database: Open Database License (ODbL 1.0). Individual contents: Database Contents License (DbCL 1.0), as stated in the [source license](https://github.com/CheyneyComputerScience/CREMA-D/blob/1658cd342dff90010aa843eaeebd53610a08b1dc/LICENSE.txt). Retain attribution and source notices; keep raw media outside this repository. |
| Task | Six-class audio–visual classification of acted emotion; proposed instructor-approved multimodal classification task |
| Modalities/input | Speech waveform and silent RGB video from the same recorded utterance |
| Output | One intended emotion: anger, disgust, fear, happy, neutral, or sad |
| Prediction unit | One paired utterance recording; frames are not independent samples |
| Annotations | Filename encodes actor, sentence, intended emotion, and intensity; CSVs contain crowd-perceived voice, face, and audiovisual emotion votes |
| Formats/access | `AudioWAV/*.wav`, `VideoFlash/*.flv`, and CSV metadata. Media require Git LFS; GitHub ZIP archives contain media pointers, not usable recordings. |

The task studies whether vocal and facial evidence support or contradict each other when predicting intended emotion. The source contains 91 actors speaking 12 fixed sentences with different emotions and intensities. These are genuine recording-level pairs, not independently collected or randomly matched modalities. Fixed sentence text is not treated as a third modality.

The target is the actor's **intended emotion**, decoded from the filename. Crowd-perceived emotion is supplementary quality evidence; it does not replace the target. Intended and perceived emotion can differ. The dataset contains acted English utterances and is not a benchmark for clinical assessment or unrestricted real-world emotion inference.

## 2. Preliminary distribution and pair quality

The [metadata audit](metadata-audit.json) was reproduced from the pinned repository tree and `processedResults/summaryTable.csv`. It checks identifiers and annotations, not media integrity.

| Finding | Count |
|---|---:|
| Original WAV files / video files | 7,442 / 7,442 |
| Matched recording IDs | 7,442 |
| Missing counterpart paths | 0 |
| Actors / classes | 91 / 6 |
| Known defective recordings excluded | 4 |
| Planned eligible pairs | **7,438** |

| Intended emotion | Code | Eligible pairs |
|---|---|---:|
| Anger | ANG | 1,271 |
| Disgust | DIS | 1,270 |
| Fear | FEA | 1,271 |
| Happy | HAP | 1,271 |
| Neutral | NEU | 1,086 |
| Sad | SAD | 1,269 |
| **Total** | | **7,438** |

Neutral is modestly underrepresented; the largest-to-smallest class ratio is approximately 1.17. The dataset exceeds the handbook's default of at least five classes and about 5,000 paired samples.

All 7,438 eligible annotation rows have nonempty voice, face, and audiovisual vote fields. Among 6,379 rows with single-valued voice and face votes, 3,020 disagree (47.34%); 1,059 rows contain tied or other non-single voice/face votes. These are annotation-level observations. Perceptual disagreement can reflect conflicting modality evidence and is not proof of incorrectly matched files.

The [authors document four defective videos](https://github.com/CheyneyComputerScience/CREMA-D/blob/1658cd342dff90010aa843eaeebd53610a08b1dc/README.md#video-files), excluded as whole pairs before experimentation:

| Recording ID | Documented issue |
|---|---|
| `1076_MTI_NEU_XX` | Very short video with no audio |
| `1076_MTI_SAD_XX` | Error picture, no audio, very short video |
| `1064_TIE_SAD_XX` | Video has no duration |
| `1064_IEO_DIS_MD` | One-minute compilation rather than the named utterance |

**Pending media audit:** retrieve real LFS payloads, verify decoding and waveform validity, measure durations and silence, inspect audio/video alignment, check exact/content duplicates, and review representative pairs across classes and intensities. Report missing or mismatched media separately from annotation disagreement. Further invalid pairs will be excluded with reasons and revised counts; no successful media audit or model result is claimed at M1.

## 3. Split and leakage prevention

Use a custom actor-disjoint split: sort all 91 actor IDs, shuffle with Python `random.Random(42)`, and take 73 training, nine validation, and nine test actors. Apply the four fixed exclusions without reassigning actors. The [actor lists](actors.json), [recording manifest](manifest.csv), and [split counts](split-counts.csv) are committed.

| Partition | Actors | Eligible pairs | Purpose |
|---|---:|---:|---|
| Training | 73 | 5,965 | Parameter fitting and normalization statistics |
| Validation | 9 | 735 | Hyperparameters and checkpoint selection |
| Test | 9 | 738 | Final evaluation after selection |
| **Total** | **91** | **7,438** | |

All six classes occur in every partition. This is grouped splitting, not sample-level stratification. Keep all utterances, audio variants, frames, and cached features of each actor in one partition. Never split frames independently or use actor IDs, filenames, crowd votes, or intensity annotations as model inputs. The same fixed sentences may occur across actors; the generalization claim concerns unseen actors, not unseen sentences.

Before training, investigate cross-partition duplicate content; quarantine any confirmed duplicate group without moving actors, and record exclusions. Fit audio statistics using training data only; ImageNet normalization uses published pretrained constants. Test labels and crowd votes are excluded from model selection. Use the entire eligible dataset; smaller actor-respecting subsets may be used only for debugging, with size, purpose, and seed recorded. If further exclusions reduce scope materially or threaten default thresholds, seek instructor approval for the revision.

## 4. Preprocessing and alignment

```text
Same recording ID → paired audio/video validation → actor-fixed split
                  → waveform/log-mel + sampled silent frames → data loader
                  → modality encoders → unimodal head or fusion → cross-entropy
                  → optimization → intended emotion → evaluation
```

- **Audio:** convert/resample to mono 16 kHz; compute log-mel power features with 64 mel bins, a 25 ms Hann window, 10 ms hop, and 512-point FFT. Pad variable-length sequences within each batch and retain length masks; exclude padded regions from pooling/statistics. Expected batch shape `(B, 1, 64, T_max)`.
- **Video:** sample eight evenly spaced timestamps over the matching utterance, convert to RGB, resize the shorter side to 256 and center-crop to `224 × 224`, then apply ImageNet normalization. Expected batch shape `(B, 8, 3, 224, 224)`. Do not feed embedded video audio into the visual encoder.
- **Alignment:** pair files by their complete recording ID and use their corresponding utterance intervals. Investigate duration discrepancies and synchronization failures before accepting a pair. Both encoders summarize the same utterance; frame-level phoneme alignment is not claimed.
- **Controls:** deterministic preprocessing for the core comparison; no augmentation initially. Preserve recording IDs and masks in each batch. Raw missing/corrupt modalities exclude the pair from the primary benchmark. Do not substitute unrelated recordings.

## 5. Models and training plan

All models predict the same six labels on the same actor split. PyTorch supplies training primitives; torchvision supplies pretrained ResNet-18 [2]. The audio CNN, pooling, projections, classifiers, fusion, and evaluation protocol will be implemented by the group.

| Configuration | Representation and trainable components |
|---|---|
| Audio only | Three convolutional blocks on log-mel features; masked temporal/global frequency pooling; 128-dimensional embedding; six-class linear head. Train from scratch. |
| Video only | ImageNet-pretrained ResNet-18 [2] produces 512 features per frame; average across eight frames; train a six-class linear head. Freeze the backbone and keep it in evaluation mode, including BatchNorm buffers. |
| Simple fusion | Freeze the selected unimodal encoders; concatenate audio/video embeddings (128 + 512); train a 128-unit ReLU MLP and six-class head. |
| Improved fusion | Freeze the same encoders; project each modality to 128 dimensions and learn a sigmoid gate over their concatenation; classify the gated representation, inspired by gated multimodal units [3]. |
| Gate ablation | Same projected representations and classifier as improved fusion; replace the learned gate with equal weights `g = 0.5` and retrain the remaining fusion components. |

For improved fusion, `h = g * h_audio + (1 - g) * h_video`, where `g` is a learned 128-dimensional gate. The fixed-gate ablation removes adaptive weighting; report the resulting difference in trainable parameters.

Train the audio model and video classifier separately using their own cross-entropy losses. Select their checkpoints, freeze the encoders, and train each fusion configuration with cross-entropy on the same target. There is no auxiliary multimodal loss: the overall loss for each configuration is its supervised cross-entropy. Backbone fine-tuning is outside the core scope.

Initial schedule: AdamW, learning rate `1e-3`, weight decay `1e-4`, batch size 32, maximum 20 epochs per configuration. Reset Python, NumPy, CPU, CUDA, and data-loader seeds to `42` before each configuration. Select the highest validation macro-F1 checkpoint, breaking ties by lower validation loss and then earlier epoch. Keep optimizer, epoch budget, split, seed, preprocessing, target labels, and checkpoint rule fixed across fusion comparisons. Tune only on validation data and document any departure from these initial settings.

## 6. Experiments, evaluation, and error analysis

**Primary metrics:** accuracy and macro-F1, with macro-F1 used for model selection. Also report per-class precision/recall/F1 and confusion matrices. Evaluate each selected configuration once on test after selection; no test-driven tuning.

| Experiment | Hypothesis / changed factor | Fixed factors | Decision criterion |
|---|---|---|---|
| Audio vs. video | Vocal and facial inputs have different strengths; change modality and its encoder. This is a baseline comparison, not an isolated architecture test. | Actor split, labels, seed, metrics, checkpoint rule | Validation macro-F1 and final test metrics, interpreted with compute cost |
| Unimodal vs. concatenation | Combining representations may improve over either modality; change available representations and head | Selected encoder checkpoints, split, preprocessing, evaluation | Validation macro-F1; report whether test findings agree |
| Concatenation vs. learned gate | Adaptive weighting may help conflicting pairs; change fusion design | Same frozen encoder features, training budget, seed, optimizer, selection | Validation macro-F1 and final test metrics; acknowledge capacity differences |
| Learned vs. fixed gate | Input-dependent weighting contributes beyond equal weighting; remove adaptive gate | Projection/head design, encoder features, split, seed, schedule | Validation macro-F1, final test metrics, parameter/time difference |

Plan qualitative examples where both modalities are correct, one correct modality rescues the other, or modalities conflict. Use prediction disagreement and supplementary crowd votes for retrospective analysis only. Report counts and representative cases, not gate values alone as causal explanations. Examine intended/perceived label disagreement, low intensity, weak speech or facial cues, and confused emotion pairs.

Diagnostic audio/video removal from the selected fusion model will zero the corresponding representation and bypass its gate contribution, retaining only the available modality. Label this an inference intervention, not a missing-modality-trained model or a replacement for independently trained unimodal baselines. Both modalities absent produces no prediction.

Measure total/trainable parameters, training time, feature-extraction time, warmed-up model-only and end-to-end inference latency, and peak GPU memory. Include decoding/preprocessing in end-to-end cost and identify batch size, hardware, timing boundaries, and GPU synchronization. Multimodal improvement is not guaranteed; explain negative results using evidence. One seed and acted, short, fixed-sentence English clips limit generalization.

## 7. Compute and reproducibility

| Item | Initial estimate / record |
|---|---|
| Hardware | One CUDA GPU with approximately 8 GB VRAM; actual hardware availability is not yet confirmed |
| Storage | 20 GB free disk budget; official full source clone is approximately 7.55 GB, with additional space for caches/checkpoints |
| Training budget | Five configurations, up to 20 epochs each; batch size 32 |
| Time | Estimated 2–6 GPU-hours including feature extraction; validate with a pilot after approval |
| Memory control | Freeze visual backbone; process frames in microbatches and cache deterministic features; reduce batch size consistently if required and record the change |
| Environment | Record actual Python/PyTorch/torchvision/audio-processing/FFmpeg versions and dependency lock before experiments |
| Provenance | Save configs, seeds, actor manifest, media hashes, exclusions, normalization statistics, per-epoch logs, checkpoint hashes, run IDs, and repository commit/tag |

The metadata evidence is reproducible with Python 3 and GitHub CLI:

```bash
gh auth login
python3 assignments/assignment-3/proposal/audit_metadata.py
```

The command fetches only the pinned file tree and annotation CSV, verifies structural invariants, and compares its outputs with committed evidence. Use `--write` only to regenerate those evidence files. See [audit notes and requirement mapping](README.md). Dataset media, caches, and later checkpoints remain outside Git; reconstruction instructions will accompany implementation.

## 8. Approval and milestone boundaries

**Pending instructor approval.** No instructor decision or submission is claimed. The proposal requests approval for this audio–visual classification task and actor-disjoint protocol. Main experiments begin only after **Approved** or **Approved with conditions**; record any conditions before implementation.

- **M1 — 18 November 2026:** proposal, preliminary metadata evidence, split plan, and AI disclosure ready for review. Group review and instructor submission/approval remain external actions.
- **M2 — 25 November 2026:** implement and evaluate audio-only, video-only, and simple fusion after approval.
- **M3 — 2 December 2026:** improved fusion, ablation, compute/error analysis, final report, slides, video, assignment page, and checkpoints or reconstruction instructions. All deadlines are 23:59 GMT+7.

The handbook requires a Dataset Proposal record and approval status; it does not require a proposal PDF. Final report, slides, video, trained checkpoints, and experiment results are later deliverables and are not claimed here.

## AI Usage Disclosure

OpenAI Codex assisted with source research, metadata-audit scripting, proposal and experiment planning, and repository/webpage updates. Automated checks do not constitute student approval. Quach Kien Long requested the work; final student verification remains pending. See the [detailed Assignment 3 record](../source/AI_USAGE.md).

## References

1. Cao, H., Cooper, D. G., Keutmann, M. K., Gur, R. C., Nenkova, A., and Verma, R. (2014). *CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset*. IEEE Transactions on Affective Computing, 5(4), 377–390. [doi:10.1109/TAFFC.2014.2336244](https://doi.org/10.1109/TAFFC.2014.2336244). [Dataset documentation and defect notices](https://github.com/CheyneyComputerScience/CREMA-D/blob/1658cd342dff90010aa843eaeebd53610a08b1dc/README.md).
2. He, K., Zhang, X., Ren, S., and Sun, J. (2016). *Deep Residual Learning for Image Recognition*. CVPR. [Paper](https://arxiv.org/abs/1512.03385); [torchvision ResNet-18 and ImageNet weights](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html).
3. Arevalo, J., Solorio, T., Montes-y-Gomez, M., and Gonzalez, F. A. (2017). *Gated Multimodal Units for Information Fusion*. [arXiv:1702.01992](https://arxiv.org/abs/1702.01992). The proposed audio–visual adaptation is not a claim to reproduce the paper's movie-genre experiments.
