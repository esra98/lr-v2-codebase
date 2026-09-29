# Literature notes

Compiled 2026-09-27. Two sources: (A) the two survey papers in `papers/`, (B) a web
literature search in which every citation was confirmed online (arXiv / ISCA / CVF /
ACL / NeurIPS / ICLR pages).

**Reliability caveats**

- Most numbers in part B were extracted from arXiv HTML automatically. Only the MultiVSR
  and "E-Branchformer vs Conformer" tables were read directly. Re-check any figure against
  the paper before quoting it in a manuscript.
- Survey tables were flattened by PDF-to-text conversion, so row/number pairings from
  the TPAMI survey's Table 3 are inferred from ordering.
- Rankings are expectations for this project, not claims made by the papers.

## A. What the surveys say

S1 = Sheng et al., "Deep Learning for Visual Speech Analysis: A Survey", IEEE TPAMI.
S2 = Rezaee & Yeganeh, "Automatic visual lip reading: A comparative review of
machine-learning approaches", Results in Engineering 28 (2025) 107171.

Neither survey discusses CLAHE, histogram equalisation, Procrustes alignment, mixup,
cutout, flipping, random cropping, speed perturbation, landmark smoothing, AV-HuBERT or
RAVEn. They are weak on preprocessing and augmentation in general.

| Topic | Finding | Source |
|---|---|---|
| ROI extent | Called high-impact and unresolved: "still unable to determine the optimal ROI" | S1 §4.1, §4.4.2, ref [121]; S2 §3.2, §5 |
| Head pose | Listed as a main challenge; "no formal robustness analysis" exists | S1 §2.2.1, §4.4.2 |
| Partition-Time Masking | LRW 91.8 % top-1, LRW-1000 56.1 %; DC-TCN+PTM 92.15 % on LRW. Word-level only | S2 §6.2, ref [137] |
| Backends | Transformers beat RNNs at sentence level but overfit small datasets | S1 §4.2.2, Table 2 |
| Losses | seq2seq better than CTC at sentence level; CTC adapts faster to longer sequences | S1 §4.3.1 |
| Distillation from ASR | Speeds up training; CTC + KD improves further. No sentence-level figure | S1 refs [52], [36], [34] |
| Acoustic alignments in CTC/attention | About 24 % absolute WER reduction on VLRF (Spanish) | S2 §6.2, ref [139] |
| External language model | "Can significantly improve performance". No figure | S1 §4.3.1, §4.4.1 |

## B. Verified recent work

### Encoder architecture

| Work | Reported result | Cost | Notes |
|---|---|---|---|
| E-Branchformer — Kim et al., SLT 2022, arXiv 2210.00077 | 1.81 / 3.65 WER LibriSpeech clean/other | medium | Parallel attention and cgMLP branches |
| E-Branchformer vs Conformer — Peng et al., Interspeech 2023, arXiv 2305.11073 | LibriSpeech-100h at equal size: 6.3 / 17.0 vs 6.6 / 17.2. Pure-CTC TEDLIUM2: Conformer failed 6 of 10 runs, E-Branchformer 2 of 10 | — | More stable when data is small |
| Branchformer for AVSR — Gimeno-Gómez & Martínez-Hinarejos, Computer Speech & Language 2025, arXiv 2407.06606 | 59.3M params, LRS3 VSR 41.2 / AVSR 2.1 | medium | |
| Zipformer — Yao et al., ICLR 2024, arXiv 2310.11230 | Zipformer-M 2.21 / 4.79 | high | U-Net downsampling suits long audio, not 50-frame clips |
| Squeezeformer (NeurIPS 2022, arXiv 2206.00888), Fast Conformer (ASRU 2023, arXiv 2305.05084) | Mainly efficiency gains | — | Little expected benefit at 25 fps |

### Transformer components

| Work | Reported result | Cost |
|---|---|---|
| RoPE for ASR — Zhang et al., arXiv 2501.06051 (preprint) | test-other 4.50 vs 4.57 RelPos; 13–21 % faster training | low; breaks RelPos checkpoint loading |
| Stochastic depth — Huang et al., ECCV 2016; LayerDrop — Fan et al., ICLR 2020 | No isolated gain; USR uses 0.1 | low |
| LayerScale, RMSNorm, SwiGLU | No ASR/VSR ablation found | low priority |

### CTC regularisation

| Work | Reported result | Cost |
|---|---|---|
| InterCTC — Lee & Watanabe, ICASSP 2021, arXiv 2102.03216 | — | low |
| InterCTC in VSR — Burchi & Timofte, WACV 2023, arXiv 2301.01456 | Visual-only LRS2 33.82 vs 37.39; LRS3 40.63 vs 44.96 | low |
| Self-conditioned CTC — Nozaki & Komatsu, Interspeech 2021, arXiv 2104.02724 | More than 20 % relative on WSJ | low |
| Hierarchical CTC — Higuchi et al., ICASSP 2022, arXiv 2110.04109 | Not verified | medium |
| CR-CTC — Yao et al., ICLR 2025, arXiv 2410.05101 | Conformer-CTC 2.92 / 7.15 to 2.43 / 5.78 | low-medium; doubles memory |
| OTC for noisy transcripts — Gao et al., ASRU 2023, arXiv 2309.15796 | Robust up to 70 % transcript errors | high (needs k2) |

### Fusion and unified training

| Work | Reported result | Cost |
|---|---|---|
| USR — Haliassos et al., NeurIPS 2024, arXiv 2411.02256 | LRS3 Base, 30 h labels: 36.0 / 3.2 / 3.0 (V/A/AV). VSR alone from scratch 85.5 vs 36.4 with shared weights (check table) | medium (supervised part) |
| USR 2.0 — Haliassos et al., ICLR 2026, arXiv 2602.19316 | About 50 % less training time | high |
| Modality dropout — Shi et al., Interspeech 2022, arXiv 2201.01763 | — | low |
| Modality-bias caveat — Dai et al., CVPR 2024, arXiv 2403.04245 | Video dropout increases reliance on audio | — |
| MLCA-AVSR cross-attention — Wang et al., ICASSP 2024, arXiv 2401.03424 | Up to 3.17 % relative cpCER | medium |

### Pretraining, pseudo-labels and low-resource languages

| Work | Reported result |
|---|---|
| MultiVSR — Prajwal, Hegde & Zisserman, ICASSP 2025 (github.com/Sindhu-Hegde/multivsr) | 104 h Turkish train. From scratch on low-resource languages: 97–105 WER. Fine-tuned on Turkish: about 62.9 WER at 25 h, 53.5 at 100 h (read off Fig. 4) |
| Whisper labels for low-resource VSR — Yeo et al., ICASSP 2024, arXiv 2309.08535 | Whisper labels close to human labels (Italian 59.17 vs 60.40) |
| Zero-AVSR / MARC — Yeo et al., ICCV 2025, arXiv 2503.06273 | 50.6 h Turkish in the corpus; no Turkish WER |
| RAVEn (ICLR 2023), BRAVEn (ICASSP 2024), AV-HuBERT (ICLR 2022), XLAVS-R (ACL 2024) | Usable as initialisation only; pretraining our own is out of budget |
| LiteVSR — Laux et al., ICASSP 2024, arXiv 2312.09727 | LRS3 45.7 WER on a consumer GPU by distilling from an ASR encoder |

### Front-end and augmentation

| Work | Reported result |
|---|---|
| SyncVSR — Ahn et al., Interspeech 2024, arXiv 2406.12233 | LRS2 30.7 vs 45.9 baseline (check table). Needs an audio tokeniser |
| ROI size — Zhang et al., arXiv 2205.14295 (preprint) | Full face helps with self-supervised pretraining, hurts at 30 h without it |
| Speed perturbation 0.9–1.1 — CNVSRC 2023 system, arXiv 2406.10313 | 4.86 % relative |
| Time masking and mixup — Ma et al., ICASSP 2022, arXiv 2209.01383 | Time masking most important, then mixup (word level) |

### LLM decoders

Llama-AVSR (ICASSP 2025, arXiv 2409.12319), VSP-LLM (Findings of EMNLP 2024,
arXiv 2402.15151), Whisper-Flamingo (Interspeech 2024, arXiv 2406.10082). Judged
impractical as the main line here: they depend on English-centric video encoders and an
8B LLM in memory, and the VSR gains come mostly from the encoder.

### Tokenisation and Turkish

| Work | Reported result |
|---|---|
| BPE-dropout — Laptev et al., arXiv 2103.07186 | Babel Turkish 73 h: vocab 1000 gives 46.5 WER, 44.0 with dropout 0.1; vocab 3000 gives 49.2, 43.2 with dropout |
| Vocabulary sizes in VSR | AVEC uses 256; USR uses 1000 |
| Turkish lip reading | Only small word/phrase-level sets: Atila & Sabaz 2022 (DOI 10.1016/j.jestch.2022.101206); Berkol et al. 2023 (DOI 10.3390/data8010015) |

No sentence-level Turkish AVSR paper with a modern architecture was found.

### Training recipe notes

- Auto-AVSR's README recommends a length curriculum: a subset of clips up to 4 s at
  lr 2e-4, then the full set at 1e-3.
- Practitioner reports describe NaN or blank collapse with CTC in fp16. Compute
  log-softmax and CTC in float32 with `zero_infinity=True`.
- With very short clips, check that the token count never exceeds the frame count.

## Ranked shortlist for A/B testing

1. Fine-tune from pretrained VSR weights (Auto-AVSR English or MultiVSR multilingual).
2. Filter, and optionally re-label, transcripts. Label noise caps everything else.
3. InterCTC, optionally self-conditioned.
4. Joint ASR/VSR/AVSR training with a shared encoder and modality sampling.
5. Tokeniser sweep: unigram 256 / 500 / 1000, with and without subword sampling.
6. Augmentation bundle: speed perturbation, stronger time masking, stochastic depth.
7. CR-CTC with a higher CTC weight (cloud GPU; doubles memory).
8. SyncVSR-style audio-token auxiliary loss.
9. Add MultiVSR / MARC Turkish data.
10. E-Branchformer encoder with RoPE (incompatible with pretrained Conformer weights).
