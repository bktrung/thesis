# SALT3 thesis — reference paper library

Deep technical notes on every foundational paper the **SALT3** thesis stands on. Each note has a
citation/arXiv header, the core idea, key math/architecture details, results, an explicit **"How
NeoBERT / SALT3 uses this"** section, and `[[wikilinks]]` to related notes (link target = the other
file's slug, e.g. `[[rope]]` → `01_architecture/rope.md`).

## What the thesis is (one paragraph)
SALT3 adapts the English **NeoBERT** encoder to **Vietnamese** in two stages. **(1) Cross-lingual
embedding initialization** (the WECHSEL → FOCUS → **SALT** lineage, improved): mine anchor pairs and
build the Vietnamese token embeddings as a **sparsemax-weighted average** of donor (**ViDeBERTa** /
**PhoBERT**) rows mapped into NeoBERT space — with a **Procrustes** alternative for the map, a
**global emb→dec map** for the LM head, and a **unigram log-frequency decoder bias** (the freq-bias
init). **(2) Continued pre-training (CPT)** on **CulturaX-vi** with a **Warmup-Stable-Decay (WSD)**
schedule (global-step-keyed warmup → flat peak → short cosine/**1-sqrt** cooldown), AdamW moments
carried across staged sessions.

## Map: paper → thesis section

### 01_architecture/ — the NeoBERT foundation (what the model *is*)
| Note | Paper | arXiv | Role in thesis |
|---|---|---|---|
| [transformer](01_architecture/transformer.md) | Attention Is All You Need (2017) | 1706.03762 | base encoder block |
| [bert](01_architecture/bert.md) | BERT (2019) | 1810.04805 | MLM objective (CPT trains this) |
| [neobert](01_architecture/neobert.md) | **NeoBERT (2025)** | 2502.19587 | **the base model SALT3 adapts** |
| [rope](01_architecture/rope.md) | RoFormer / RoPE (2021) | 2104.09864 | positions (4,096 ctx) |
| [swiglu](01_architecture/swiglu.md) | GLU Variants (2020) | 2002.05202 | FFN activation |
| [rmsnorm](01_architecture/rmsnorm.md) | RMSNorm (2019) | 1910.07467 | Pre-RMSNorm; init scale interacts |
| [pre-ln](01_architecture/pre-ln.md) | On Layer Norm / Pre-LN (2020) | 2002.04745 | stable deep training + warmup |
| [gelu](01_architecture/gelu.md) | GELU (2016) | 1606.08415 | alt activation; donor era |
| [adamw](01_architecture/adamw.md) | Decoupled Weight Decay (2019) | 1711.05101 | optimizer; moments carried in CPT |
| [flash-attention](01_architecture/flash-attention.md) | FlashAttention (2022) | 2205.14135 | attention backend (xFormers/Flash) |
| [ngpt](01_architecture/ngpt.md) | nGPT (2024) | 2410.01131 | NormNeoBERT variant; norm geometry |
| [yarn](01_architecture/yarn.md) | YaRN (2023) | 2309.00071 | RoPE context-window extension (4,096+) |
| [mask-ratio](01_architecture/mask-ratio.md) | Should You Mask 15%? (2023) | 2202.08005 | MLM objective; NeoBERT/CPT mask 20% |

### 02_init_crosslingual/ — the embedding-init lineage (SALT3's core contribution)
| Note | Paper | arXiv | Role in thesis |
|---|---|---|---|
| [wechsel](02_init_crosslingual/wechsel.md) | WECHSEL (2022) | 2112.06598 | lineage root: similarity-weighted donor avg |
| [focus](02_init_crosslingual/focus.md) | FOCUS (2023) | 2305.14481 | **sparsemax-over-anchors** (direct ancestor) |
| [salt](02_init_crosslingual/salt.md) | **SALT — Semantic Aware Linear Transfer (2025)** | 2505.10945 | **the origin method SALT3 extends** |
| [procrustes](02_init_crosslingual/procrustes.md) | Orthogonal Procrustes mapping (2017) | 1702.03859 | the `procrustes_init` arm |
| [sparsemax](02_init_crosslingual/sparsemax.md) | Sparsemax (2016) | 1602.02068 | the anchor combination rule |
| [freq-bias-init](02_init_crosslingual/freq-bias-init.md) | freq-bias synthesis | 1708.02002 + others | **SALT3 decoder-bias contribution** |
| [zero-shot-tokenizer-transfer](02_init_crosslingual/zero-shot-tokenizer-transfer.md) | ZeTT (2024) | 2405.07883 | context: learned-init comparison |
| [ethayarajh-anisotropy](02_init_crosslingual/ethayarajh-anisotropy.md) | Geometry of contextual emb. (2019) | 1909.00512 | **theory: why per-token, not global** |
| [representation-degeneration](02_init_crosslingual/representation-degeneration.md) | Representation Degeneration (2019) | 1907.12009 | **theory: why untied head** |
| [ormazabal-mapping-limits](02_init_crosslingual/ormazabal-mapping-limits.md) | Limits of cross-lingual mappings (2019) | P19-1492 | **theory: global map is limited** |
| [ofa](02_init_crosslingual/ofa.md) | OFA (2024) | 2311.08849 | init baseline (factorized, multilingual) |
| [fasttext-subword](02_init_crosslingual/fasttext-subword.md) | fastText subword (2017) | 1607.04606 | static space; OOV-robust vectors |
| [fasttext-157lang](02_init_crosslingual/fasttext-157lang.md) | fastText 157 langs (2018) | 1802.06893 | **`cc.vi.300` similarity space for anchors** |
| [marian-nmt](02_init_crosslingual/marian-nmt.md) | Marian / MarianMT (2018) | 1804.00344 | **vi→en back-translation for anchor mining** |

### 03_cpt_schedule/ — continued pre-training (warmup / WSD / decay)
| Note | Paper | arXiv | Role in thesis |
|---|---|---|---|
| [dont-stop-pretraining](03_cpt_schedule/dont-stop-pretraining.md) | DAPT/TAPT (2020) | 2004.10964 | why CPT works |
| [wsd-minicpm](03_cpt_schedule/wsd-minicpm.md) | MiniCPM / WSD (2024) | 2404.06395 | **the CPT schedule** |
| [hagele-cooldown](03_cpt_schedule/hagele-cooldown.md) | Constant-LR + cooldown (2024) | 2405.18392 | **1-sqrt cooldown** (cited in code) |
| [cosine-sgdr](03_cpt_schedule/cosine-sgdr.md) | SGDR / cosine (2017) | 1608.03983 | NeoBERT's decay + a cooldown shape |
| [warmup](03_cpt_schedule/warmup.md) | RAdam / warmup (2020) | 1908.03265 | why warmup; 2%-clamp rule |
| [language-adaptation](03_cpt_schedule/language-adaptation.md) | Recycle GPT-2 to new langs (2021) | 2010.02559 | freeze-then-adapt blueprint |
| [emergent-cpt-language-adaptation](03_cpt_schedule/emergent-cpt-language-adaptation.md) | Emergent abilities under CPT (2025) | 2506.00288 | English-replay / forgetting in lang-adapt CPT |
| [ibrahim-continual-pretrain](03_cpt_schedule/ibrahim-continual-pretrain.md) | Continual pretrain strategies (2024) | 2403.08763 | **re-warm LR after cooldown-to-0** |

### 04_baselines_donors/ — comparators + Vietnamese donors
| Note | Paper | arXiv | Role in thesis |
|---|---|---|---|
| [roberta](04_baselines_donors/roberta.md) | RoBERTa (2019) | 1907.11692 | baseline; PhoBERT's architecture |
| [modernbert](04_baselines_donors/modernbert.md) | ModernBERT (2024) | 2412.13663 | baseline NeoBERT beats |
| [nomicbert](04_baselines_donors/nomicbert.md) | Nomic Embed (2024) | 2402.01613 | MTEB baseline; embedding pipeline |
| [llama](04_baselines_donors/llama.md) | LLaMA / Llama 2 (2023) | 2302.13971 / 2307.09288 | the AdamW+RoPE+SwiGLU recipe source |
| [phobert](04_baselines_donors/phobert.md) | PhoBERT (2020) | 2003.00744 | **donor + downstream baseline** |
| [videberta](04_baselines_donors/videberta.md) | ViDeBERTa (2023) | 2301.10439 | **primary embedding donor** |
| [culturax](04_baselines_donors/culturax.md) | CulturaX (2023) | 2309.09400 | **the CPT corpus** |
| [refinedweb](04_baselines_donors/refinedweb.md) | RefinedWeb (2023) | 2306.01116 | **NeoBERT's English pre-training corpus** |
| [deepseek](04_baselines_donors/deepseek.md) | DeepSeek-V3 (2024) | 2412.19437 | modern-LM trend (intro motivation) |

## Reading order for the report's related-work chapter
1. **Architecture story:** [transformer] → [bert] → ([rope], [swiglu], [rmsnorm], [pre-ln], [adamw], [flash-attention]) → [neobert] (with [llama] as the recipe source, [roberta]/[modernbert]/[nomicbert] as the encoder landscape).
2. **Init story:** [wechsel] → [focus] → [salt] → SALT3's additions ([sparsemax], [procrustes], [freq-bias-init]); donors [videberta]/[phobert]; [zero-shot-tokenizer-transfer] as the learned-init contrast.
3. **CPT story:** [dont-stop-pretraining] / [language-adaptation] (why) → [warmup] + [cosine-sgdr] → [wsd-minicpm] + [hagele-cooldown] (how) → [culturax] (data).

## Notes on sourcing
- Titles, authors, years, and arXiv IDs were verified against each paper's arXiv abstract page.
- Equations/architecture facts cross-checked against the repo's ground truth: `NeoBERT/docs/architecture.md`, `NeoBERT/README.md`, `NeoBERT/conf/**` (optimizer/scheduler/model yamls), `code/salt3_staged_schedule.py`, `code/salt3_decoder_variants.py`, `scripts/test_decoder_global_map_and_freq_bias.py`.
- `freq-bias-init` is a **synthesis note** (no single source paper); it cites Zipf, Focal Loss bias-init (1708.02002), tied embeddings (1608.05859), and adaptive softmax (1609.04309).
