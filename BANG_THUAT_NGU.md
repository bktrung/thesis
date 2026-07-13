# BẢNG THUẬT NGỮ KHÓA LUẬN — ViNeoBERT

Ghi chú đối chiếu thuật ngữ Anh–Việt dùng thống nhất trong toàn khóa luận (Chương 1–5,
Tóm tắt, Đề cương). Nguồn chuẩn hóa: bản dịch cộng đồng *Dive into Deep Learning* (d2l),
bài mẫu văn phong (KLTN cùng GVHD), và các paper peer-reviewed. Nguyên tắc: **ưu tiên từ
Việt đã chuẩn hóa, giải thích tiếng Anh ở lần đầu, một khái niệm một tên**.

---

## 1. THUẬT NGỮ DỊCH SANG TIẾNG VIỆT

### 1.1. Kiến trúc & thành phần mô hình
| Tiếng Anh | Tiếng Việt (dùng trong bài) | Ghi chú |
|---|---|---|
| embedding | **véc-tơ** (không gian/lớp/ma trận/hàng véc-tơ) | KHÔNG dùng "nhúng" |
| hidden size / hidden dimension | **kích thước ẩn** | không "chiều ẩn" |
| dimension / dimensionality | **chiều** (số chiều) | phân biệt với "phần tử" |
| element (of a vector) | **phần tử** | một ô/tọa độ của véc-tơ |
| activation function | **hàm kích hoạt** | |
| feed-forward network (FFN) | **mạng truyền thẳng** | |
| residual / skip connection | **kết nối tắt** (residual connection) | |
| normalization layer | **lớp chuẩn hóa** (normalization layer) | |
| layer / block (Transformer) | **lớp** / **khối** | không "tầng" |
| sub-layer | **lớp con** | không "tầng con" |
| positional encoding | **mã hóa vị trí** | absolute → vị trí tuyệt đối |
| attention head | **đầu chú ý** (attention head) | |
| self-attention | **tự chú ý** | |
| multi-head attention | **tự chú ý đa đầu** (Multi-Head Attention) | |
| attention score | **điểm chú ý** (attention score) | trước softmax ($q^\top k$) |
| attention weight / matrix | **trọng số chú ý** / **ma trận chú ý** | sau softmax |
| output head / LM head | **lớp đầu ra** / **đầu ra** | không "giải mã" |
| gain / scale vector (γ, β) | **véc-tơ tỉ lệ** / **dịch chuyển** | |
| recurrent (RNN) | **hồi quy** | cơ chế hồi quy; không "hồi tiếp" |
| autoregressive | **tự hồi quy** | phân biệt bằng tiền tố "tự" |

### 1.2. Huấn luyện & tối ưu
| Tiếng Anh | Tiếng Việt | Ghi chú |
|---|---|---|
| masked language model / mask | **mặt nạ** (mô hình hóa ngôn ngữ có mặt nạ); **token mặt nạ** | chuẩn d2l/FAIR |
| dynamic masking | **mặt nạ động** | RoBERTa |
| causal mask | **che các token phía sau** (causal mask) | không "mặt nạ nhân quả" |
| self-supervised | **tự giám sát** (self-supervised) | |
| catastrophic forgetting | **mất thông tin** (catastrophic forgetting) | không "quên lãng thảm họa" |
| learning rate | **tốc độ học** (learning rate) | |
| learning rate schedule | **lịch điều chỉnh tốc độ học** | không "lịch tốc độ học" |
| warmup | **khởi động** | |
| cooldown / decay | **làm nguội** | |
| re-warming | **khởi động lại tốc độ học** | |
| re-centering / re-scaling | **trừ trung bình** / **chia theo độ lớn** | (re-centering) / (re-scaling) |
| bias (tham số mô hình) | **hệ số điều chỉnh** (bias) | chuẩn d2l; không "độ chệch/độ lệch" |
| frequency bias | **hệ số điều chỉnh theo tần suất token** | |
| weight tying / tied | **dùng chung trọng số** | không "buộc trọng số" |
| untied | **không dùng chung trọng số** | không "tách rời/untied head" |
| weight decay | weight decay (**suy giảm trọng số**) | gloss lần đầu |
| gradient clipping | gradient clipping (**cắt gradient theo chuẩn**) | |
| downstream task | **tác vụ ứng dụng** | không "hạ nguồn/xuôi dòng" |
| domain adaptation | **thích ứng miền** (domain adaptation) | |
| language adaptation / transfer | **chuyển đổi ngôn ngữ** / **thích ứng** | không "thích nghi" |
| held-out set | **tập giữ lại** (held-out) | |
| streaming | **truyền trực tuyến** (streaming) | |
| random seed | **seed ngẫu nhiên** (random seed) | giữ "seed" |
| freeze | **đóng băng** | |

### 1.3. Phương pháp SALT & tokenizer
| Tiếng Anh | Tiếng Việt | Ghi chú |
|---|---|---|
| SALT | **Chuyển đổi tuyến tính trong không gian có ngữ nghĩa** (Semantic Aware Linear Transfer – SALT) | |
| donor model | **mô hình nguồn véc-tơ** (donor) = ViDeBERTa | |
| target / base model | **mô hình đích** / **mô hình nền** = NeoBERT | KHÔNG dịch theo SALT-paper ("source model") |
| anchor set / anchor | **tập điểm neo** (anchor set) / **điểm neo** | không "tập neo" |
| surface anchor | **điểm neo trùng mặt chữ** | |
| numeric anchor | **điểm neo chữ số** | |
| overlapping vocabulary | **từ vựng chồng lấp** (overlapping vocabulary) | |
| back-translation | **dịch ngược** (back-translation) | |
| false friends | **đồng tự khác nghĩa** (false friends) | |
| vocabulary pruning | **giảm kích thước từ vựng** (vocabulary pruning) | |
| frozen-alignment stage | **giai đoạn căn chỉnh với phần thân đóng băng** | |
| norm calibration | **hiệu chỉnh độ lớn** (về độ lớn trung bình của NeoBERT) | |
| tokenizer | **tokenizer** (giữ tiếng Anh) | bài dùng "tokenizer" xuyên suốt; "bộ token hóa" chỉ là nghĩa mô tả, KHÔNG dùng trong văn bản |
| tokenization / segment | **tách** (tách token / tách văn bản thành token) | không "phân đoạn thành token" |
| word segmentation | **tách từ** (word segmentation) | chỉ cho VnCoreNLP |
| sub-word | **từ con** (sub-word) | |
| morpheme | **hình vị** | đơn vị nhỏ nhất mang nghĩa |
| syllable | **âm tiết** | |
| Unigram log-probability | **điểm log-xác suất** | |

### 1.4. Toán học & đánh giá
| Tiếng Anh | Tiếng Việt | Ghi chú |
|---|---|---|
| pseudo-inverse (Moore–Penrose) | **giả nghịch đảo** | nghiệm bình phương tối thiểu chuẩn nhỏ nhất |
| orthogonal matrix | **ma trận trực giao** | |
| probability simplex | **đơn hình xác suất** | |
| Abel transformation | **biến đổi Abel** (tổng từng phần) | |
| product of experts | **tích của hai phân phối** (product of experts) | |
| long-term decay | **suy giảm theo khoảng cách** (long-term decay) | |
| Laplace smoothing | **làm trơn Laplace** | |
| moment-matched | **khớp mô-men** | |
| centroid | **tâm** (centroid) | |
| mean pooling | **gộp trung bình** (mean pooling) | |
| baseline | **đường cơ sở** / **cơ sở** | |
| adversarial (NLI) | **đối kháng** | |
| joint PCA basis | **hệ trục chung** (joint PCA basis) | |
| norm / magnitude | **chuẩn** / **độ lớn** | |
| scale (γ) | **giảm biên độ** / **hệ số tỉ lệ** / **co giãn** | |

---

## 2. THUẬT NGỮ GIỮ NGUYÊN TIẾNG ANH (không dịch)

Giữ nguyên vì là **tên riêng, ký hiệu, hoặc thuật ngữ chưa có bản dịch Việt phổ biến/ổn định**.

**Tên mô hình / kiến trúc:** NeoBERT, ViNeoBERT, ViDeBERTa, PhoBERT, XLM-R, BERT, RoBERTa,
ModernBERT, NomicBERT, DeBERTaV3, LLaMA, Gemma, DeepSeek.

**Tên kỹ thuật / thành phần (danh từ riêng):** RoPE (Rotary Position Embeddings), SwiGLU,
GELU, SiLU, RMSNorm, LayerNorm, Pre-RMSNorm, Post-norm/Pre-norm, GLU (Gated Linear Unit),
FlashAttention-2, YaRN, WSD (Warmup–Stable–Decay), WECHSEL, FOCUS, OFA, Sparsemax, AdamW,
MarianMT, VnCoreNLP, fastText.

**Khái niệm nền / viết tắt phổ biến:** token, Transformer, Encoder, Decoder, LLM, MLM, NSP,
CPT (Continued Pre-training), BPE (Byte-Pair Encoding), Unigram, SentencePiece, softmax,
logit, gradient, epoch, batch / batch size, throughput.

**Độ đo / chuẩn đánh giá:** BLEU, EM (Exact Match), F1, F1-macro, F1-weighted, MCC,
Spearman, nDCG@10, MRR, Accuracy, perplexity (PPL), pseudo-perplexity (PPPL), MTEB.

**Bộ dữ liệu / thư viện:** CulturaX, RefinedWeb, CC-100, OSCAR, Wikipedia, UIT-ViQuAD, XNLI,
ViANLI, QNLI, UIT-VSFC, UIT-VSMEC, UIT-ViCTSD, ViCoLA-syn, UD-VTB, STS, GLUE, SQuAD, VLSP,
PyTorch, Hugging Face (transformers, datasets, safetensors).

**Ký hiệu toán:** $\gamma$ (hệ số tỉ lệ), $\theta$, $d$/$d_h$/$d_{ff}$, $\eta$, $E$, $W_{dec}$,
$b_{dec}$, $\mathcal{V}$…

---

## 3. GHI CHÚ CÁC QUYẾT ĐỊNH QUAN TRỌNG (tránh nhầm)

- **"chiều" ≠ "phần tử"**: "chiều" = số chiều của véc-tơ (768 chiều); "phần tử" = một ô của véc-tơ.
- **"tầng" chỉ dùng nghĩa *tier/mức*** (ba tầng điểm neo, tầng lọc dữ liệu); *layer* luôn là **"lớp"**.
- **"mô hình nguồn véc-tơ" = ViDeBERTa, "mô hình đích/nền" = NeoBERT** — SALT paper gọi ngược ("source model" = LLM), TUYỆT ĐỐI không dịch theo paper.
- **SALT không yêu cầu cùng kích thước từ vựng** — việc giảm về 30.522 token là *lựa chọn* để đồng bộ tham số + giảm tài nguyên (phần thân xử lý véc-tơ 768 chiều, không phụ thuộc cỡ từ vựng).
- **Văn phong**: không dùng từ biểu cảm/thi đua (thắng/thua/bỏ xa/sụp đổ/bật lên…), không nói quá (hơn hẳn/tuyệt đối/khổng lồ), ngôi "chúng tôi" cho phần thân.
- Giải thích thuật ngữ tiếng Anh ở **lần đầu** xuất hiện; sau đó dùng nhất quán một dạng.
- **Nhãn CHỮ BÊN TRONG hình: để tiếng Anh** (đồng nhất cả 6 hình — "Held-out MLM loss", "F1 on UIT-ViQuAD", "Naive random", "Weight tying", "SALT (both)", "SALT + freeze"…), số dùng **dấu chấm** thập phân. Riêng **caption tiếng Việt** đặt dưới hình. Đừng Việt hóa chữ trong hình (từng thử với `fig_init_strategy` rồi hoàn nguyên vì 2 hình không có script không Việt hóa được).
