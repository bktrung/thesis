# Mô tả chi tiết slide deck bảo vệ khóa luận — ViNeoBERT

> **Khung thời gian**: 15 phút thuyết trình + Q&A. Tổng 19 slide chính + 6 slide dự phòng (backup) cho Q&A.
> **Nguyên tắc thiết kế**: mỗi slide MỘT thông điệp; số liệu then chốt phóng to, in đậm; chữ tối thiểu — nói bằng miệng, không đọc slide; dùng lại hình trong cuốn luận (export từ PDF hoặc file trong `images/`).
> **Câu chuyện xuyên suốt (story arc)**: *Encoder tiếng Việt đang tụt hậu về kiến trúc → huấn luyện lại từ đầu là bất khả thi → chúng tôi "cấy" tiếng Việt vào NeoBERT bằng khởi tạo thông minh (SALT cải tiến 3 điểm) → chỉ với ~5 tỷ token (ít hơn đối thủ 1–2 bậc) mà dẫn đầu 4/10 tác vụ.*

---

## PHẦN MỞ (2 slide — 1,5 phút)

### Slide 1 — Trang bìa
- **Nội dung**: Tên đề tài "Nghiên cứu phương pháp xây dựng NeoBERT cho tiếng Việt"; logo trường; 2 SV: Bùi Khắc Trung (22120396) – Trần Anh Tú (22120400); GVHD: TS. Nguyễn Hồng Bửu Long – TS. Lương An Vinh; BM Công nghệ tri thức; 07/2026.
- **Hình**: logo KHTN (`images/logo-khtn.png`).
- **Lời thoại**: Chào hội đồng, giới thiệu nhóm, 1 câu tóm đề tài: "xây dựng phiên bản tiếng Việt của NeoBERT — Encoder hiện đại nhất hiện nay — mà không huấn luyện lại từ đầu."
- **Thời lượng**: 20 giây.

### Slide 2 — Nội dung trình bày
- **Nội dung**: 5 mục: (1) Bối cảnh & vấn đề, (2) Kiến thức nền: NeoBERT & SALT, (3) Phương pháp: quy trình 3 giai đoạn + 3 cải tiến, (4) Thực nghiệm & kết quả, (5) Kết luận & hướng phát triển.
- **Thiết kế**: timeline ngang hoặc list đánh số; sẽ hiện lại thanh tiến trình nhỏ ở góc các slide sau.
- **Thời lượng**: 20 giây.

---

## PHẦN 1 — BỐI CẢNH & VẤN ĐỀ (3 slide — 2,5 phút)

### Slide 3 — Encoder vẫn là "ngựa thồ" của NLU, nhưng đang cũ
- **Thông điệp**: Decoder (LLM) tiến hóa liên tục, Encoder đứng yên ~5 năm.
- **Nội dung** (2 cột đối lập):
  - Cột trái "Decoder-only": LLaMA → DeepSeek… cải tiến kiến trúc liên tục (RoPE, SwiGLU, RMSNorm…).
  - Cột phải "Encoder-only": BERT (2019) → RoBERTa (2019) → … gần như dừng; ngữ cảnh 512 token; kiến thức cũ.
  - Dòng chốt: **các tác vụ NLU thực tế (tìm kiếm, phân loại, hỏi–đáp trích xuất) vẫn chạy bằng Encoder.**
- **Lời thoại**: nhấn nghịch lý "cái được dùng nhiều nhất lại ít được cập nhật nhất".
- **Thời lượng**: 45 giây.

### Slide 4 — NeoBERT (2025): Encoder thế hệ mới — nhưng chỉ có tiếng Anh
- **Nội dung**:
  - NeoBERT = BERT hiện đại hóa: **28 lớp × 768**, RoPE, SwiGLU, Pre-RMSNorm, ngữ cảnh **4.096 token**, huấn luyện **2,1 nghìn tỷ token** RefinedWeb; vượt BERT-large/RoBERTa-large trên MTEB với 250M tham số.
  - Vấn đề: **chỉ tiếng Anh**. Encoder tiếng Việt hiện có (PhoBERT, ViDeBERTa) vẫn là kiến trúc cũ, 512 token.
- **Hình**: Hình 2.2 trong luận (kiến trúc NeoBERT dạng dọc, khung ×28) — export từ PDF trang ~39.
- **Thời lượng**: 45 giây.

### Slide 5 — Ba rào cản & câu hỏi nghiên cứu
- **Nội dung** (3 khối chướng ngại → 1 câu hỏi):
  1. Huấn luyện lại từ đầu: **bất khả thi** (hàng nghìn tỷ token, chi phí khổng lồ).
  2. Thay từ vựng + khởi tạo ngẫu nhiên rồi CPT: **quên lãng thảm họa**, học lại từ đầu.
  3. Lớp nhúng NeoBERT gắn chặt từ vựng tiếng Anh; đầu MLM **tách rời** → phải xử lý CẢ HAI ma trận.
  - **Câu hỏi nghiên cứu**: *Làm sao "cấy" tiếng Việt vào NeoBERT mà kế thừa tối đa những gì nó đã học?*
- **Thiết kế**: 3 icon chướng ngại + mũi tên xuống câu hỏi in đậm ở giữa.
- **Thời lượng**: 60 giây. Đây là slide "đặt cược" — nói chậm, rõ.

---

## PHẦN 2 — KIẾN THỨC NỀN (2 slide — 2 phút)

### Slide 6 — Dòng phương pháp khởi tạo nhúng xuyên ngôn ngữ → SALT
- **Nội dung**: timeline WECHSEL (2022) → FOCUS (2023) → OFA (2024) → **SALT (ACL 2025)**; chung 1 ý tưởng: *token mới không phải ký hiệu rỗng — khởi tạo bằng tri thức có sẵn*.
  - SALT: tái sử dụng (recycle) mô hình tiền huấn luyện ở **ngôn ngữ đích** (donor); dùng token neo trong phần từ vựng trùng để học **ánh xạ tuyến tính cục bộ theo từng token**.
- **Hình**: `images/salt-summary.png` (sơ đồ SALT gốc — Lee et al., ghi nguồn).
- **Lời thoại**: giải thích hình bằng ví dụ "pedestrian": tìm neo gần nghĩa → học map X từ các cặp neo → áp X lên nhúng donor.
- **Thời lượng**: 60 giây.

### Slide 7 — Vì sao ánh xạ CỤC BỘ theo token, không phải một ánh xạ toàn cục?
- **Nội dung** (3 kết quả lý thuyết, mỗi cái 1 dòng):
  - Không gian nhúng **bất đẳng hướng** (nón hẹp) — Ethayarajh 2019.
  - Hai không gian huấn luyện độc lập **không đẳng cấu** — Ormazabal 2019.
  - Buộc trọng số gây **thoái hóa biểu diễn** → đầu vào ≠ đầu ra → phải khởi tạo riêng 2 ma trận — Gao 2019.
  - Chốt: **1 phép biến đổi tuyến tính toàn cục là không đủ → mỗi token một ánh xạ riêng.**
- **Thiết kế**: hình minh họa nón anisotropy (vẽ đơn giản) bên phải.
- **Lời thoại**: đây là nền lý thuyết cho MỌI lựa chọn thiết kế ở phần sau — hội đồng thích slide này.
- **Thời lượng**: 60 giây.

---

## PHẦN 3 — PHƯƠNG PHÁP (6 slide — 5 phút, phần quan trọng nhất)

### Slide 8 — Tổng quan: quy trình 3 giai đoạn
- **Nội dung**: Hình 3.1 trong luận (NeoBERT + ViDeBERTa → GĐ1 Khởi tạo SALT → GĐ2 Căn chỉnh đóng băng → GĐ3 CPT WSD → ViNeoBERT), chú thích dưới mỗi giai đoạn: "thân đóng băng / thân đóng băng / toàn bộ tham số".
  - Khung nhỏ góc: **3 cải tiến so với SALT gốc** (đánh số ① ② ③ — sẽ lần lượt trình bày): ① tập neo mở rộng bằng cặp dịch; ② khởi tạo lớp đầu ra tách rời + bias tần suất + hệ số co γ; ③ giai đoạn căn chỉnh đóng băng.
- **Hình**: export Hình 3.1 (PDF trang ~32).
- **Animation**: hiện từng giai đoạn theo click.
- **Thời lượng**: 60 giây.

### Slide 9 — Cải tiến ①: Tập neo — vấn đề "bạn giả" (false friends)
- **Nội dung** (kể chuyện bằng ví dụ):
  - SALT gốc: neo = token **trùng chuỗi** giữa 2 từ vựng. Nhưng tiếng Việt cũng dùng chữ Latin →
  - Ví dụ to giữa slide: **"song"** 🇻🇳 = song song / nhưng ≠ 🇬🇧 = bài hát; **"sang"** 🇻🇳 = sang trọng ≠ 🇬🇧 = quá khứ của *sing*.
  - Giải pháp: **xác minh nghĩa bằng dịch máy** — chỉ giữ neo khi dịch Việt→Anh trả về đúng chuỗi gốc; giữ **số** (universal) không cần xác minh.
- **Thiết kế**: 2 thẻ từ "song"/"sang" với 2 nghĩa đối lập gạch chéo đỏ.
- **Thời lượng**: 50 giây.

### Slide 10 — Cải tiến ① (tiếp): mở rộng độ phủ bằng cặp dịch 3 tầng
- **Nội dung**:
  - Vấn đề: neo trùng chuỗi + số → ít, chỉ phủ từ vay mượn/tên riêng.
  - Bổ sung nguồn 3: **cặp dịch** ("nước" ↔ "water") qua 3 tầng: T1 dịch ngược (vi→en→vi phải khớp — tự loại từ đa nghĩa), T2 từ ghép, T3 từ đơn có blacklist.
  - Trước đó lọc **chính tả tiếng Việt nghiêm ngặt** (loại f/j/w/z, quy tắc k/gh/ngh, thanh–phụ âm cuối…).
- **Thiết kế**: sơ đồ phễu: ứng viên → bộ lọc chính tả → 3 tầng dịch → tập neo hợp nhất.
- **Lời thoại**: nhấn "dịch ngược khớp = bộ lọc đa nghĩa tự nhiên".
- **Thời lượng**: 50 giây.

### Slide 11 — Phép chiếu SALT theo từng token (cả 2 ma trận)
- **Nội dung** (pipeline 4 bước cho 1 token t):
  1. Vector fastText của t → cosine với các neo.
  2. **Sparsemax** → chỉ k neo liên quan nhất có trọng số ≠ 0 (thưa — hợp không gian bất đẳng hướng).
  3. k ≥ 8: học map cục bộ X = (A_donor)⁺ A_target → e_t = d_t·X; k < 8: trung bình có trọng số.
  4. Áp dụng **2 lần**: đích = hàng nhúng (cho E) và đích = hàng đầu ra (cho W_dec) — vì đầu tách rời.
  - Kèm: hiệu chỉnh norm về trung bình NeoBERT (vì Pre-RMSNorm nhạy độ lớn).
- **Thiết kế**: flow ngang 4 khối; công thức chỉ giữ X = (A^src)⁺A^tgt.
- **Thời lượng**: 60 giây.

### Slide 12 — Cải tiến ②: đầu ra "biết trước" tần suất tiếng Việt + hệ số co γ = 0,1
- **Nội dung** (2 nửa):
  - Nửa trái — **bias = log tần suất từ đơn**: bước 0 mô hình "đoán" theo phân phối tần suất → loss khởi điểm ≈ **entropy từ đơn (~7,3)** thay vì ngẫu nhiên (log V ≈ **10,3**).
  - Nửa phải — **γ = 0,1**: y = γ·W_dec·h + b. Gần 0 → tiên nghiệm dẫn dắt, khỏi học lại phân phối biên; khác 0 → nhân vô hướng **giữ nguyên hướng** các hàng = giữ ngữ nghĩa SALT, sau chỉ khuếch đại dần.
  - Teaser: "γ = 0,5 hay 1,0 thì sao? — mắc kẹt vĩnh viễn. Xem phần kết quả."
- **Thời lượng**: 60 giây.

### Slide 13 — Cải tiến ③ + Giai đoạn 3: căn chỉnh đóng băng & CPT theo lịch WSD
- **Nội dung** (2 nửa):
  - Trái — **Căn chỉnh đóng băng**: đóng băng toàn thân, chỉ luyện {E, W_dec, b} trên ~20M token → 2 ma trận mới "làm quen" với thân trước khi mở khóa toàn bộ → giảm quên lãng.
  - Phải — **CPT WSD**: Hình 3.2 (warmup → ổn định → làm nguội 1−√); điểm ăn tiền: **khóa theo bước toàn cục** → huấn luyện nhiều phiên Colab nối tiếp không re-warm; rẽ nhánh làm nguội giữa chừng để lấy mô hình đánh giá.
- **Hình**: export Hình 3.2 (PDF trang ~61).
- **Thời lượng**: 50 giây.

---

## PHẦN 4 — THỰC NGHIỆM & KẾT QUẢ (5 slide — 4 phút)

### Slide 14 — Thiết lập thực nghiệm
- **Nội dung** (bảng mini 2 cột):
  - Dữ liệu CPT: CulturaX-vi, **5 triệu tài liệu ≈ 5 tỷ token**, seq 1024, MLM 20%.
  - Phần cứng: A100 80GB (Colab), huấn luyện theo phiên 1M tài liệu.
  - Đánh giá: **10 tác vụ / 6 nhóm năng lực** (MRC, NLI ×3, phân loại ×3, POS, STS, PPPL); trung bình 5 seed.
  - Đối sánh: PhoBERT-base-v2, XLM-R-base.
- **Thời lượng**: 40 giây.

### Slide 15 — Kết quả 1: khởi tạo quyết định tất cả (so sánh 8 phương án)
- **Hình chính**: `images/eval_loss_overlay.png` (fullscreen).
- **Chú thích chồng lên hình** (annotation, hiện theo click):
  - Nhóm SALT γ=0,1: xuất phát **7,2 ≈ entropy từ đơn** → hội tụ ~3,0.
  - γ=1,0 xuất phát **10,1 ≈ ngẫu nhiên** (logit nhấn chìm bias); γ=0,5: 8,1 → **cả hai mắc kẹt 6,6–6,9**.
  - Random meannorm hội tụ 3,5 → khoảng cách còn lại với SALT = giá trị của **ngữ nghĩa** được chuyển giao.
- **Lời thoại**: đây là bằng chứng trực tiếp cho giả thuyết γ ở Slide 12.
- **Thời lượng**: 60 giây.

### Slide 16 — Kết quả 2: thứ hạng giữ nguyên ở hạ nguồn — chọn phương án đầy đủ
- **Hình chính**: `images/hardtask_MRC-ViQuAD_bar.png`.
- **Chú thích**: freezealigned (~68) > decpertoken (~67,5) > globalmap (~66) ≫ random tốt nhất (~56); **γ sai tụt DƯỚI cả random (42–47)** → khởi tạo sai đầu ra gây hại vĩnh viễn.
- **Chốt slide**: mỗi thành phần thiết kế đều đóng góp dương → chọn **per-token + freeze-align** làm mô hình chính.
- **Thời lượng**: 45 giây.

### Slide 17 — Kết quả 3: đường cong theo ngân sách
- **Hình chính**: `images/mrc_learning_curve.png`.
- **Chú thích**: vượt XLM-R-base chỉ sau **0,5M tài liệu (~10% ngân sách)**; chững 1–4M rồi **bật lên ở 5M** nhờ pha làm nguội trên dữ liệu mới → còn dư địa tăng.
- **Thời lượng**: 40 giây.

### Slide 18 — Kết quả 4: bảng đa nhiệm + "proof the point"
- **Nội dung**:
  - Bảng rút gọn từ Bảng 4.4 (chỉ 5–6 hàng nổi bật): MRC **76,24** (XLM-R 72,10), PPPL **2,85**, POS **0,8089**, XNLI **0,7706** — 4 hàng thắng in xanh; 1–2 hàng PhoBERT thắng (VSFC, ViANLI) để thể hiện trung thực.
  - Khối "proof the point" (to, giữa slide): **ViNeoBERT chỉ "đọc" ~5 tỷ token tiếng Việt** — PhoBERT: 40 epoch × 20GB (~30× nhiều hơn) + bản v2 thêm 120GB OSCAR; XLM-R: 2,5TB CC-100 (riêng tiếng Việt ~137GB). → **Ít hơn 1–2 bậc dữ liệu mà dẫn đầu 4/10, ≥ XLM-R-base ở 8/10.**
- **Lời thoại**: nhấn mạnh nhóm thắng = tác vụ cần biểu diễn ngữ cảnh sâu (đọc hiểu, POS, LM) — đúng năng lực kế thừa từ thân NeoBERT.
- **Thời lượng**: 75 giây. Slide đinh — dừng lâu nhất.

---

## PHẦN 5 — KẾT (2 slide — 1,5 phút)

### Slide 19 — Kết luận
- **Nội dung** (4 gạch, mirror mục tiêu Chương 1):
  - Xây dựng thành công **ViNeoBERT** — Encoder kiến trúc 2025 đầu tiên cho tiếng Việt (245M, kế thừa trần 4.096 token).
  - Quy trình 3 giai đoạn tái lập được + **3 cải tiến** so với SALT gốc (neo cặp dịch, đầu ra tách rời + bias + γ, freeze-align) — **mỗi cải tiến đều có bằng chứng ablation**.
  - Hiệu quả dữ liệu: 5 tỷ token → dẫn đầu 4/10 tác vụ, ≥ XLM-R-base 8/10.
  - Mã nguồn công khai.
- **Thời lượng**: 45 giây.

### Slide 20 — Hướng phát triển + Cảm ơn
- **Nội dung**: 4 mũi tên: tăng ngân sách CPT (+ pha làm nguội dữ liệu mới); kiểm chứng ngữ cảnh dài 4.096 (YaRN); mở rộng đánh giá (retrieval, thêm 2 model large); tổng quát hóa quy trình cho ngôn ngữ ít tài nguyên khác.
  - Dòng cuối: "Em xin cảm ơn hội đồng. Nhóm sẵn sàng trả lời câu hỏi."
- **Thời lượng**: 30 giây.

---

## SLIDE DỰ PHÒNG (backup — chỉ mở khi hội đồng hỏi)

### B1 — Chi tiết bộ lọc chính tả tiếng Việt (9 quy tắc, từ Phụ lục A)
### B2 — Toán đầy đủ phép chiếu: sparsemax + pinv + norm-calib (công thức 3.2–3.6)
### B3 — Bảng 4.4 đầy đủ 10 tác vụ (nếu hội đồng muốn xem hết)
### B4 — Bảng siêu tham số 3 giai đoạn (Bảng 4.2) + lịch WSD chi tiết, cơ chế resume phiên
### B5 — Vì sao ViDeBERTa làm donor (128k vocab Unigram → cắt tỉa còn 30.522) + quy trình đồng bộ từ vựng (eq 3.1)
### B6 — Hạn chế & trả lời trước các câu hỏi dự kiến:
  - "Sao chưa so PhoBERT-large?" → đang bổ sung, đường cong Hình 4.3 đã có tham chiếu large.
  - "Vì sao thua PhoBERT ở phân loại câu ngắn?" → tách từ + 40 epoch dữ liệu trong miền; các tác vụ này bão hòa (VSFC chênh <1 điểm).
  - "PPPL so giữa các tokenizer khác nhau có công bằng?" → chỉ mang tính tham khảo tương đối (đã ghi chú trong luận).
  - "4.096 token đã kiểm chứng chưa?" → kiến trúc kế thừa, CPT ở 1.024; kiểm chứng dài là hướng phát triển (đã ghi rõ phạm vi).

---

## CHECKLIST DỰNG SLIDE

1. **Export hình từ PDF luận** (400 DPI): Hình 2.2 (tr.~39), Hình 3.1 (tr.~32), Hình 3.2 (tr.~61) — dùng `pdftoppm -r 400 -png`.
2. Hình PNG có sẵn trong `images/`: `salt-summary.png`, `eval_loss_overlay.png`, `hardtask_MRC-ViQuAD_bar.png`, `mrc_learning_curve.png`.
3. Font chữ slide ≥ 20pt; số liệu đinh (76,24 / 4-10 / 8-10 / 5 tỷ / γ=0,1 / 7,2 vs 10,3) cỡ ≥ 40pt.
4. Màu nhất quán với hình sẵn có: xanh dương = ViNeoBERT/SALT, đỏ = cảnh báo/mắc kẹt, xám = baseline.
5. Tổng thời gian tập dượt: đọc to toàn bộ ≤ 14 phút (chừa 1 phút trễ máy).
6. Trang bị trước phiên bản tiếng Anh của thuật ngữ chính phòng khi hội đồng hỏi bằng thuật ngữ gốc: anchor set, per-token projection, untied head, frequency-prior bias, weight scaling, freeze-align, WSD schedule.
