# Quan điểm sửa bài của thầy & Kế hoạch viết lại toàn bộ khóa luận

> Bối cảnh: Thầy **chỉ đọc 2 trang mục lục, gạch gần hết và cho 0 điểm**. Điều này cực kỳ quan trọng: nó nói rằng vấn đề **KHÔNG** nằm ở nội dung khoa học (thầy chưa đọc tới), mà nằm ở **văn phong & thuật ngữ** — thứ lộ ra ngay từ tiêu đề mục. Mục lục là "bộ mặt" của bài; nếu tiêu đề đọc vào không hiểu / dịch sượng, thầy kết luận cả bài cẩu thả.

---

## 1. Chẩn đoán: vì sao thầy gạch (rút từ chính các mục bị gạch)

Mọi mục bị gạch đều rơi vào **một trong bốn lỗi** sau — tất cả đều là hệ quả của **dịch 1-1 máy móc từ tiếng Anh**:

| # | Kiểu lỗi | Ví dụ thầy gạch | Vì sao sai |
|---|----------|-----------------|-----------|
| 1 | **Dịch từng chữ (calque)** tạo cụm không tồn tại trong tiếng Việt học thuật | *chuyển giao không gian nhúng*, *quên lãng thảm họa*, *cắt tỉa từ vựng* | Người Việt đọc không ra nghĩa; nghe như Google Translate. (LƯU Ý: *"ngôn ngữ có mặt nạ"* KHÔNG phải calque — đó là thuật ngữ chuẩn cho MLM) |
| 2 | **Thuật ngữ tự chế, mờ nghĩa** | *mô hình hiến tặng*, *neo trùng lặp bề mặt*, *neo số*, *căn chỉnh đóng băng* | Không phải thuật ngữ chuẩn; đọc xong vẫn không biết nó là gì |
| 3 | **Sai register / từ không thông dụng** | *diễn tiến*, *đa nhiệm* | Có từ khoa học thông dụng hơn (*diễn biến*, *đa tác vụ*) |
| 4 | **Jargon dịch ép thay vì giữ/nói rõ** | *nhúng* (embedding), *bộ tách từ* (tokenizer, lại trùng nghĩa "tách từ" của VnCoreNLP) | Gây rối; nên chọn từ rõ hoặc giữ nguyên tiếng Anh có chú thích |

**Kết luận cốt lõi — điều thầy muốn:**
> *"Dịch sao cho đọc vào là hiểu ngay, không dịch 1-1. Ưu tiên từ tiếng Việt đã có/chuẩn hóa; thuật ngữ chưa có bản dịch tốt thì giữ tiếng Anh kèm chú thích; một khái niệm gọi một tên nhất quán xuyên suốt."*

Đây cũng đúng tinh thần `CLAUDE.md` (Phần V.3 – xử lý thuật ngữ Việt–Anh, và Phần XIV – các lỗi hình thức thường gặp).

---

## 2. Bảng thuật ngữ chuẩn đã chốt (đã áp dụng cho mục lục + toàn thân bài)

Dùng bảng này làm **từ điển khóa** cho mọi chỉnh sửa về sau — tuyệt đối nhất quán.

| Tiếng Anh / cũ (đã gạch) | ✅ Dùng thống nhất | Ghi chú |
|--------------------------|--------------------|---------|
| masked language modeling / mask | **mặt nạ** ("ngôn ngữ có mặt nạ", "token mặt nạ", "mặt nạ động/nhân quả") | ⚠️ ĐÃ HOÀN NGUYÊN (10/07): "đánh dấu" là SAI — chuẩn d2l + FAIR peer-reviewed đều dùng **mặt nạ** |
| embedding / *nhúng* | **véc-tơ** | *không gian véc-tơ, lớp véc-tơ, ma trận véc-tơ, hàng véc-tơ*; gộp "véc-tơ nhúng" → "véc-tơ" |
| transfer | **chuyển giao** | ⚠️ ĐÃ ĐẢO (10/07): "hoán chuyển" là từ lạ → dùng "chuyển giao" (chuẩn cho transfer/transfer learning) |
| adaptation / *thích nghi* | **thích ứng** ("thích ứng miền", "tiền huấn luyện thích ứng") | ĐÃ ĐỔI (10/07): FAIR peer-reviewed dùng chuẩn "thích ứng miền" |
| language adaptation / *thích nghi ngôn ngữ* | **chuyển đổi ngôn ngữ** | |
| output-weight scale γ | γ = **hệ số tỉ lệ**; thao tác = **giảm biên độ trọng số** | thống nhất TOÀN bài (Ch1/Ch5 từng sót "hệ số co" — đã sửa) |
| catastrophic forgetting / *quên lãng thảm họa* | **mất thông tin** | |
| donor model / *mô hình hiến tặng* | **mô hình nguồn véc-tơ** | NeoBERT = **mô hình đích**; donor embedding = **véc-tơ nguồn** |
| tokenizer / *bộ tách từ* | **tokenizer** (lần đầu: *bộ token hóa (tokenizer)*) | "tách từ" **chỉ** dành cho phân đoạn từ VnCoreNLP → tránh nhầm |
| vocabulary pruning / *cắt tỉa* | **giảm kích thước** (từ vựng) | |
| anchor set / *tập neo* | **tập điểm neo**; anchor = **điểm neo** | |
| surface-overlap anchor / *neo trùng lặp bề mặt* | **điểm neo trùng mặt chữ** | |
| numeric anchor / *neo số* | **điểm neo chữ số** | |
| frozen-alignment stage / *căn chỉnh đóng băng* | **căn chỉnh với phần thân đóng băng** | nói rõ *cái gì* bị đóng băng |
| Semantic Aware Linear Transfer | **Chuyển đổi tuyến tính trong không gian có ngữ nghĩa** (Semantic Aware Linear Transfer – SALT) | chú nghĩa lần đầu, sau đó dùng **SALT** |
| multi-task / *đa nhiệm* | **đa tác vụ** | |
| progression / *diễn tiến* | **diễn biến** | |
| random seed / *hạt giống* | **seed ngẫu nhiên** (lần đầu: *(random seed)*) | chỗ hợp lý dùng "các lần chạy" (vd *độ lệch chuẩn giữa các lần chạy*) |
| token | **token** (giữ nguyên tiếng Anh) | đã tra: bài NLP VN peer-reviewed giữ "token"; dịch "đơn vị từ" SAI vì token (tiểu từ) ≠ từ |
| downstream / *hạ nguồn* | **tác vụ ứng dụng** (hoặc bỏ hẳn khi ngữ cảnh đã rõ) | thầy gạch "hạ nguồn"; *đánh giá hạ nguồn*→*đánh giá*, *tác vụ hạ nguồn*→*tác vụ ứng dụng* |
| output-weight scaling / *co / thu nhỏ* | **giảm biên độ trọng số** (thao tác); $\gamma$ = **hệ số tỉ lệ**; mô tả: *"nhân trọng số lớp đầu ra với một hệ số nhỏ"* | thầy hỏi "thu nhỏ…là gì / thu nhỏ véc-tơ à"; giữ "co giãn" ở chỗ rescale khác |
| frequency prior / *tiên nghiệm tần suất* | **phân phối tần suất** (prior tổng quát → *phân phối/hằng số định sẵn*) | thầy hỏi "tiên nghiệm là gì" — bỏ jargon Bayes |
| prior-guided / *(tiên nghiệm/tần suất) dẫn dắt* | **phân phối tần suất chi phối** | thầy hỏi "dẫn dắt là gì"; đã tra: không có bản dịch chuẩn VN → chọn "chi phối" (đã dùng sẵn trong mục) |
| *Vietnamese NeoBERT* | **NeoBERT cho tiếng Việt** (viết tắt ViNeoBERT) | khớp tên đề tài; tiêu đề Ch3 + các chỗ định nghĩa |
| tokens seen / *token đã duyệt/xử lý* | **token đã thấy** (*"mô hình đã thấy X token"*; "một lượt/epoch" dùng "đi qua") | metaphor "tokens seen"; user chốt. GIỮ *"lượt duyệt qua d phần tử"* (duyệt mảng thuật toán — nghĩa khác) |

---

## 3. Những gì ĐÃ sửa xong (đợt 1)

- **Trang bìa:** ngắt dòng tên đề tài để "xây dựng…" xuống dòng (macro `\tenKLcover` mới; giữ `\tenKL` cho lời cam đoan inline).
- **Lời cảm ơn:** chừa `2.5cm` dưới "Nhóm sinh viên thực hiện" để ký tên.
- **245 → 250 triệu tham số** (6 chỗ, Ch1/Ch2/Ch4) cho khớp bài NeoBERT gốc.
- **Toàn bộ thuật ngữ trong Bảng mục 2** đã quét sạch trên: Ch1–Ch5, Tóm tắt (`summary.tex`) — kể cả các cụm bị **ngắt dòng** mà sed bỏ sót.
- **Mục lục (`main.toc`) đã tự sinh lại** với tiêu đề mới; **biên dịch `latexmk` thành công, không lỗi.**

---

## 4. Kế hoạch viết lại TOÀN BỘ bài (đợt 2 — chưa làm)

Đợt 1 mới xử lý *thuật ngữ ở tiêu đề + quét từ khóa trong thân*. "Bệnh dịch 1-1" gần như chắc chắn còn nằm rải rác trong **câu chữ thân bài**. Kế hoạch:

**Bước 1 — Rà từng chương theo "bộ lọc dịch sượng".** Với mỗi đoạn, tự hỏi: *"Câu này có phải đang dịch nguyên văn tiếng Anh không? Người Việt sẽ nói thế nào?"* Sửa:
- Câu bị động kiểu Anh ("được thực hiện bởi…") → chủ động tiếng Việt.
- Cụm danh từ chồng chất (calque) → tách mệnh đề.
- Code-mixing văn nói ("train/fine-tune/dataset") → thuật ngữ Việt.

**Bước 2 — Chuẩn hóa theo `CLAUDE.md`:**
- Ngôi xưng: **"chúng tôi"** cho đóng góp, **"ta"** khi dẫn suy luận — nhất quán.
- Mỗi thuật ngữ Anh–Việt: giải thích **lần đầu**, sau đó một dạng duy nhất.
- Mỗi **hình/bảng** phải được **dẫn + phân tích** (không "mồ côi", không bảng "trơ").
- Bảng kết quả: in đậm giá trị tốt nhất + mũi tên ↑/↓ + chú thích.

**Bước 3 — Rà lại 3 mục lục (mục lục / danh sách hình / danh sách bảng)** sau khi mọi tiêu đề/caption đã sửa: đảm bảo tất cả caption hình–bảng cũng sạch thuật ngữ (đợt 1 chưa quét sâu caption).

**Bước 4 — Soát chính tả & thuật ngữ lần cuối** toàn bài; build lại `latexmk` và đọc lại 2 trang mục lục bằng con mắt của thầy.

**Thứ tự ưu tiên chương:** ~~Ch3~~ ✅ → ~~Ch2~~ ✅ → ~~Ch4~~ ✅ → ~~Ch1~~ ✅ → ~~Ch5~~ ✅ → Tóm tắt/Phụ lục (còn lại).

### Rà thuật ngữ theo CHUẨN d2l (10/07) — làm giàu nguồn + soát toàn bài
- **Nguồn tham khảo mở rộng:** thêm 3 bài peer-reviewed (FAIR 2023/2019, HUFLIT) + bảng chuẩn `MyCode/Example_Paper/CHUAN_THUAT_NGU_d2l.md`.
- **Sửa theo chuẩn thực tế:** mask *đánh dấu*→**mặt nạ**; tokens seen *xử lý*→**đã thấy**; transfer *hoán chuyển*→**chuyển giao**; adaptation *thích nghi*→**thích ứng**.
- **Soát nhất quán TOÀN bài (Ch1–Ch5+Phụ lục):** bắt & sửa stragglers mà các sweep trước bỏ sót ở Ch1/Ch2/Ch5 — *hiệu chỉnh chuẩn*→*độ lớn* (Ch2×2, Ch5), *hệ số co*→*hệ số tỉ lệ* (Ch1, Ch5), *mất thông tin thảm họa*→*mất thông tin* (Ch5, do ngắt dòng). **Quét cuối: sạch hoàn toàn, không còn thuật ngữ cũ.** Tên riêng (NeoBERT/PhoBERT…) nhất quán trong văn xuôi. Build `latexmk` sạch.

- **Chương 4 — ✅ ĐÃ XONG (2026-07-10):** sửa 2.3.3 (*"…có thông tin"*→*"…dựa trên ngữ nghĩa"*). Rà thân Ch4: bảng đa tác vụ đạt chuẩn (mũi tên ↑ mọi cột độ đo, in đậm giá trị tốt nhất + chú thích, 4 nhận xét phân tích); mọi hình/bảng đều được dẫn + phân tích, ngôi xưng nhất quán. Sửa: 3 chỗ "co"→"thu nhỏ" bị ngắt dòng sed bỏ sót; 4 chỗ *"vectơ"*→*"véc-tơ"*; caption bảng đa tác vụ *"tại mốc 5 tỷ token"*→*"tại điểm kiểm tra 5 tỷ token"*. Build sạch.

### Tiến độ đợt 2
- **Chương 3 — ✅ ĐÃ XONG (2026-07-10):** rà câu chữ, sửa calque bị động (*"BPE được sử dụng bởi PhoBERT"* → *"PhoBERT dùng BPE"*; *"được ViDeBERTa sử dụng"* → *"ViDeBERTa sử dụng"*), *"việc thích nghi sang tiếng Việt"* → *"việc chuyển sang tiếng Việt"*, *"không ràng buộc"* → *"không phụ thuộc"*. Kiểm tra: ngôi xưng "chúng tôi" nhất quán; mọi hình/bảng/thuật toán đều được dẫn + phân tích (không mồ côi). Build `latexmk` sạch.

> ⚠️ **BÀI HỌC QUAN TRỌNG (áp dụng cho các chương còn lại):** đợt-1 quét sed toàn cục đã gây **3 lỗi substring/định danh** trong Ch3, phải kiểm & sửa khi rà từng chương:
> 1. `s/diễn tiến/diễn biến/` ăn nhầm **"diễn tiếng"** → "diễn biếng" (calque substring). → Luôn grep `biếng`, và kiểm các từ dễ bị ăn nhầm.
> 2. `s/\bneo\b/điểm neo/` đổi nhầm **định danh toán** `\mathrm{neo}` → `\mathrm{điểm neo}` (render "điểmneo") và **tên node TikZ** `(neo)`. → Với mỗi chương, grep `\mathrm{điểm neo}`, `(điểm neo)`, và mọi "điểm neo" nằm trong `$...$`/`\mathrm{}`/tikz; hoàn nguyên về định danh kỹ thuật gốc.
> 3. Cụm bị **ngắt dòng** (sed theo dòng bỏ sót) → dùng `perl -0777` với `\s+` để bắt.

- **Chương 2 — ✅ ĐÃ XONG (2026-07-10):** rà toàn bộ. Kết quả **rất sạch** — không dính lỗi sed (đã grep `\mathrm{điểm neo}`, tikz, `biếng`, double: đều trống), không còn "nhúng", ngôi xưng "chúng tôi" nhất quán, mọi hình/bảng (Hình 2.1–2.5, Bảng 2.1–2.2) đều được dẫn + phân tích, 2 chỗ "bởi" là bị động toán học hợp lệ (*bị chặn/giới hạn bởi*). Chỉ chỉnh 1 chỗ dồn "véc-tơ" (*"chiều véc-tơ của mô hình nguồn véc-tơ ViDeBERTa-base… phép chiếu không gian véc-tơ"* → *"chiều véc-tơ của ViDeBERTa-base (mô hình nguồn véc-tơ)… phép chiếu"*). Build sạch.
> 💡 **Lưu ý phong cách phát sinh:** thuật ngữ *"mô hình nguồn véc-tơ"* đôi khi đứng cạnh "véc-tơ" khác gây dồn từ — khi rà Ch4/Ch1/Ch5 để ý làm mượt các chỗ tương tự.

### ⭐ BƯỚC 0 BẮT BUỘC MỖI CHƯƠNG: kiểm TIÊU ĐỀ trước (thầy chấm mục lục!)
Tôi từng sai khi khen "Ch3 tốt" dựa trên văn xuôi — nhưng **thầy gạch 0đ vì tiêu đề mục 3.5** (jargon nén). Rút kinh nghiệm: **soi từng tiêu đề qua phép thử "đọc lướt có hiểu ngay?" TRƯỚC khi rà câu chữ.**

- **Mục 3.5 — ĐÃ SỬA (2026-07-10):** viết lại cả 7 tiêu đề theo hướng *làm gì / để làm gì*:
  - 3.5.6 *"Co trọng số đầu ra để tiên nghiệm dẫn dắt giai đoạn đầu"* → *"Thu nhỏ trọng số lớp đầu ra để ổn định giai đoạn huấn luyện đầu"*
  - 3.5.5 *"…theo tần suất"* → *"…theo tần suất từ vựng tiếng Việt"*
  - 3.5.4 *"Hiệu chỉnh chuẩn của lớp véc-tơ"* → *"Hiệu chỉnh độ lớn của các véc-tơ cho khớp với NeoBERT"*
  - (+ 3.5, 3.5.1–3.5.3, và 4.4.2). Đồng bộ "co"→"thu nhỏ", "hiệu chỉnh chuẩn"→"hiệu chỉnh độ lớn" khắp Ch3/Ch4/Tóm tắt; chú thích *độ chệch (bias)*, *thu nhỏ (co, scale)*.
- **Đã rà TẤT CẢ tiêu đề Ch1/Ch2/Ch4/Ch5:** phần lớn **đạt** phép thử (đều là khái niệm/kỹ thuật có tên rõ). 3.5 là ổ jargon tập trung (đúng chỗ thầy chỉ). Borderline nhẹ: 2.3.3 *"Khởi tạo véc-tơ có thông tin"* ("có thông tin" hơi trừu tượng) — có thể chỉnh nếu cần.

- **Gộp mục con 3.5 + sửa Ch4 (2026-07-10):**
  - **3.5 gộp 6 → 4 mục con:** 3.5.4/3.5.5/3.5.6 (ba bước tinh chỉnh nhỏ) gộp thành **3.5.4 "Tinh chỉnh kết quả khởi tạo: độ lớn véc-tơ, độ chệch và trọng số lớp đầu ra"** với 3 đoạn in đậm (giữ nhãn `\ref` cũ → cùng trỏ 3.5.4).
  - **Ch4 tiêu đề:** 4.1.3 *"Dữ liệu đánh giá hạ nguồn"*→*"Dữ liệu đánh giá"*; 4.4.3 bỏ *"hai"* → *"…khởi tạo ma trận từ vựng"*; 4.5.1 *"Diễn biến theo ngân sách huấn luyện"*→*"Diễn biến kết quả theo lượng dữ liệu huấn luyện"*; 4.5.2 bỏ *"tại mốc 5 tỷ token"* → *"Đánh giá đa tác vụ"*.
  - **"hạ nguồn" bỏ toàn bài** (18 chỗ); **"token" giữ nguyên** (đã kiểm chứng).

> Học văn phong: đối chiếu với 3 bài mẫu tiếng Việt trong `MyCode/Example_Paper/` (đặc biệt bài PhoBERT ĐH Huế — cùng chủ đề).
