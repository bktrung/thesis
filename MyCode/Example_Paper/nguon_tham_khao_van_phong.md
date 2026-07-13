# Nguồn tham khảo văn phong khoa học tiếng Việt (NLP / AI / LLM)

> **Mục đích:** Tập hợp các bài báo/công bố khoa học **viết bằng tiếng Việt**, đã qua bình duyệt, thuộc lĩnh vực NLP / AI / LLM — để học theo **văn phong, bố cục, cách trích dẫn, cách trình bày hình–bảng–công thức** chuẩn mực. Dùng kèm với checklist trong `CLAUDE.md`.
>
> **Ngày lập:** 09/07/2026.

> 📥 **Đã tải sẵn 6 bài peer-reviewed (md, đọc offline) + 1 bảng chuẩn thuật ngữ** trong folder này:
> - `01_HUFLIT_2024_Mo-hinh-ngon-ngu-lon-va-ung-dung.md` — LLM
> - `02_DHHue_2022_Cai-tien-PhoBERT-chatbot.md` — PhoBERT (sát đề tài)
> - `03_HaUI_2025_SVM-phan-loai-cam-xuc.md` — SVM phân loại cảm xúc
> - `04_FAIR2023_Hoi-dap-tu-dong-du-lich.md` — QA + BERT/RoBERTa/PhoBERT (kỷ yếu FAIR)
> - `05_FAIR2019_Thich-ung-mien-dich-may-noron.md` — dịch máy nơ-ron, **thích ứng miền** (kỷ yếu FAIR)
> - `06_HUFLIT_Gom-cum-bai-bao-PhoBERT.md` — gom cụm bằng PhoBERT
> - ⭐ **`CHUAN_THUAT_NGU_d2l.md`** — bảng đối chiếu thuật ngữ ML Anh–Việt chuẩn (d2l) + **các chỗ luận đang lệch chuẩn** (mask, prior, downstream…). Đọc file này TRƯỚC khi sửa thuật ngữ.
> - PDF gốc trong `pdfs/`.
>
> (Text trích tự động bằng `pdftotext`; cần trích dẫn chính xác thì mở PDF gốc.)

---

## ⚠️ Lưu ý quan trọng về "nguồn uy tín" khi học văn phong TIẾNG VIỆT

Cần phân biệt rõ hai chuyện:

- **Nơi làm nghiên cứu AI mạnh nhất VN** (VinAI, VinUni, HUST/SoICT, UET-VNU, VNU-HCM, FPT AI…) — các lab này công bố **gần như hoàn toàn bằng tiếng Anh** ở ACL, EMNLP, NeurIPS, arXiv. **Không** phải nơi để học *văn phong tiếng Việt*.
- **Nơi có văn phong khoa học TIẾNG VIỆT chuẩn để học theo** — nằm ở **các hội nghị quốc gia** (FAIR, VNICT/@) và **tạp chí khoa học của các trường/viện** (bình duyệt, có ISSN/ISBN). Đây mới là nguồn đúng cho mục tiêu của bạn.

👉 Vì luận của bạn viết bằng tiếng Việt, hãy ưu tiên nhóm thứ hai. Các bài dưới đây đều **đúng lĩnh vực** (PhoBERT, LLM, phân loại/tóm tắt văn bản tiếng Việt) nên vừa học được văn phong, vừa học được cách trình bày đúng chủ đề của bạn.

---

## 1. Bài báo tiếng Việt SÁT chủ đề luận (PhoBERT / LLM / NLP tiếng Việt)

Đây là nhóm nên đọc kỹ nhất — cùng "họ" với đề tài ViNeoBERT/SALT của bạn.

### [1] Mô hình ngôn ngữ lớn và ứng dụng (2024) — ⭐ ưu tiên
- **Tác giả:** Trần Khải Thiện, Đinh Minh Hòa, Phạm Ngọc Bảo, Huỳnh Vũ Lê, Lê Huỳnh Nghiêm, Nguyễn Thị Thúy An.
- **Nguồn:** Tạp chí Khoa học HUFLIT, Tập 8, Số 3 (2024). ISSN riêng của tạp chí.
- **Nội dung:** Giới thiệu LLM, kiến trúc, công cụ xây trên LLM, thách thức & rủi ro, case study phân loại cảm xúc.
- **Học được gì:** Cách viết phần tổng quan/khái niệm về LLM **bằng tiếng Việt**, xử lý thuật ngữ Anh–Việt, cách dẫn dắt từ nền tảng → ứng dụng.
- **Link xem:** https://hjs.huflit.edu.vn/index.php/hjs/article/view/206
- **Link tải PDF:** https://hjs.huflit.edu.vn/index.php/hjs/article/download/206/121/967

### [2] Một cải tiến của PhoBERT nhằm tăng khả năng hiểu tiếng Việt của chatbot thông tin khách sạn (2022) — ⭐ ưu tiên
- **Tác giả:** Ngô Văn Sơn, Nguyễn Thị Minh Nghĩa, Hoàng Thị Huế, Nguyễn Hữu Liêm, Võ Viết Minh Nhật (Đại học Huế).
- **Nguồn:** Tạp chí Khoa học Đại học Huế: Kỹ thuật và Công nghệ, Vol. 131, No. 2A (2022).
- **Nội dung:** Cải tiến mô hình tiền huấn luyện PhoBERT cho chatbot tiếng Việt. Kết quả: Accuracy 96.4%, F1 96.9%, Precision 97.4%.
- **Học được gì:** **Rất sát đề tài của bạn** — cách mô tả PhoBERT/pretraining, cách trình bày bảng kết quả + độ đo (Accuracy/F1/Precision), cách diễn giải bảng bằng tiếng Việt.
- **Link xem:** https://jos.hueuni.edu.vn/index.php/hujos-tt/article/view/6978
- **Link tải PDF:** https://jos.hueuni.edu.vn/index.php/hujos-tt/article/download/6978/1618

### [3] Gom cụm bài báo khoa học dựa trên PhoBERT
- **Nguồn:** Tạp chí Khoa học HUFLIT.
- **Nội dung:** Dùng PhoBERT để gom cụm/phân loại bài báo (thử nghiệm trên dữ liệu hội nghị FAIR).
- **Học được gì:** Quy trình tiền xử lý → embedding PhoBERT → phân loại (softmax), trình bày bằng tiếng Việt.
- **Link xem:** https://hjs.huflit.edu.vn/index.php/hjs/article/view/147

### [4] Xây dựng ứng dụng AI hỗ trợ tóm tắt bài đọc tiếng Việt cho học sinh tiểu học
- **Nguồn:** Tạp chí Khoa học Trường Đại học Quốc tế Hồng Bàng.
- **Nội dung:** Kết hợp PhoBERT (tóm tắt trích xuất) + mT5 (tóm tắt diễn giải), tinh chỉnh trên ~6.000 bài đọc.
- **Học được gì:** Cách mô tả pipeline hai mô hình, cách viết phần dữ liệu & thực nghiệm.
- **Link xem:** https://tapchikhoahochongbang.vn/js/article/view/1098

---

## 2. Bài báo tiếng Việt về ML/NLP (học thêm văn phong & cấu trúc)

### [5] An Investigation of Vietnamese Document Classification
- **Nguồn:** Journal of Applied Science and Technology, ĐH Sư phạm Kỹ thuật Hưng Yên (UTEHY).
- **Link:** http://jst.utehy.edu.vn/index.php/jst/article/view/397

### [6] Ứng dụng mô hình học máy SVM trong phân loại (văn bản/bình luận tiếng Việt)
- **Nguồn:** Tạp chí Khoa học và Công nghệ, ĐH Công nghiệp Hà Nội (HaUI).
- **Link tải PDF:** https://jst-haui.vn/media/32/uffile-upload-no-title32083.pdf

### [7] Phân tích diễn ngôn (ứng dụng NLP)
- **Nguồn:** Tạp chí Khoa học và Công nghệ, ĐH Đà Nẵng (JST-UD).
- **Link tải PDF:** https://jst-ud.vn/jst-ud/article/download/7597/5445/7829

---

## 3. Các VENUE uy tín để tự tìm thêm bài (bình duyệt, tiếng Việt)

Vào các trang dưới, tìm chuyên mục **"Xử lý ngôn ngữ tự nhiên" / "Trí tuệ nhân tạo" / "Học máy"** rồi lọc các bài có **toàn văn tiếng Việt**:

### Hội nghị quốc gia (kỷ yếu có ISBN — chuẩn mực để học)
- **FAIR** — Nghiên cứu cơ bản và ứng dụng Công nghệ thông tin (có hẳn Tiểu ban Xử lý ngôn ngữ tự nhiên): https://fair.conf.vn/
- **VNICT / @** — Một số vấn đề chọn lọc của CNTT & Truyền thông: https://hoithaoquocgiacntt.ac.vn/ và https://vnict.vn/

### Tạp chí khoa học (bình duyệt, có ISSN)
- **Journal of Computer Science and Cybernetics (Tin học và Điều khiển học)** — Viện Hàn lâm KHCN VN (VAST); có bài tiếng Việt lẫn tiếng Anh. jcc@vast.ac.vn
- **VNU Journal of Science** — ĐHQG Hà Nội: https://js.vnu.edu.vn/
- **Tạp chí Phát triển Khoa học & Công nghệ** — ĐHQG-HCM: https://vnuhcmpress.org/
- **Tạp chí Khoa học Đại học Huế: Kỹ thuật và Công nghệ:** https://jos.hueuni.edu.vn/
- **Tạp chí Khoa học HUFLIT:** https://hjs.huflit.edu.vn/
- **Tạp chí Khoa học & Công nghệ** các trường: ĐH Đà Nẵng (jst-ud.vn), ĐH Công nghiệp Hà Nội (jst-haui.vn), UTEHY (jst.utehy.edu.vn)…

---

## 4. Cách DÙNG các bài này để sửa luận (không đọc suông)

Khi đọc mỗi bài, soi đúng những điểm mà `CLAUDE.md` yêu cầu — ghi chú lại mẫu câu để bắt chước:

1. **Tóm tắt (Abstract):** xem họ nén "bối cảnh → mục tiêu → phương pháp → kết quả" trong ½–1 trang thế nào.
2. **Ngôi xưng:** để ý họ dùng **"chúng tôi"** cho đóng góp, **"ta"** khi dẫn dắt suy luận — nhất quán ra sao.
3. **Thuật ngữ Anh–Việt:** cách họ chú tiếng Anh trong ngoặc ở lần đầu (vd: *cơ chế tự chú ý (self-attention)*), sau đó dùng nhất quán.
4. **Bảng kết quả:** cách in đậm giá trị tốt nhất, ghi mũi tên ↑/↓, và **đoạn văn diễn giải** bảng (không để bảng "trơ").
5. **Trích dẫn:** cách đặt `[n]` ngay tại điểm phát biểu, và định dạng đồng nhất ở danh mục tài liệu.
6. **Câu chữ:** câu đủ chủ–vị, mỗi đoạn một ý, từ nối logic (*do đó, tuy nhiên, cụ thể, đáng chú ý…*), **không** code-mixing kiểu văn nói ("train model trên dataset").

> 💡 Gợi ý: chọn **[1]** và **[2]** làm "bài mẫu chính" vì đúng chủ đề LLM/PhoBERT của bạn. Đọc kỹ, đối chiếu từng mục với chương tương ứng trong luận, và viết lại các đoạn bị thầy chê theo đúng mạch văn của chúng.

---

*Ghi chú: link được xác minh ngày 09/07/2026. Nếu link tải PDF trực tiếp bị đổi, dùng link "xem" để vào trang bài rồi bấm Download/PDF.*
