# Ghi chú: ảnh & số liệu cần cung cấp để hoàn thiện Chương 4–5

## MỚI (sau khi thêm bảng đa nhiệm, 02/07)

- **PhoBERT-large + XLM-R-large trên bảng đa nhiệm** — m nói "để sau"; khi có, t chèn
  thêm 2 cột vào Bảng 4.4 (tab:multitask).
- **Ô MRC của PhoBERT-base-v2** đang "--" (lần chạy n=5 chưa có) — chạy bổ sung hay để vậy?
- **Nguồn QNLI (vi) và STS (vi)**: bản dịch nào / lấy từ đâu? Cần ghi rõ nguồn trong
  §4.1.3 để hội đồng khỏi hỏi.
- **PPPL đo trên mẫu văn bản nào** (bao nhiêu câu, nguồn?) — hiện ghi "mẫu văn bản giữ lại".
- **Xác nhận 5M docs ≈ 5 tỷ token** (t đang ghi ~5 tỷ theo lời m).

Chương 4 và 5 đã viết xong và biên dịch được với 3 hình + 1 bảng số liệu hiện có
(`eval_loss_overlay.png`, `hardtask_MRC-ViQuAD_bar.png`, `mrc_learning_curve.png`,
bảng EM/F1 từ `mrc_summary.md`). Danh sách dưới đây là những thứ **còn thiếu hoặc
sẽ làm chương mạnh hơn** — có gì cứ thả vào `images/` rồi báo, t sẽ chèn và viết
phân tích.

## Bắt buộc (đang có placeholder/TODO trong LaTeX)

1. **Phần cứng thực tế** — GPU gì (A100 40GB?), có dùng loại khác giữa các phiên
   không. Hiện Chương 4 §4.1.1 đang ghi "Colab + A100 40GB" theo ghi chú trong
   notebook → cần m xác nhận hoặc sửa.
   `→ Chapter4/chapter4.tex, %TODO dòng ~28`

2. **Số dòng chính xác UIT-ViQuAD** sau lọc answerable + chia đôi validation:
   train / dev / test = ? / ? / ?. Hiện ghi "~18 nghìn câu hỏi huấn luyện".
   `→ Bảng 4.1 + §4.1.3`

3. **Tổng số token đã huấn luyện qua ở mốc 5M tài liệu** (nếu có trong log/metrics
   .jsonl). Hiện Bảng 4.1 chỉ ghi "5 triệu tài liệu".

## Nên có (nâng cấp từ "đọc ước lượng từ hình" → số chính xác)

4. **Số F1/EM chính xác của 8 phương án khởi tạo @100k** (dữ liệu vẽ bar chart
   `hardtask_MRC-ViQuAD_bar.png`, dạng bảng model → mean ± std). Có nó t sẽ thêm
   bảng số cạnh hình, phân tích sẽ chắc hơn thay vì "khoảng 66–68".

5. **Step-0 loss chính xác của từng phương án** (hiện text dùng ~7,2 / ~8,1 /
   ~10,1 / >16 / >21 đọc từ hình).

## Tùy chọn (mở rộng đánh giá — nếu muốn Chương 4 dày hơn)

6. **Ảnh/bảng kết quả VSFC và ViANLI @100k** (bakeoff các arms) — nếu cung cấp,
   t thêm mục "đánh giá trên tác vụ phân loại" + giải thích vì sao VSFC bão hòa
   (hiện chỉ nhắc 1 câu trong Thảo luận).

7. **Đường cong so sánh có/không căn chỉnh đóng băng** (freezealigned vs
   decpertoken là proxy rồi, nhưng nếu có hình riêng thì §4.4 mạnh hơn).

8. **Số lượng neo theo từng nguồn**: neo bề mặt đã xác minh / neo số / cặp dịch
   (tier 1, 2, 3) — t sẽ thêm bảng thống kê tập neo vào §4.2 (hoặc §3.3), rất đẹp
   khi bảo vệ.

9. **Thời gian huấn luyện thực tế** (giờ/phiên, tổng số phiên) — cho phần "chi phí
   tính toán" nếu muốn nhấn mạnh tính tiết kiệm.

## Việc t sẽ làm tiếp sau khi có số liệu

- Điền các TODO trong `Chapter4/chapter4.tex` (grep `TODO(user)` là thấy).
- Viết `Appendix/summary.tex` (Tóm tắt) — cần kết quả chốt ở trên trước.
- Bảng thống kê tập neo (mục 8) nếu m cung cấp.
