# Chuẩn thuật ngữ ML/NLP tiếng Việt — đối chiếu với luận

> Nguồn chuẩn: **Bảng thuật ngữ dự án dịch "Dive into Deep Learning" (d2l.aivivn.com/glossary.html)** — bản dịch cộng đồng, dùng tham khảo ở nhiều đại học; đây là một trong các chuẩn được chấp nhận rộng cho thuật ngữ học sâu tiếng Việt. Bổ sung từ bản dịch chương BERT của d2l và các paper FAIR/tạp chí.
>
> Cập nhật: 10/07/2026.

## 1. Thuật ngữ KHỚP giữa luận và chuẩn d2l ✅
| English | d2l | Luận đang dùng |
|---|---|---|
| pre-training | **tiền huấn luyện** | tiền huấn luyện ✅ |
| fine-tuning | **tinh chỉnh** | tinh chỉnh ✅ |
| loss | **mất mát** | mất mát ✅ |
| learning rate | **tốc độ học** | tốc độ học ✅ |
| warmup | **khởi động** | khởi động ✅ |
| representation | **biểu diễn** | biểu diễn ✅ |
| corpus | **kho ngữ liệu** | kho ngữ liệu ✅ |
| activation function | **hàm kích hoạt** | hàm kích hoạt ✅ |
| sequence | **chuỗi** | chuỗi ✅ |
| token | **token** (giữ nguyên) | token ✅ (đã xác nhận) |
| transformer | **transformer** (giữ) | Transformer ✅ |
| gradient | **gradient** (giữ) | gradient ✅ |

## 2. ⚠️ CHỖ LUẬN ĐANG LỆCH CHUẨN — cần cân nhắc
| English | d2l (chuẩn) | Luận đang dùng | Ghi chú |
|---|---|---|---|
| **masked language model / mask** | **mặt nạ** ("mô hình ngôn ngữ có mặt nạ") | **đánh dấu** | 🔴 "đánh dấu" KHÔNG phải chuẩn; "mặt nạ" mới là chuẩn. Cân nhắc **hoàn nguyên "mặt nạ"** hoặc giữ English "MLM". |
| **prior** | **tiên nghiệm** | phân phối tần suất | 🟡 "tiên nghiệm" là chuẩn của "prior", nhưng "phân phối tần suất" mô tả đúng bản chất (frequency prior) và dễ hiểu hơn → chấp nhận được. |
| **downstream task** | **tác vụ xuôi dòng** | tác vụ ứng dụng | 🟡 thầy đã gạch "hạ nguồn"; d2l dùng "xuôi dòng". "tác vụ ứng dụng" rõ hơn cả hai → giữ, nhưng biết "xuôi dòng" là biến thể chuẩn. |
| **attention / self-attention** | **cơ chế tập trung / tự tập trung** | chú ý / tự chú ý | 🟡 cả "chú ý" và "tập trung" đều dùng rộng; "(self-)attention" nên kèm tiếng Anh lần đầu. |
| **bias** | **hệ số điều chỉnh** | độ chệch (bias) | 🟡 "độ chệch" phổ biến trong thống kê; đã kèm "(bias)" → an toàn. |
| **embedding** | **embedding** (giữ nguyên) / "embedding từ" | véc-tơ | 🟡 d2l GIỮ "embedding"; luận đổi "véc-tơ" theo yêu cầu — chấp nhận được nhưng biết chuẩn phổ biến là giữ "embedding". |
| **positional encoding** | biểu diễn vị trí | mã hóa vị trí | 🟢 cả hai đều ổn (encoding = mã hóa). |
| **feed-forward** | truyền xuôi | truyền thẳng | 🟢 "truyền thẳng" rất phổ biến. |
| **encoder / decoder** | bộ mã hóa / bộ giải mã | Encoder (giữ) / bộ giải mã | 🟢 giữ "Encoder" cho tên kiến trúc là ổn. |

## 3. Cách diễn đạt "được huấn luyện trên / tokens seen"
- d2l (chương BERT): **"được tiền huấn luyện trên [kho ngữ liệu]"** (vd *"BERT được tiền huấn luyện trên BookCorpus và Wikipedia"*).
- "tokens seen" (số token đã đi qua, kể cả lặp epoch) **chưa có bản dịch chuẩn** trong nguồn tiếng Việt → cần chọn cách rõ nghĩa (xem quyết định trong `QUAN_DIEM_SUA_BAI.md`).

## 4. Nguyên tắc rút ra
1. Thuật ngữ đã có trong bảng d2l → **theo d2l** trừ khi có lý do rõ ràng dễ hiểu hơn.
2. Thuật ngữ giữ tiếng Anh trong d2l (token, transformer, gradient, embedding, batch, perplexity) → **giữ tiếng Anh**, chú thích Việt lần đầu.
3. Chỗ luận cố tình khác chuẩn (véc-tơ, tác vụ ứng dụng) → chấp nhận vì **rõ hơn**, nhưng phải nhất quán tuyệt đối.
