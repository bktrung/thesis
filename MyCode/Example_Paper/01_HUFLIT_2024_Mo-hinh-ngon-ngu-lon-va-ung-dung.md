---
tieu_de: "Mô hình ngôn ngữ lớn và ứng dụng"
tac_gia: "Đinh Minh Hòa, Phạm Ngọc Bảo, Huỳnh Vũ Lê, Lê Huỳnh Nghiêm, Nguyễn Thị Thúy An, Trần Khải Thiện"
don_vi: "Trường ĐH Ngoại ngữ – Tin học TP.HCM (HUFLIT)"
nguon: "Tạp chí Khoa học HUFLIT, Tập 8, Số 3 (2024)"
chu_de: "LLM, xử lý ngôn ngữ tự nhiên, trí tuệ nhân tạo"
link_xem: "https://hjs.huflit.edu.vn/index.php/hjs/article/view/206"
link_pdf: "https://hjs.huflit.edu.vn/index.php/hjs/article/download/206/121/967"
ghi_chu: "Trích xuất từ PDF bằng pdftotext ngày 09/07/2026 — dùng để học VĂN PHONG, có thể có lỗi bố cục cột/hình/bảng."
---

# [Bài mẫu 1] Mô hình ngôn ngữ lớn và ứng dụng (HUFLIT, 2024)

> Toàn văn trích xuất tự động bên dưới. Đọc để học cách viết tổng quan LLM bằng tiếng Việt.

---

HUFLIT Journal of Science

RESEARCH ARTICLE

MÔ HÌNH NGÔN NGỮ LỚN VÀ ỨNG DỤNG
Đinh Minh Hòa, Phạm Ngọc Bảo, Huỳnh Vũ Lê, Lê Huỳnh Nghiêm, Nguyễn Thị Thúy A,
Trần Khải Thiện*
Ho Chi Minh City University of Foreign Languages –Information Technology (HUFLIT)
thientk@huflit.edu.vn
TÓM TẮT— Bài báo này giới thiệu về mô hình ngôn ngữ lớn (Large Language Model - LLM) và khả năng ứng dụng LLM vào
trong hoạt động của các tổ chức doanh nghiệp. Chúng tôi đi từ việc giới thiệu kiến trúc của LLM và các công cụ, phần mềm
thông minh được xây dựng dựa trên LLM đến các thách thức và mối nguy cơ của LLM. Để minh họa cho sức mạnh của LLM,
chúng tôi cũng trình bày một case study cho việc áp dụng LLM vào bài toán phân lớp cảm xúc - một bài toán được nhiều sự
quan tâm của giới nghiên cứu cũng như doanh nghiệp trong hơn một thập kỷ nay.
Từ khóa— Mô hình ngôn ngữ lớn, LLM, xử lý ngôn ngữ tự nhiên, trí tuệ nhân tạo.

I. GIỚI THIỆU
Một viễn cảnh nơi mà máy tính có thể trò chuyện như con người, hiểu được ý nghĩa sâu xa trong lời nói và sáng
tạo ra những tác phẩm văn học. Điều này đang dần trở thành hiện thực với sự ra đời của các mô hình ngôn ngữ
lớn (Large Language Model - LLM) - một loại trí tuệ nhân tạo (Artificial Intelligence - AI) tiên tiến có khả năng xử
lý và tạo sinh ngôn ngữ tự nhiên. LLM là bước tiến đột phá trong lĩnh vực AI, mang đến nhiều ứng dụng thực tế
trong đời sống. Có thể kể đến đầu tiên là chatbot thông minh, cỗ máy có thể tạo ra những cuộc trò chuyện mượt
mà và ý nghĩa với con người [1]. Sức mạnh về ngôn ngữ của các LLM không chỉ dừng lại ở việc trò chuyện mà còn
ở khả năng dịch tự động [2], một thành tựu vĩ đại, phá vỡ các rào cản ngôn ngữ, tạo điều kiện cho giao tiếp hiệu
quả và hợp tác toàn cầu. Với khả năng xử lý ngôn ngữ tự nhiên (Natural Language Processing - NLP), LLM dễ
dàng dịch nội dung từ một ngôn ngữ gốc sang một ngôn ngữ khác, tạo sự thấu hiểu và tương tác giữa những
người dùng các ngôn ngữ khác nhau. Thêm nữa, khả năng tóm tắt văn bản của LLM [3] giúp chúng trở thành
những công cụ vô giá để rút ngắn, tóm tắt các bài viết, văn bản với khả năng vượt trội hơn cả các chuyên gia
trong cùng lĩnh vực. Nhưng có lẽ khía cạnh tuyệt vời nhất của LLM nằm ở khả năng khơi nguồn sáng tạo. Với sự
hỗ trợ từ LLM, người dùng có thể khám phá thế giới viết lách, sáng tác thơ ca, âm nhạc, viết kịch bản và tạo nội
dung cuốn hút qua nhiều thể loại [4].
Trong bài báo này, chúng tôi giới thiệu về LLM và khả năng ứng dụng của LLM, cũng như thảo luận về những mối
nguy cơ và thách thức của LLM.
Các đóng góp chính của bài báo bao gồm:
 Cung cấp một trình bày tổng quan cũng như kiến trúc của các LLM.
 Giới thiệu về các ứng dụng và thành quả tiên tiến của LLM.
 Nêu lên các thách thức và mối nguy cơ trong việc áp dụng LLM.
Phần còn lại của bài báo được trình bày như sau: Mục II giới thiệu tổng quan về LLM. Ở mục III, các giải pháp ứng
dụng LLM và các thách thức và mối nguy cơ của LLM được trình bày. Mục IV giới thiệu một case study ứng dụng
LLM vào bài toán phân lớp cảm xúc. Cuối cùng là phần kết luận được giới thiệu ở mục V.

II. TỔNG QUAN VỀ MÔ HÌNH NGÔN NGỮ LỚN
Để đạt được hiệu suất xử lý tối đa, LLM áp dụng một chuỗi các quy trình quan trọng và phức tạp biểu diễn bởi
hình 1, trong đó gồm:

18

MÔ HÌNH NGÔN NGỮ LỚN VÀ ỨNG DỤNG

Hình 1. Các bước xử lý của LLM

A. TEXT PREPROCESSING
Text preprocessing là giai đoạn tiền xử lý dữ liệu văn bản, bao gồm nhiều phương pháp và bước xử lý khác nhau,
tùy thuộc vào bài toán cụ thể. Giai đoạn này gồm ba bước phổ biến và gần như không thể thiếu là tokenization,
lemmatization, và stopword removal. Trong đó, bước tokenization thực hiện việc chia nhỏ văn bản thành các
đơn vị nhỏ hơn, có thể là từ, cụm từ. Bước lemmatization giúp chuyển đổi các từ về dạng cơ sở (danh từ số ít,
động từ ở thể cơ bản...). Bước stopword removal sẽ loại bỏ các không mang ý nghĩa (and, the, a…) khỏi văn bản
để giảm kích thước từ vựng.
B. TEXT REPRESENTATION
Text representation hay biểu diễn văn bản là bước chuyển đổi dữ liệu văn bản thành dạng có thể xử lý và hiểu
được bởi máy tính. Phương pháp phổ biến dùng cho LLM là nhúng từ (word embedding), một kỹ thuật biểu diễn
từ vào trong không gian có số chiều thấp hơn. Những từ có ý nghĩa tương tự thường có biểu diễn gần giống nhau
trong không gian vector.
C. PRE-TRAINING
Pre-training là việc luyện mô hình trên một lượng lớn dữ liệu, giúp mô hình hiểu và biểu diễn ngôn ngữ tự nhiên
bằng cách học các mẫu ngôn ngữ và sự phụ thuộc từ xa trong dữ liệu văn bản. Pre-training thường được xây
dựng dựa trên kiến trúc Transformer được biểu diễn ở Hình 2.

Hình 2. Kiến trúc Transformer [5]

Đinh Minh Hòa, Phạm Ngọc Bảo, Huỳnh Vũ Lê, Lê Huỳnh Nghiêm, Nguyễn Thị Thúy A, Trần Khải Thiện*

19



Word Embeddings: Word Embeddings hay nhúng từ là quá trình mã hóa thông tin về từ và vị trí của văn
bản trong không gian vector.
 Positional Encoding: Để giúp mô hình nhận biết vị trí của từ, Transformer sử dụng lớp positional
encoding giúp bổ sung thông tin về vị trí tương đối của các từ trong câu vào word embeddings.
 Encoder: Encoder là một lớp trong Transformer đảm nhận việc mã hóa thông tin đầu vào thành biểu
diễn có ý nghĩa. Bao gồm các lớp con: i) Self-Attention: Self-Attention giúp xác định mức độ quan trọng
của mỗi từ đối với các từ khác trong câu. Cụ thể, Self-Attention sẽ tính toán attention cho các cặp từ và
tạo ra một biểu diễn tổng quát cho mỗi từ. ii) Feed-Forward Neural Networks: Lớp này sẽ biến đổi các
biểu diễn từ dựa trên thông tin về sự tương quan từ lớp self-attention. Encoder thường bao gồm các lớp
linear và hàm kích hoạt như ReLU.
 Decoder: Lớp Decoder nhận đầu vào là biểu diễn từ qua Encoder và sau đó tạo ra đầu ra. Gồm các lớp
con: i) Masked Self-Attention: Trong lớp này tin chỉ được truyền từ các từ trước đó và không được
truyền từ các từ sau. Giúp mô hình sáng tạo trong quá trình dự đoán, không biết trước các từ tương lai.
ii) Encoder-Decoder Attention: Một lớp sử dụng encoder-decoder attention, giúp mô hình tập trung vào
các phần quan trọng của ngữ cảnh từ Encoder để tạo ra đầu ra cuối cùng.
 Multi-Head Attention: Self-attention, encoder-decoder attention đều sử dụng multi-head attention. Điều
này cho phép mô hình học các mối quan hệ khác nhau bằng cách thực hiện đồng thời với nhiều bộ trọng
số attention khác nhau. Sau đó kết hợp các kết quả lại để tạo ra đầu ra cuối cùng.
 Layer Normalization và Residual Connections: Mỗi lớp con trong Transformer đều được đặt trong một
kiến trúc layer normalization và residual connections. Lớp Normalization chuẩn hóa các biểu diễn từ
sau mỗi lớp con. Lớp Residual connections sẽ cho phép thông tin truyền qua các lớp con mà không bị
mất đi.
 Output Layer: Output Layer hay còn gọi là lớp đầu ra có nhiệm vụ đưa đầu ra qua một lớp tuyến tính và
chuyển qua hàm softmax để tạo ra phân bố xác suất trên từ vựng đầu ra. Cuối cùng, từ có xác suất cao
nhất sẽ được chọn làm đầu ra dự đoán.
D. FINE-TUNING
Fine-tuning là quá trình điều chỉnh mô hình đã được huấn luyện trước trên một tác vụ cụ thể hoặc một tập dữ
liệu nhất định. Việc thực hiện Fine-tuning cho phép mô hình học cách giải quyết một tác vụ cụ thể, các kỹ thuật
của Fine-tuning bao gồm:


Zero-shot learning là Fine-tuning trên một tác vụ chưa từng được huấn luyện. Thay vì phải huấn luyện
mô hình trên tác vụ mới, ta chỉ cần cung cấp một mô tả về tác vụ mới.
 Few-shot learning là Fine-tuning trên một tác vụ mới với một số lượng nhỏ dữ liệu huấn luyện. Ta chỉ
cần cung cấp một vài ví dụ huấn luyện cho tác vụ mới.
 Multi-shot learning là Fine-tuning trên một tác vụ mới với một số lượng lớn dữ liệu huấn luyện. Ở đây
chúng ta sẽ cung cấp một tập dữ liệu đầy đủ trên tác vụ mới.
E. ADVANCED FINE-TUNING
Trong quá trình huấn luyện LLM, dữ liệu huấn luyện có thể chứa nhiễu, sai sót, làm giảm độ chính xác. Để khắc
phục, phương pháp "Reinforcement Learning through Human Feedback" (RLHF) [6] có thể được sử dụng giúp
giải quyết các vấn đề bằng cách sử dụng phản hồi từ con người. Các lợi ích chính của phương pháp này gồm:





Tận dụng kiến thức của con người: Phương pháp này sẽ dùng kiến thức của con người để đánh giá chất
lượng và đưa ra phản hồi chính xác về văn bản được tạo ra bởi mô hình.
Cải thiện khả năng tương tác ngôn ngữ: RLHF sẽ giúp mô hình hiểu và sử dụng ngữ cảnh, tuân thủ các
quy tắc và phong cách ngôn ngữ được xác định bởi con người.
Điều chỉnh mô hình theo yêu cầu tác vụ: Phương pháp này có thể tùy chỉnh mô hình để đáp ứng các tiêu
chí chất lượng và định hình đầu ra theo mong muốn.
Đảm bảo tính nhân quyền và đạo đức: Bằng RLHF, sự can thiệp và kiểm soát từ con người sẽ giúp ngăn
chặn các đầu ra không phù hợp và thông tin sai lệch.

III. CÁC ỨNG DỤNG CỦA MÔ HÌNH NGÔN NGỮ LỚN
A. CÁC ỨNG DỤNG CỦA LLM
Các tổ chức doanh nghiệp đang nỗ lực tích hợp LLM vào hoạt động hàng ngày nhằm mang đến nhiều lợi ích đáng
kể như tự động hóa, cải thiện hiệu quả và khả năng sáng tạo. Các "copilot" dựa trên LLM đã thay đổi cách thức
làm việc của con người, ví dụ như "Microsoft 365 Copilot" [7] hỗ trợ người dùng trong các ứng dụng như MS.
Word, PowerPoint và Excel, giúp tổ chức hộp thư đến, tóm tắt cuộc họp và xử lý các tác vụ lặp đi lặp lại. Tương
tự, "Github Copilot" [8] cung cấp gợi ý mã lệnh lập trình cho các nhà phân tích dữ liệu, khoa học dữ liệu và kỹ sư
phần mềm. Doanh nghiệp cũng có thể tạo ra "copilot" tùy chỉnh bằng LLM nguồn mở. LLM cũng đã được áp dụng

20

MÔ HÌNH NGÔN NGỮ LỚN VÀ ỨNG DỤNG

trong lĩnh vực giáo dục [9], mang đến cho người học trải nghiệm tương tác với "giáo viên" AI có khả năng điều
chỉnh phương pháp giảng dạy dựa trên sự hiểu biết về học sinh. Ví dụ, Khan Academy đã tích hợp LLM vào sản
phẩm của mình với tên gọi Khanmigo, cho phép học sinh luyện tập toán, cải thiện từ vựng hoặc ôn luyện cho các
kỳ thi. Khanmigo cung cấp cả những trải nghiệm tương tác thú vị như trò chuyện với các nhân vật hư cấu hoặc
lịch sử. Trong lĩnh vực tài chính, LLM cũng được sử dụng để phân tích dữ liệu phi cấu trúc như báo cáo thường
niên, bài báo và mạng xã hội, giúp hiểu rõ hơn về thị trường, quản lý đầu tư và tìm kiếm cơ hội mới. Bloomberg
đã giới thiệu BloombergGPT™, một LLM được thiết kế đặc biệt cho lĩnh vực tài chính, hỗ trợ các tác vụ phân tích
cảm xúc, nhận dạng đối tượng và phân loại tin tức. LLM đã trở thành một công cụ quan trọng và đa năng, đóng
vai trò quan trọng trong nhiều lĩnh vực và mang lại lợi ích sáng tạo cho các doanh nghiệp, giáo dục và tài chính.
LLM đóng vai trò ngày càng quan trọng trong cải thiện hiệu suất và tăng cường khả năng sáng tạo của tổ chức
doanh nghiệp bằng cách tự động hóa các tác vụ lặp đi lặp lại, phân tích dữ liệu nhanh chóng và chính xác, cung
cấp dự đoán và đề xuất hữu ích. Một số ví dụ cụ thể:


Tự động hóa dịch vụ khách hàng: LLM có thể hỗ trợ giải quyết các tác vụ như trả lời câu hỏi thường gặp,
giải quyết các vấn đề cơ bản, xử lý yêu cầu bồi hoàn và hỗ trợ khách hàng 24/7. Cụ thể, chatbot được hỗ
trợ bởi LLM có thể giải đáp các thắc mắc về sản phẩm hoặc dịch vụ của doanh nghiệp, giúp giảm tải công
việc cho nhân viên và cải thiện trải nghiệm khách hàng.
 Tăng cường năng suất của nhân viên: LLM có thể hỗ trợ việc viết báo cáo, tóm tắt tài liệu, dịch ngôn ngữ,
tạo nội dung sáng tạo, sắp xếp lịch hẹn, quản lý email và theo dõi dự án. Cụ thể, LLM có thể giúp nhân
viên viết báo cáo nhanh chóng và hiệu quả hơn bằng cách tự động thu thập dữ liệu, phân tích thông tin
và tạo bản thảo.
 Cải thiện quy trình ra quyết định: LLM có khả năng phân tích dữ liệu thị trường, dự đoán xu hướng, xác
định rủi ro và đưa ra các đề xuất kinh doanh sáng suốt. Cụ thể, LLM có thể giúp doanh nghiệp dự đoán
nhu cầu của khách hàng, phát triển sản phẩm mới và tối ưu hóa chiến lược marketing.
B. THÁCH THỨC – NGUY CƠ TỪ LLM
Mặc dù LLM mang lại nhiều lợi ích, nhưng cũng có những hạn chế nhất định. Các tổ chức doanh nghiệp cần cân
nhắc kỹ lưỡng trước khi sử dụng LLM cho các tác vụ quan trọng. Có thể liệt kê như sau:


Độ chính xác: LLM có thể mắc sai sót, đặc biệt khi xử lý dữ liệu phức tạp, nhạy cảm hoặc chưa được huấn
luyện đầy đủ. Doanh nghiệp cần kiểm tra kỹ lưỡng kết quả trước khi sử dụng và có kế hoạch dự phòng
cho các trường hợp sai sót.
 Tính thiên vị: LLM có thể học hỏi và phản ánh những thành kiến có trong dữ liệu mà chúng được huấn
luyện. Tổ chức doanh nghiệp cần lựa chọn dữ liệu cẩn thận, theo dõi các vấn đề tiềm ẩn về tính thiên vị
và có biện pháp để giảm thiểu ảnh hưởng của chúng.
 Khả năng giải thích: LLM có thể đưa ra kết quả mà không có lời giải thích rõ ràng, gây hiểu nhầm hoặc
khó hiểu cho người dùng. Điều này đặc biệt quan trọng đối với các quyết định quan trọng và nhạy cảm.
Tổ chức doanh nghiệp cần cân nhắc việc sử dụng LLM trong các tình huống mà khả năng giải thích là
quan trọng.
Ngoài ra, một số nguy cơ và thách thức trong việc sử dụng LLM mà tổ chức doanh nghiệp cần cẩn trọng và có
trách nhiệm như sau:




Bảo mật thông tin: LLM có thể yêu cầu truy cập vào dữ liệu nhạy cảm của tổ chức doanh nghiệp để hoạt
động hiệu quả. Điều này đặt ra rủi ro về bảo mật thông tin. Tổ chức doanh nghiệp cần áp dụng các biện
pháp bảo mật phù hợp để đảm bảo rằng dữ liệu không bị rò rỉ hoặc truy cập trái phép.
Trách nhiệm đạo đức và pháp lý: LLM có thể sản xuất nội dung không đúng, gây ra phân biệt đối xử hoặc
vi phạm quy định pháp luật. Tổ chức doanh nghiệp phải đảm bảo rằng việc sử dụng LLM tuân thủ các
quy tắc đạo đức và pháp lý, và chịu trách nhiệm về nội dung mà chúng phát sinh.
Phụ thuộc công nghệ: Tổ chức doanh nghiệp cần nhớ rằng LLM là công nghệ mới có thể phụ thuộc vào
các công nghệ hỗ trợ. Nếu có sự cố với hệ thống hoặc không có quyền truy cập vào dữ liệu, hoạt động
của LLM có thể bị gián đoạn.

IV. CASE STUDY: PHÂN LỚP CẢM XÚC TWITTER
A. BÀI TOÁN
Phân loại cảm xúc trên Twitter là việc nhóm các tweet thành tích cực, tiêu cực hoặc trung lập, giúp hiểu và phân
tích tâm trạng của người dùng mạng xã hội.
B. CÁC PHƯƠNG PHÁP PHÂN LỚP CẢM XÚC VÀ PHƯƠNG PHÁP SỬ DỤNG LLM

Đinh Minh Hòa, Phạm Ngọc Bảo, Huỳnh Vũ Lê, Lê Huỳnh Nghiêm, Nguyễn Thị Thúy A, Trần Khải Thiện*









21

Phương pháp dựa trên ngữ nghĩa để phân tích cảm xúc liên quan đến việc phân tích ý nghĩa của văn bản
để xác định tình cảm hoặc tông điệu cảm xúc trong văn bản. Phương pháp này thường dùng một từ điển
chứa các từ cảm xúc, cùng với các phân cực và độ đo tương ứng của chúng nhằm tính toán mức độ cảm
xúc của toàn văn bản [10], [11].
Phương pháp học máy [12]: Học máy sử dụng biểu diễn đặc trưng văn bản cùng với các thuật toán như
Naïve Bayes, Support Vector Machines, Logistic Regression để làm bộ phân lớp. Các bộ phân lớp này có
khả năng học các quy tắc hoặc đặc trưng quan trọng từ dữ liệu huấn luyện và tạo ra một mô hình dự
đoán và được sử dụng để phân lớp tự động.
Phương pháp học sâu [13]: Học sâu là một phân lĩnh của học máy đã thu hút sự quan tâm lớn về những
khả năng đáng kinh ngạc của nó trong việc giải quyết các nhiệm vụ phức tạp. Các mô hình học sâu được
xây dựng dựa trên các mạng nơ-ron nhân tạo, lấy cảm hứng từ kiến trúc của não người. Một trong
những lĩnh vực mà học sâu đã có sự tác động đáng kể là phân tích cảm xúc.
Phương pháp dựa trên LLM: Là mô hình ngôn ngữ có quy mô lớn, được huấn luyện trên một lượng lớn
dữ liệu văn bản, như GPT-3, GPT-4, hoặc BERT. Những mô hình ngôn ngữ lớn này có khả năng hiểu và
sinh ra văn bản tự nhiên với độ chính xác cao. Các nhà nghiên cứu có thể sử dụng các mô hình LLM như
nền tảng, sau đó tinh chỉnh chúng bằng cách huấn luyện thêm trên tập dữ liệu chuyên biệt về cảm xúc.
Quá trình này cho phép tận dụng được sức mạnh của LLMs trong việc hiểu ngôn ngữ, đồng thời cũng
điều chỉnh chúng để đạt được hiệu suất cao hơn trong bài toán phân loại cảm xúc cụ thể. Hình 3 biểu
diễn việc sử dụng LLM, cụ thể là API ChatGPT-3.5, cho phân loại các đánh giá của người dùng bằng cách
cung cấp các hướng dẫn giống như ngôn ngữ tự nhiên.

Hình 3. Sử dụng API ChatGPT-3.5, chúng ta có thể nhanh chóng phân loại các đánh giá của người dùng bằng cách cung cấp
các hướng dẫn giống như ngôn ngữ tự nhiên.

Ở bài toán này chúng tôi thực hiện phân tích cảm xúc 2 lớp (positive/negative) sử dụng LLM và so sánh với các
phương pháp dựa trên học máy, sử dụng các mô hình phân lớp Naïve Bayes, Support Vector Machines, Logistic
Regression và Decision Tree, ở đây sử dụng mô hình GPT-3.5-Turbo. Sau đó so sánh các kết quả thực hiện. Về tập
dữ liệu, chúng tôi đã sử dụng tập dữ liệu Twitter với 3 thuộc tính là ID, Tweet, Sentiment. Tập dữ liệu này được
thu thập từ Kaggle*.


*

Với phương pháp học máy, chúng tôi thực hiện huấn luyện 6000 mẫu Tweet. Chia tập dữ liệu Tweet
thành tập train và tập test. Dùng 5000 mẫu để huấn luyện, 1000 mẫu để kiểm tra độ chính xác. Để đánh
giá hiệu quả của các mô hình, chúng tôi đã sử dụng các số đo phổ biến như Accuracy, Precision, Recall và
F1-Score.

https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis

22

MÔ HÌNH NGÔN NGỮ LỚN VÀ ỨNG DỤNG



Với phương pháp sử dụng LLM, do mô hình ngôn ngữ lớn đã được huấn luyện từ trước, chúng tôi chỉ
dùng 10 mẫu đã phân lớp cảm xúc để làm ví dụ khi mô tả bài toán cho mô hình. Sau đó dùng đúng 1000
mẫu test đã chia và sử dụng cho các mô hình học máy để kiểm tra hiệu suất của LLM nhằm đảm bảo tính
công bằng khi so sánh. Kết quả thực nghiệm được cho bởi Bảng 1.
Bả ng 1. So sánh kết quả phân tích cảm xúc của phương pháp học máy với phương pháp LLM.

Phương pháp

Accuracy

Precision

Recall

F1-Score

SVM

77.4

81.6

77.4

78.4

Naive Bayes

76.5

81.3

76.5

77.6

Logistic Regression

78.2

81.6

78.2

79.0

Decision Tree

72.8

74.4

72.8

72.8

LLM (GPT-3.5-Turbo)

78.7

87.8

78.7

80.1

Bảng 1 cho thấy LLM cho ra kết quả cao hơn so với các phương pháp học máy thông thường. Các mô hình LLM có
khả năng phân tích và hiểu ngữ cảnh của văn bản tweet một cách sâu sắc hơn, giúp cải thiện độ chính xác trong
việc phân loại cảm xúc. Khi sử dụng một mô hình học máy, chúng ta cần thu thập một tập dữ liệu huấn luyện lớn
từ các bình luận, nhận xét của người dùng và gán nhãn thủ công cho các đánh giá này, chẳng hạn như "positive"
hoặc "negative". Tập dữ liệu này sau đó có thể được sử dụng để huấn luyện mô hình phân loại văn bản. Tuy
nhiên, nếu sử dụng các gợi ý và LLMs, chúng ta chỉ cần chuẩn bị vài chục văn bản nhận xét của người dùng và sử
dụng các hướng dẫn bằng ngôn ngữ tự nhiên để nhanh chóng có được kết quả phân lớp. Việc xây dựng một mô
hình phân loại văn bản cảm xúc với hiệu quả như vậy gần như không tốn chi phí.

V. KẾT LUẬN
Các mô hình ngôn ngữ lớn - LLMs, một công nghệ đột phá đang dẫn đầu sự thay đổi trong kỷ nguyên AI, mà các
tổ chức doanh nghiệp phải tận dụng nhằm đem lại nhiều lợi ích hứa hẹn. Để thành công với LLM, tổ chức doanh
nghiệp cần đào tạo nhân viên về cách sử dụng các mô hình này một cách hiệu quả và có trách nhiệm từ việc
trang bị kiến thức về kỹ thuật AI, yếu tố đạo đức và quy trình sử dụng LLM. Đội ngũ nhân viên phải trở thành
những người chủ động và tự tin trong việc áp dụng LLM vào công việc hàng ngày để luôn sẵn sàng trong kỷ
nguyên AI.
Bài báo đã giới thiệu về LLM và những ứng dụng của LLM cũng như trình bày một case study cụ thể về việc áp
dụng LLM cho phân lớp cảm xúc các bình luận mạng xã hội. Nghiên cứu tương lai của chúng tôi là việc xây dựng
các công cụ, phần mềm thông minh nhờ vào sự hỗ trợ của LLM.

VI. TÀI LIỆU THAM KHẢO
[1]
[2]
[3]
[4]
[5]
[6]

F. Khennouche, Y. Elmir, Y. Himeur, N. Djebari, and A. Amira (2024), Revolutionizing generative pretraineds: Insights and challenges in deploying ChatGPT and generative chatbots for FAQs, Expert Syst Appl
(doi: 10.1016/J.ESWA.2024.123224), vol. 246, pp. 123224-123232.
Z. He et al.(2024), Exploring Human-Like Translation Strategy with Large Language Models. Trans Assoc
Comput Linguist (doi: 10.1162/TACL_A_00642/119992/EXPLORING-HUMAN-LIKE-TRANSLATIONSTRATEGY-WITH), vol. 12, pp. 229–246 vol 30, pp. 1134-1142.
D. Van Veen et al.(2024), Adapted large language models can outperform medical experts in clinical text
summarization, Nat Med (doi: 10.1038/S41591-024-02855-5), vol. 30, pp. 1134-1142.
N. Imasato, K. Miyazawa, C. Duncan, and T. Nagai (2023), Using a Language Model to Generate Music in Its
Symbolic
Domain
While
Controlling
Its
Perceived
Emotion,
IEEE
Access
(doi:
10.1109/ACCESS.2023.3280603), vol. 11, pp. 52412-52428.
A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin (2017),
Attention is all you need, In Proceedings of the 31st International Conference on Neural Information
Processing Systems (NIPS'17), Curran Associates Inc., pp. 6000-6010.
Y. Du et al.(2023), Guiding Pretraining in Reinforcement Learning with Large Language Models, in
Proceedings of the 40th International Conference on Machine Learning, PMLR, pp. 8657-8677,
https://proceedings.mlr.press/v202/du23f.html, Acceesed date: 07/4/2024.

Đinh Minh Hòa, Phạm Ngọc Bảo, Huỳnh Vũ Lê, Lê Huỳnh Nghiêm, Nguyễn Thị Thúy A, Trần Khải Thiện*

[7]
[8]
[9]
[10]
[11]
[12]
[13]

23

N. Edwards (2023), Microsoft’s unified Copilot is coming to Windows, Edge, and everywhere else, The
Verge, https://www.theverge.com/2023/9/21/23883798/microsoft-copilot-unified-windows-11-appslaunch-date, Acceesed date: 07/4/2024.
M. Wermelinger (2023), Using GitHub Copilot to Solve Simple Programming Problems, in SIGCSE 2023 Proceedings of the 54th ACM Technical Symposium on Computer Science Education, Association for
Computing Machinery Inc. (doi: 10.1145/3545945.3569830), pp. 172-178.
S. Milano, J. A. McGrane, and S. Leonelli (2023), Large language models challenge the future of higher
education, Nature Machine Intelligence (doi: 10.1038/s42256-023-00644-2), vol. 5, no. 4, pp. 333–334.
T. K. Tran and T. T. Phan (2016), Computing Sentiment Scores of Adjective Phrases for Vietnamese,
Springer, Cham (doi: 10.1007/978-3-319-49397-8_25), pp. 288–296.
T. T. Tran, T.K. & Phan(2016), Computing Sentiment Scores of Verb Phrases for Vietnamese, Proceedings of
the 28th Conference on Computational Linguistics and Speech Processing (ROCLING 2016), pp. 204–213,
https://aclanthology.info/papers/O16-1020/o16-1020, Acceesed date: 07/4/2024.
T. K. Tran and T. T. Phan (2016), Multi-Class Opinion Classification for Vietnamese Hotel Reviews,
International
Journal
of
Intelligent
Technologies
and
Applied
Statistics
(doi:
10.6148/IJITAS.2016.0901.02), vol. 9, no. 1, pp. 7–18.
T. K. Tran and T. T. Phan (2019), Deep Learning Application to Ensemble Learning-The Simple, but
Effective, Approach to Sentiment Classifying, Applied Sciences (doi: 10.3390/app9132760), vol. 9, no. 13,
pp. 6-18.

LARGE LANGUAGE MODELS AND APPLICATIONS
Dinh Minh Hoa, Pham Ngoc Bao, Huynh Vu Le, Le Huynh Nghiem, Nguyen Thi Thuy A, Tran Khai Thien*
Ho Chi Minh City University of Foreign Languages –Information Technology (HUFLIT)
thientk@huflit.edu.vn
ABSTRACT— This article introduces the concept of Large Language Models (LLMs) and the potential applications of LLMs in
the operations of business organizations. We begin with an overview of the architecture of LLMs and the intelligent software
tools built upon LLMs, progressing to the challenges and risks associated with LLMs. To illustrate the power of LLMs, we also
present a case study of applying LLMs to sentiment classification - a problem that has garnered significant interest from both
researchers and businesses for over a decade.
Keywords — large language model, LLM, natural language processing, artificial intelligence.

TS. Trần Khải Thiện lãnh bằng tiến
sĩ ngành Khoa học máy tính tại
trường ĐH Bách Khoa, ĐHQG-HCM
năm 2022. Hiện ông là Giảng viên tại
khoa Công nghệ thông tin trường ĐH
Ngoại ngữ - Tin học Tp. Hồ Chí Minh
(HUFLIT) và là Trưởng nhóm nghiên
cứu IDPS của trường. TS. Thiện là
bình duyệt viên và là tác giả của trên
20 công bố trong các tạp chí SCIE và
các hội nghị quốc tế uy tín.

ThS. Đinh Minh Hòa lãnh bằng
thạc sĩ ngành Công nghệ Thông
tin tại trường ĐH Ngoại ngữ Tin học Tp. Hồ Chí Minh
(HUFLIT) vào năm 2022. Hiện
tại Thạc sĩ Minh Hòa đang là
giảng viên tại khoa Công nghệ
thông tin trường HUFLIT
Hướng nghiên cứu chính của
ông là Xử lý ngôn ngữ tự nhiên,
Trí tuệ nhân tạo.

Hướng nghiên cứu chính của ông là Xử lý ngôn ngữ tự nhiên,
Trí tuệ nhân tạo.
ThS. Nguyễn Thị Thúy A lãnh bằng
thạc sĩ ngành Khoa học máy tính tại
trường ĐH Khoa học tự nhiên, ĐHQGHCM vào năm 2018. Hiện tại Thạc sĩ
Thúy A đang là giảng viên tại khoa Công
nghệ thông tin, trường ĐH Ngoại ngữ Tin học Tp. Hồ Chí Minh (HUFLIT).
Hướng nghiên cứu chính: Xử lý ngôn
ngữ tự nhiên, Trí tuệ nhân tạo.

Huỳnh Vũ Lê hiện là học viên
ngành Công nghệ thông tin trường
ĐH Ngoại ngữ - Tin học TP. Hồ Chí
Minh (HUFLIT). Hiện tại là chuyên
viên phòng Tuyển sinh trường
HUFLIT.
Hướng nghiên cứu chính của ông
là Xử lý ngôn ngữ tự nhiên, Trí tuệ
nhân tạo.

24

MÔ HÌNH NGÔN NGỮ LỚN VÀ ỨNG DỤNG

Phạm Ngọc Bảo hiện là sinh viên
chuyên ngành Khoa học dữ liệu, ngành
Công Nghệ Thông Tin tại Trường ĐH
Ngoại ngữ – Tin học Tp. Hồ Chí Minh
(HUFLIT).

Lê Huỳnh Nghiêm hiện là sinh
viên chuyên ngành Khoa học dữ
liệu, ngành Công Nghệ Thông Tin
tại Trường ĐH Ngoại ngữ – Tin
học Tp. Hồ Chí Minh (HUFLIT).

Hướng nghiên cứu chính: Xử lý ngôn
ngữ tự nhiên, Trí tuệ nhân tạo.

Hướng nghiên cứu chính: Xử lý
ngôn ngữ tự nhiên, Trí tuệ nhân
tạo.

