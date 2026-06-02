**Mục tiêu chính của bộ dữ liệu Retrieval:**
Mục tiêu quan trọng nhất của bộ dữ liệu này là **hỗ trợ các mô hình offline nhỏ**, vốn chỉ có khả năng xử lý câu đơn (single turn), có thể **thực hiện các tính năng cần đến bộ nhớ hội thoại**. Các mô hình này cần lấy lại (retrieve) đầy đủ ngữ cảnh để gọi công cụ (tool) một cách chính xác dựa trên những gì người dùng đã đề cập trước đó.

**Quy trình và nguyên tắc tạo bộ dữ liệu Retrieval:**

**1. Xác định cấu trúc dữ liệu hội thoại (Conversation Format):**
*   Dữ liệu đầu vào phải là một đoạn hội thoại nhiều lượt (conversation) chứ không phải câu đơn lẻ. 
*   Lịch sử hội thoại cần được xây dựng đa dạng về số lượng lượt tương tác, từ ít nhất 2 lượt (1 cặp người dùng - hệ thống) lên tới nhiều nhất là 4 lượt người dùng và 3 lượt hệ thống. 
*   Cần có sự phân chia tỉ lệ rõ ràng dựa trên số lượng lượt tương tác và theo các miền (domain) khác nhau.

**2. Thiết lập nguyên tắc xử lý ngữ cảnh (Rewriting Rules):**
*   **Tránh lặp lại hành động:** Nguyên tắc tối quan trọng là không bao giờ được lặp lại một hành động mà hệ thống đã thực hiện ở các lượt hội thoại trước. Mô hình phải được hướng dẫn chỉ "retrieve" những hành động chưa làm.
*   **Giữ nguyên khi đã rõ ràng:** Nếu câu nói cuối cùng của người dùng đã đầy đủ thông tin để thực hiện lệnh (ví dụ: "Bật đèn pha như tôi yêu cầu đi"), thì hệ thống phải giữ nguyên câu đó.
*   **Dạy mô hình phân biệt:** Cần xây dựng tỷ lệ trộn dữ liệu, khuyến nghị khoảng 2/3 dữ liệu bắt buộc dùng ngữ cảnh và 1/3 dữ liệu không cần dùng ngữ cảnh hội thoại, nhằm huấn luyện cho mô hình biết lúc nào thì cần gọi ngữ cảnh (retrieval) và lúc nào thì không.

**3. Kỹ thuật sinh và đánh giá dữ liệu (Generation & Evaluation):**
*   **Sử dụng Mô hình Ngôn ngữ (LLMs):** Có thể khởi tạo bằng một tập dữ liệu mầm (seed dataset) khoảng 200 câu, sau đó dùng các mô hình ngôn ngữ tối ưu (sử dụng model tốt nhất đã quantize có thể chạy trên Kaggle) để sinh thêm dữ liệu và dùng prompt để đánh giá chéo. Ở giai đoạn đầu, có thể dùng các LLM lớn như ChatGPT hay Gemini để hỗ trợ tạo dữ liệu chất lượng cao.
*   **Tinh chỉnh tham số (Temperature):** Quá trình sinh dữ liệu cần đặt nhiệt độ (`temperature`) cao để kết quả đa dạng, và khi đánh giá dữ liệu thì để `temperature` thấp để tăng tính chính xác. Hai tác vụ sinh và đánh giá bắt buộc phải có `system prompt` khác nhau.
*   **Tối ưu hóa chạy đa luồng:** Trong quá trình sinh dữ liệu, không được chạy nối tiếp (sequential) vì chỉ tận dụng được rất ít hiệu suất (chỉ khoảng 5% GPU utilization). Cần sửa code để chuyển sang chạy đa luồng, gọi nhiều tác vụ cùng một lúc nhằm đẩy hiệu suất tính toán lên tối đa 100% giúp tiết kiệm đáng kể thời gian sinh.

Để lựa chọn và đánh giá hiệu quả các mô hình ngôn ngữ kích thước 7B cho dự án, quy trình được thực hiện theo các chiến lược cụ thể về khảo sát benchmark, tối ưu phần cứng và thiết lập cấu hình chạy như sau:

**1. Cách lựa chọn mô hình:**
*   **Khảo sát Benchmark:** Cần tiến hành khảo sát (survey) các mô hình mã nguồn mở (open model) có kích thước có thể chạy trên kaggle để tìm ra mô hình nào đang đứng đầu các bảng xếp hạng (benchmark) hiện tại. Ưu tiên chọn những mô hình tốt nhất, nhanh nhất và phù hợp nhất với tài nguyên hạ tầng đang có.
*   **Tính toán tài nguyên phần cứng (VRAM):** Một mô hình 7B khi chạy ở định dạng BF16 sẽ cần ít nhất 14GB VRAM để tải trọng số, và khi suy luận (inference) sẽ tốn khoảng 21GB VRAM, mức này hoàn toàn có thể chạy tốt trên cấu hình GPU 24GB hoặc 30GB.
*   **Sử dụng kỹ thuật Lượng tử hóa (Quantization):** Để tối ưu tài nguyên hoặc sử dụng các mô hình lớn hơn, có thể dùng các mô hình đã được lượng tử hóa xuống mức INT4 hoặc sử dụng kỹ thuật AWQ. Lượng tử hóa INT4 từ các nhà phát triển lớn hầu như không làm giảm chất lượng mô hình, độ chính xác chỉ thay đổi chưa đến 1% so với bản gốc BF16. Tuy nhiên, **bắt buộc phải chọn những mô hình đã được ép kiểu lượng tử hóa ngay từ quá trình huấn luyện ban đầu** thay vì tự fine-tune một mô hình 16-bit rồi mới ép lượng tử hóa, vì cách làm này sẽ khiến mô hình kém thông minh đi rất nhiều.

**2. Quy trình đánh giá và ứng dụng mô hình:**
*   **Sử dụng chính mô hình đó để tự đánh giá:** Thay vì dùng mô hình lớn hơn để sinh dữ liệu và mô hình nhỏ hơn để đánh giá, dự án nên tận dụng sức mạnh của chính mô hình lớn đã serve xuất sắc nhất vừa chọn để thực hiện cả hai tác vụ này. 
*   **Tách biệt System Prompt:** Khi dùng chung một mô hình, bắt buộc phải sử dụng hai `system prompt` khác nhau. Đặc biệt, ở bước đánh giá dữ liệu, các yêu cầu và luật lệ trong prompt phải được viết cực kỳ tường minh (rõ ràng) để mô hình có thể chấm điểm một cách chuẩn xác nhất.
*   **Điều chỉnh Temperature:** Đây là thông số quyết định độ sáng tạo của mô hình. Khi **sinh dữ liệu**, cần đặt `temperature` cao để tạo ra các trường hợp đa dạng. Ngược lại, khi **đánh giá**, cần hạ `temperature` xuống thấp để kết quả mang tính xác định (deterministic) và chuẩn xác.
*   **Chạy đa luồng (Multi-threading) để tăng tốc:** Quá trình sinh và đánh giá dữ liệu bằng LLM thường mất rất nhiều thời gian. Nếu chỉ chạy nối tiếp (sequential), GPU chỉ hoạt động ở mức khoảng 5% công suất. Để tối ưu, cần phải sửa code sang chạy đa luồng (gọi song song nhiều luồng cùng lúc, ví dụ 32 tiến trình) để ép mức sử dụng GPU lên tối đa 100%, giúp tiết kiệm đáng kể thời gian sinh và đánh giá dữ liệu.