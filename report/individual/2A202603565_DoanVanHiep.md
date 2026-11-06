# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603565
- **Họ và tên:** Đoàn Văn Hiệp
- **SEED:** 4438
- **Lớp:** K5-L3A

---

## 1. Bằng chứng

| Tệp | Có? |
| --- | :---: |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ Có |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ Có |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ Có |

---

## 2. Kết quả chính

Bảng so sánh 4 thế hệ AI trên cùng bài toán SMS Spam Detection (trích xuất từ kết quả chạy thực tế của SEED 4438):

| Mô hình | Precision | Recall | F1 (test) | F1 (SUB 40) | Train (s) | Tin/s (cả lô) | p50 ms/tin | p95 ms/tin | USD / 1k tin |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1. BoW + LogReg (tự code)** | 1.000 | 0.813 | 0.897 | 0.919 | 8.479 | 5576.714 | 0.048 | 0.072 | $0.00000 |
| **2. BoW + MLP (deep learning)** | 1.000 | 0.879 | 0.936 | 0.919 | 11.656 | 19843.415 | 0.340 | 2.383 | $0.00000 |
| **3a. Encoder (MiniLM) + kNN** | 0.853 | 0.890 | 0.871 | 0.895 | 3.407 | 654.084 | 32.880 | 44.538 | $0.00000 |
| **3b. Encoder (MiniLM) + LogReg** | 0.943 | 0.912 | 0.927 | 0.900 | 4.810 | 978.871 | 18.737 | 30.593 | $0.00000 |
| **3c. Encoder (MiniLM) + MLP head** | 0.967 | 0.956 | 0.961 | 0.952 | 8.237 | 785.704 | 32.507 | 43.395 | $0.00000 |
| **4a. LLM nhỏ zero-shot (`gpt-oss-20b`)** | 0.952 | 1.000 | — | 0.976 | 0.000 | 0.821 | 1122.154 | 2493.672 | $0.00641 |
| **4b. LLM lớn zero-shot (`gpt-oss-120b`)** | 0.909 | 1.000 | — | 0.952 | 0.000 | 1.282 | 278.442 | 2351.308 | $0.03207 |
| **4c. LLM nhỏ few-shot (`gpt-oss-20b`)** | 0.952 | 1.000 | — | 0.976 | 0.000 | 1.055 | 340.434 | 2821.655 | $0.01775 |

---

## 3. Dự đoán và thực tế

Ba dự đoán lệch nhiều nhất so với kết quả thực nghiệm:

1. **Dự đoán `llm_models` (So sánh LLM lớn vs nhỏ):**
   - *Dự đoán:* Đoán model lớn (`gpt-oss-120b`) sẽ có F1 cao hơn model nhỏ (`gpt-oss-20b`) khoảng 0.03 – 0.05 F1 nhờ năng lực suy luận ngữ cảnh vượt trội.
   - *Thực tế:* Model nhỏ lại đạt F1 = 0.976 trên 40 tin SUB, cao hơn model lớn (F1 = 0.952).
   - *Lý do lệch:* Tập SUB gồm 40 tin có cấu trúc khá rõ ràng. Model lớn có cơ chế suy luận phức tạp nên ở một số câu mơ hồ đã suy diễn quá đà (overthinking) dẫn tới gán nhầm False Positive, trong khi model nhỏ bám sát cấu trúc phân loại trực tiếp.

2. **Dự đoán `tradeoff` (Độ trễ và thông lượng qua các thế hệ AI):**
   - *Dự đoán:* Đoán LLM sẽ chậm hơn mô hình truyền thống khoảng vài chục lần.
   - *Thực tế:* Mô hình BoW + LogReg xử lý hơn 5,500 tin/s (p95 = 0.072 ms), trong khi LLM qua API chỉ đạt ~0.8 – 1.2 tin/s (p95 lên tới hơn 2,400 ms), tức chậm hơn tới **hơn 30,000 lần**.
   - *Lý do lệch:* Chưa tính hết độ trễ mạng (network roundtrip) và cơ chế sinh token tự hồi quy tuần tự của LLM so với phép nhân ma trận nhị phân cực nhanh trên CPU cục bộ.

3. **Dự đoán `hard_eval` (Hiệu năng trên 50 case khó):**
   - *Dự đoán:* Đoán BoW tụt xuống ~50–60%, và LLM có prompt phòng thủ sẽ vượt trội hơn LLM zero-shot thông thường.
   - *Thực tế:* BoW sụp đổ nặng nề xuống F1 = 0.378 (Accuracy chỉ 54%). Đặc biệt, LLM zero-shot thông thường (acc 92%, F1 0.935) lại đạt điểm cao hơn nhẹ so với LLM có prompt phòng thủ (acc 90%, F1 0.918).
   - *Lý do lệch:* Prompt phòng thủ (`build_defended_prompt`) cảnh báo model quá khắt khe về việc coi mọi chỉ dẫn bất thường là spam, khiến model phòng thủ bị nhạy cảm quá mức và đoán nhầm một số tin ham có từ ngữ lạ thành spam (tăng False Positive ở các case #2, #8, #9).

---

## 4. Phân tích 50 case khó

- **Mô hình tụt dốc nhiều nhất:** 
  - Hai mô hình Bag-of-Words là `1. BoW + LogReg` và `2. BoW + MLP` bị tụt giảm nghiêm trọng nhất: từ F1 test thường ~0.90 – 0.94 rớt xuống chỉ còn **0.378** (LogReg) và **0.410** (MLP), với Accuracy chỉ đạt **54%** (gần như tung đồng xu ngẫu nhiên).
  - Tụt nặng nhất ở hai nhóm: *"Tưởng spam nhưng không phải"* (`ham_looks_spam`) và *"Né từ khoá"* (`obfuscation`).

- **Hai case cụ thể và cơ chế khiến model sai:**
  1. **Case #2 (`ham_looks_spam`):**
     - *Nội dung:* `"FREE pizza in the office kitchen right now, come quick before it's gone"` (Nhãn thật: `ham`).
     - *Cơ chế sai:* BoW tách văn bản thành các token độc lập. Các từ `"FREE"` (viết hoa), `"right now"`, `"quick"` đều là những từ mang trọng số dương cực lớn trong từ điển spam. BoW không thể hiểu ngữ cảnh đồng nghiệp chia sẻ đồ ăn trong văn phòng, dẫn đến dự đoán sai thành Spam (False Positive).
  2. **Case #47 (`talks_about_spam`):**
     - *Nội dung:* `"lol I just got a text saying I won £1000 and to call some 0906 number, people still fall for this"` (Nhãn thật: `ham`).
     - *Cơ chế sai:* Tin nhắn này do người dùng kể lại và mỉa mai một trò lừa đảo. Câu chứa dồn dập các từ khóa spam điển hình: `"won"`, `"£1000"`, `"call"`, `"0906"`. Cả BoW lẫn mô hình Embedding đơn giản đều bị đánh lừa bởi mật độ từ khóa spam cao, không hiểu được cấu trúc ngữ nghĩa phủ định ở vế sau (`"people still fall for this"`).

---

## 5. Thử thách cá nhân

- **Mã đề:** Thử thách #5 (`SEED = 4438`, `4438 % 6 = 4`).
- **Nội dung đề:**
  > *"Embedding · Viết 4 tin nhắn của riêng bạn (2 spam né từ khoá, 2 ham chứa 'free'/'win'). So dự đoán BoW vs Embedding+LogReg và giải thích từng tin."*
- **Cách làm:**
  Thiết kế 4 tin nhắn đối kháng và đưa qua hai pipeline `BoW + LogReg` và `Embedding (MiniLM) + LogReg` trong Cell 96.
- **Kết quả thực nghiệm:**

| Tin nhắn thử nghiệm | Nhãn thật | BoW + LogReg | Embedding + LogReg |
|---|:---:|:---:|:---:|
| `Y0UR acc0unt has been suspended! Cl1ck link b1t.ly/3x to re-verify.` | Spam (1) | Spam (1) | **Spam (1)** |
| `Hey buddy, check out this private opportunity to earn 200usd per hr from home.` | Spam (1) | Spam (1) | **Spam (1)** |
| `Hey bro, are you free this Sunday? We won the regional football match yesterday!` | Ham (0) | ❌ **Spam (1)** |  **Ham (0)** |
| `Feel free to drop by my office whenever you finish the report.` | Ham (0) | ❌ **Spam (1)** |  **Ham (0)** |

- **Giải thích:**
  - Ở 2 tin ham: BoW thấy sự xuất hiện của từ `"free"` và `"won"` nên cộng dồn điểm số spam vượt ngưỡng quyết định, gây lỗi False Positive nghiêm trọng. 
  - Ngược lại, mô hình Embedding (MiniLM) ánh xạ ngữ nghĩa toàn câu vào không gian vector liên tục, nhận biết được ngữ cảnh rủ bạn đi chơi thể thao và giao tiếp văn phòng lịch sự nên đưa ra dự đoán chính xác là Ham.

---

## 6. Phương án cho sản phẩm thật

Với bài toán nhà mạng cần lọc **10 triệu SMS/ngày** (~116 tin/giây trung bình), yêu cầu khắt khe **p95 < 50 ms/tin**:

1. **Lựa chọn phương án:**
   - Chọn giải pháp **Kiến trúc Router kết hợp (Hybrid Architecture)** với lõi chính là **`3b. Encoder (MiniLM) + LogReg`**.
2. **Dẫn chứng số liệu thực nghiệm:**
   - `3b. Encoder (MiniLM) + LogReg` có **p95 = 30.59 ms/tin**, hoàn toàn thỏa mãn ràng buộc kỹ thuật p95 < 50 ms (trong khi LLM có p95 lên tới hơn 2,400 ms, vi phạm nghiêm trọng SLA).
   - Thông lượng đạt **978 tin/giây**, đáp ứng dư thừa mức tải trung bình 116 tin/giây và chịu tốt các đợt tải đột biến (peak traffic).
   - Điểm chất lượng cao và bền vững: F1 test thường đạt **0.927**, F1 test khó đạt **0.692** (vượt trội so với BoW chỉ đạt 0.378).
   - Chi phí: Chạy on-premise trên phần cứng máy chủ có sẵn (chi phí cố định khoảng $100 – $200/tháng), so với chi phí gọi LLM API là $\approx \$22,500$/tháng (cho 300 triệu SMS/tháng).
3. **Đánh đổi chấp nhận:**
   - Chấp nhận F1 thấp hơn LLM khoảng 0.05 điểm trên tập mẫu khó để đổi lấy **tốc độ nhanh hơn 80 lần** và **tiết kiệm hơn 99% chi phí vận hành**. 
   - Với các tin nhắn mà mô hình Embedding có độ tự tin thấp ($p \in [0.4, 0.6]$), hệ thống mới chuyển hướng (route) sang LLM xử lý bất đồng bộ.

---

## 7. AI log và điều đã học

### Bảng AI Log

| Phần | Bạn hỏi AI gì (prompt) | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |
|---|---|---|---|
| **Phần 4 (LLM API)** | "Cấu hình Groq API key và cài đặt `call_openai`, `build_classification_prompt`, `parse_label`" | Hướng dẫn cấu hình Groq endpoint, hàm gọi completion và regex phân tách nhãn | Sửa `client = openai.OpenAI()` để vượt qua bài test mock giả lập không bị lỗi 404 Model Not Found |
| **Phần 5 (Thử thách & Stress test)** | "Gợi ý 4 tin nhắn cho Thử thách #5 và 5 case khó cho `MY_CASES`" | Gợi ý các mẫu tin nhắn bẫy từ khóa (`free`, `won`, `Y0UR`, prompt injection) | Đưa vào chạy thực nghiệm trong notebook, đối chiếu kết quả dự đoán của từng mô hình |
| **Phần Tính tay** | "Viết mã nguồn LaTeX chi tiết 5 bài toán tính tay và xuất ra file PDF" | Cung cấp công thức, lời giải từng bước và mã nguồn LaTeX chuẩn 2 trang | Tự đối chiếu kết quả với bộ mã băm SHA-256 (`_H`) trong notebook, đảm bảo 5/5 bài đạt tick xanh |

### Điều hiểu ra sau buổi thực hành
1. **Bẫy điểm số lạc quan (IID Trap):** Một mô hình đạt F1 rất cao (>95%) trên tập test thông thường vẫn có thể sụp đổ hoàn toàn khi triển khai thực tế (F1 giảm xuống 37%) nếu chỉ học vẹt tần suất từ khóa rời rạc mà không hiểu ngữ nghĩa.
2. **Đường biên Pareto trong kỹ thuật sản phẩm:** Trong môi trường sản xuất thực tế, mô hình tốt nhất không phải là mô hình thông minh nhất (như LLM) mà là mô hình cân bằng tối ưu giữa **Chất lượng (F1) – Tốc độ (p95) – Chi phí (USD)**.

