# Benchmark & Metrics — bản khớp repo thực tế

> Mục tiêu file này: chốt **bộ metric phù hợp** cho bài toán Vietnamese in-car contextual rewrite và **cách áp vào repo hiện tại**, không phá format Llama Factory và không cần annotate slot-gold thủ công (scope của team là **rewrite-only**).

---

## 0. Trạng thái thực tế (đối chiếu trước khi bàn metric)


| Hạng mục                      | Thực tế trong repo                                                                                                                 |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| Bench                         | `data/bench/dialogues_bench.jsonl` = **983 mẫu** (+1044 bản browser). Eval-only, tách khỏi train.                                  |
| Format                        | Llama Factory `conversations` (from/value); output gold = JSON `{"rewrite_message": "..."}`.                                       |
| Meta mỗi mẫu                  | `pattern`, `domain`, `rationale`, `user_turns`, `total_turns`.                                                                     |
| `pattern` (8, ~đều ~120/loại) | pronoun_resolution, irrelevant_context, multi_turn_slot, correction, code_switching, implicit_reference, negation, compound_intent |
| `domain` (9)                  | climate, music, calling, navigation, charging, smart_home, vehicle, messaging, driver_assist                                       |
| Eval hiện có                  | `src/eval/eval_bench.py`: predict qua vLLM → **LLM judge (GPT-4o / MiMo) chấm 0/1** → in **overall + theo pattern**.               |
| Task                          | rewrite-only, KHÔNG classify. Student ~2B, LoRA, distillation từ teacher Qwen-32B.                                                 |


**Hệ quả cho metric:**

- KHÔNG có `required_slots`/`intents` gold → **mọi metric kiểu Slot-EM / Intent-Accuracy tính trực tiếp đều bất khả thi** nếu không bịa thêm annotation. Ta lấy thông tin đó qua **LLM judge có cấu trúc** thay vì annotate tay.
- Bench **chưa có `context_required`** → hiện chưa tách được nhóm "cần ngữ cảnh" vs "chống nhiễu". Đây là khoảng trống lớn nhất cần lấp (mục 4).
- `pattern` đang **đơn nhãn** → đủ để báo cáo slice, chưa cần đổi sang đa nhãn ngay.

---

## 1. Định nghĩa task (cái mà metric phải đo)

> Cho lịch sử hội thoại + lượt user cuối (sau tag `<REWRITE>`), model sinh **một câu lệnh độc lập** thể hiện **ý định cuối cùng** của user, **chỉ kế thừa thông tin đã xác lập & còn hiệu lực**, **không thêm ràng buộc mới**, **không bịa**.

5 luật quyết định "rewrite đúng" (đây là tiêu chí judge phải bám, không phải schema dữ liệu):

1. **context_required**: mẫu là `true` chỉ khi bỏ hết lịch sử thì người không thể khôi phục đủ intent/slot. Lịch sử chỉ gây nhiễu (số liệu lạc đề) → `false`.
2. **Correction ghi đè**: chỉ giữ giá trị sau cùng ("24°C → đổi 26°C" ⇒ gold chỉ có 26°C).
3. **Chỉ kế thừa constraint user đã chọn/xác nhận**: bot mô tả tuyến hiện tại không có nghĩa user muốn "tránh tuyến đó" — không tự thêm.
4. **Không resolve được thì giữ nguyên, không bịa** ("nhà tôi" khi chưa có địa chỉ → giữ "nhà tôi").
5. **Không ép một cách diễn đạt**: paraphrase khác nhau vẫn đúng nếu intent + slot khớp ⇒ metric phải đo **ý nghĩa**, không đo text khít.

→ Luật 5 chính là lý do **LLM judge ngữ nghĩa là trục chính**, còn Exact Match/BLEU chỉ là phụ.

---

## 2. Bộ metric đề xuất — chia tầng theo "đo gì + áp thế nào"

Triết lý: **judge ngữ nghĩa là xương sống**, bọc thêm các check **rẻ, xác định (deterministic)** ở dưới và một metric **so sánh-được-với-paper** ở trên.

### Tầng 0 — Deterministic, miễn phí, không cần gold (tính thẳng từ chuỗi pred)


| Metric                   | Đo gì                                                                             | Cách áp vào repo                                                                                                             |
| ------------------------ | --------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| **Output Validity Rate** | Student (train LF) có nhả đúng JSON `{"rewrite_message": "..."}` parse được không | Parse pred; với adapter cũ output plain-text thì check non-empty/không `__ERROR_`_. Tính trước khi đưa vào judge.            |
| **Copy Rate**            | Pred có y hệt lượt user cuối (chỉ bỏ tag) không → dấu hiệu **không rewrite**      | So `normalize(pred) == normalize(last_user_utterance)`. Cao ở nhóm `context_required=true` = model lười, không dùng context. |
| **Degenerate Rate**      | Rỗng / lặp / quá ngắn-dài bất thường                                              | Empty, hoặc `len(pred)/len(gold)` ngoài [0.3, 3.0].                                                                          |


> Tầng 0 là **sanity gate**: phải sạch thì các metric trên mới có nghĩa. Hiện `eval_bench.py` chưa tính các chỉ số này.

### Tầng 1 — Semantic correctness qua LLM judge (TRỤC CHÍNH)

Hiện judge trả `{score: 0/1}`. **Nâng cấp quan trọng:** đổi rubric để judge trả **các cờ con có cấu trúc**, vẫn 1 lần gọi, **không cần slot-gold**:

```json
{
  "intent_ok": 0/1,           // đúng hành động (bật/tắt/đổi/hủy/thêm)
  "slots_complete": 0/1,      // không THIẾU slot quan trọng có trong gold
  "no_hallucination": 0/1,    // không THÊM slot/thông tin sai (vd kéo nhầm số bot mention)
  "negation_ok": 0/1,         // giữ đúng phủ định
  "score": 0/1,               // = AND của các cờ trên (command-level đúng)
  "reason": "..."
}
```

Từ đó suy ra **trọn bộ metric "kiểu slot" mà không annotate gì thêm** — judge gánh phần suy luận slot:


| Metric                          | Công thức từ cờ                        | Ý nghĩa                                                                |
| ------------------------------- | -------------------------------------- | ---------------------------------------------------------------------- |
| **Command Accuracy** (headline) | mean(`score`)                          | Lệnh khôi phục đúng hoàn toàn. Đây là số chốt giữa các model.          |
| **Intent Accuracy**             | mean(`intent_ok`)                      | Sai hành động.                                                         |
| **Slot Completeness**           | mean(`slots_complete`)                 | Bỏ sót slot (proxy của Slot-Recall).                                   |
| **Hallucinated Slot Rate**      | mean(`1 - no_hallucination`)           | **Metric an toàn quan trọng nhất** với trợ lý xe — thêm thông tin sai. |
| **Negation Preservation**       | mean(`negation_ok`) trên nhóm negation | Mất phủ định = đảo lệnh, rất nguy hiểm.                                |


> Đây là cách lấy "Intent/Slot/Hallucination metrics" mà plan cũ đòi hỏi, **nhưng không phá scope rewrite-only**: ta không bắt model output slot, cũng không annotate gold slot — chỉ mở rộng schema output của judge.

### Tầng 2 — Reference-overlap, để SO SÁNH với paper IUR (phụ, tự động)


| Metric                     | Đo gì                                                                                          | Cách áp                                                                                                                                                                                                                    |
| -------------------------- | ---------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Normalized Exact Match** | Trùng khít sau chuẩn hoá (lowercase, bỏ dấu câu, chuẩn số "22 độ"≈"22°C")                      | Sanity check; kỳ vọng thấp vì paraphrase. Báo cáo, **không tối ưu theo nó**.                                                                                                                                               |
| **chrF / ROUGE-L / BLEU**  | Overlap bề mặt                                                                                 | Để đối chiếu với literature query/utterance rewrite (CQR…). Phụ.                                                                                                                                                           |
| **Restoration-F (F1/F2)**  | Riêng phần **token cần khôi phục từ context** (= `tokens(gold) \ tokens(last_user_utterance)`) | **Metric native của IUR** — đo đúng "model có phục hồi phần bị tỉnh lược không". Tính **tự động** từ field sẵn có, không cần gold slot. Rất hợp để đưa vào báo cáo nghiên cứu. F2 nhấn recall (sót thông tin tệ hơn thừa). |


> Restoration-F là điểm "research-grade" đáng đầu tư: nó phân biệt được model **chỉ copy câu cuối** (restoration≈0) với model **thật sự dùng context**, theo cách định lượng và reproducible — bổ sung tốt cho judge.

### Tầng 3 — Slice / breakdown (giá trị phân tích lớn nhất)

Mọi metric Tầng 1 nên cắt theo:


| Slice                       | Cần gì                          | Vì sao                                                                                                                              |
| --------------------------- | ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| theo `pattern`              | đã có                           | đã chạy.                                                                                                                            |
| theo `**context_required`** | **thêm field vào meta** (mục 4) | Tách 2 năng lực lõi: **Context-Dependent Accuracy** (acc khi `true`) vs **Distractor Robustness** (acc + over-rewrite khi `false`). |
| theo `domain`               | đã có                           | tìm domain yếu.                                                                                                                     |
| theo `total_turns`          | đã có                           | xem có sụp ở hội thoại dài không → cơ sở quyết định có cần context-tracking về sau.                                                 |


Hai metric phái sinh đáng tên riêng:

- **Over-rewrite Rate** (trên `context_required=false`): pred có kéo thêm thông tin từ lịch sử mà gold không có → dùng cờ `no_hallucination` trên nhóm này.
- **Context-Use Gain** = Command Acc(`true`) − Copy-baseline Acc(`true`): chứng minh model **thật sự dùng context** chứ không ăn may.

---

## 3. Hai baseline rẻ phải có để metric "có mốc"

Không cần CAT, không cần kiến trúc mới. Chỉ cần 2 mốc để diễn giải con số:

1. **Copy baseline**: output = lượt user cuối (bỏ tag). Cho biết nhóm `context_required=false` dễ tới đâu, và `true` không thể giải bằng copy.
2. **Prompt-only baseline**: chính model 2B chưa fine-tune + prompt. Cho biết fine-tune **thật sự thêm bao nhiêu**, không phải chỉ đổi văn phong.

Mọi cải thiện của student fine-tuned chỉ có nghĩa khi đặt cạnh 2 mốc này — nhất là trên slice `context_required=true`.

---

## 4. Cần thêm gì vào repo để bật full suite (nhỏ, cụ thể)

Theo thứ tự ưu tiên, đều là thay đổi nhỏ:

1. **Thêm `context_required` (bool) vào `meta` của bench.** Mở khoá toàn bộ Tầng-3 slice quan trọng nhất. Có thể backfill bán tự động: rule "câu cuối tự đủ nghĩa?" + review nhanh, hoặc nhờ judge gán rồi người duyệt mẫu.
2. **Đổi `JUDGE_SYSTEM` sang rubric có cấu trúc** (4 cờ + score) ở `eval_bench.py`. Một thay đổi prompt + parse thêm field; chi phí gọi không đổi.
3. **Thêm hàm metric Tầng-0** (validity/copy/degenerate) — thuần Python, không gọi API.
4. **Thêm `restoration_f.py`** (token-set F1/F2 so với `gold \ last_user`). Thuần Python.
5. **Mở rộng phần Report**: in bảng metric × slice (pattern / context_required / domain), không chỉ overall + pattern.

(Tuỳ chọn, để sau) `pattern` đơn nhãn → `challenge_types` đa nhãn nếu cần error-attribution mịn hơn. Không chặn việc gì.

---

## 5. Judge chính là metric → phải kiểm định độ tin cậy

Vì điểm số phụ thuộc judge, cần kỷ luật:

- **Validate judge một lần**: lấy ~~80–100 mẫu, người chấm 0/1, đo **agreement** judge vs người (mục tiêu ≥~~90%, hoặc Cohen's κ). Nếu lệch, sửa rubric.
- **Đóng băng judge** (model + prompt + version) suốt một đợt so sánh; đổi judge = mọi số cũ hết so sánh được. Ghi `judge_model` vào output (đã lưu trong `preds_*.jsonl`).
- **MiMo judge: `--judge-concurrency 1`** để né 429 (đã biết — [[mimo-eval-judge]]).
- Báo cáo **judge_error_rate**; loại các mẫu `__JUDGE_ERROR_`_ khỏi mẫu số hoặc xử lý rõ ràng, đừng tính ngầm thành 0.

---

## 6. Báo cáo cuối (định dạng)

Mỗi model một bảng:

```
Model X
  Output Validity   : 99.2%
  Copy Rate         : 11.0%   (cao ở context_required=true là xấu)
  Command Accuracy  : 78.4%   ← headline
  Intent Acc        : 91.0%
  Slot Completeness : 82.1%
  Halluc. Slot Rate :  6.3%   ← càng thấp càng tốt
  Negation Preserv. : 88.0%   (trên nhóm negation)
  Restoration-F2    : 0.71
                         ┌ context_required=true : 72.0%  (Context-Dependent Acc)
  Command Acc theo slice ┤ context_required=false: 90.0%  (Over-rewrite 4%)
                         └ theo pattern / domain : …bảng…
```

Kèm **error-analysis** đọc tay các mẫu sai, phân loại: wrong entity / wrong numeric slot / missed carried slot / kept-obsolete-after-correction / lost negation / added-unsupported-constraint / invalid-output / over-rewrite. Bảng này → quyết định bước data tiếp theo (tăng correction samples? numeric? multi-candidate?).

---

## 7. Thứ tự thực hiện (gọn)

```
1. Thêm context_required vào bench meta            (mở khoá slice lõi)
2. Nâng judge rubric → 4 cờ + score                (mở khoá Intent/Slot/Halluc/Negation)
3. Thêm metric Tầng-0 + Restoration-F              (deterministic, miễn phí)
4. Mở rộng report theo slice                       (pattern × context_required × domain)
5. Chạy 2 baseline (copy, prompt-only)             (lấy mốc)
6. Chạy student fine-tuned, so với mốc             (chứng minh context-use gain)
7. Error analysis theo bảng → định hướng data đợt sau
```

Trục xuyên suốt: **command-level đúng (judge) + không hallucinate** là 2 metric quyết định cho trợ lý xe; Restoration-F + slice `context_required` là phần chứng minh "model dùng context thật". Không cần slot-gold, không cần CAT để bắt đầu.