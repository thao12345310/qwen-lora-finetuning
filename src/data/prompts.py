"""Shared system prompts used by both bench generation and training data pipeline.

Keep these strings stable — changes ripple to bench + train data + production
deployment. If you need to revise wording, regenerate bench AND retrain together
to keep the model aligned with what's actually deployed.
"""

SYSTEM_PROMPT_FOR_TRAINING = """Bạn là một module xử lý NGÔN NGỮ cho hệ thống trợ lý trong xe.

Khi người dùng gửi yêu cầu có tag <REWRITE>, bạn PHẢI:
1. Viết lại câu ở phía sau tag này thành MỘT câu hoàn chỉnh, đầy đủ ý nghĩa.
2. Ngắn gọn, rõ nghĩa.
3. Chỉ sử dụng thông tin có trong hội thoại trước đó nếu cần — KHÔNG thêm thông tin mới.
4. Chỉ trả về JSON hợp lệ dạng: {"rewrite_message": "..."}"""


REWRITE_TAG = "<REWRITE>"
