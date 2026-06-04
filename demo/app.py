"""Gradio UI để test trực quan model rewrite đã train (served qua vLLM + ngrok).

Thiết kế: BẤM-LÀ-CHẠY. Hàng chục mẫu test thật được nạp sẵn từ chính bộ benchmark
(data/bench/dialogues_bench_browser.jsonl, có sẵn đáp án chuẩn). Bấm 1 dòng trong
bảng → tự gửi model và hiện kết quả cạnh đáp án chuẩn. Muốn tự nhập thì mở mục
"Tự nhập hội thoại" ở dưới.

Model được serve bằng kaggle_serve.ipynb (vLLM, OpenAI-compatible, phơi qua ngrok).
UI gửi hội thoại theo ĐÚNG format lúc train (giống src/eval/eval_bench.py):
system = SYSTEM_PROMPT_FOR_TRAINING, lịch sử nhiều lượt, lượt user cuối gắn tag
<REWRITE>, model trả JSON {"rewrite_message": ...}.

Chạy:
    pip install gradio
    python demo/app.py        # mở http://127.0.0.1:7860, dán vLLM URL (ngrok) rồi bấm 1 mẫu

Prefill sẵn:
    VLLM_URL=https://xxx.ngrok-free.dev python demo/app.py
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import gradio as gr
from openai import OpenAI

# Cho phép chạy `python demo/app.py` từ gốc repo mà vẫn import được src.*
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.prompts import (  # noqa: E402
    BASELINE_SYSTEM_PROMPT,
    REWRITE_TAG,
    SYSTEM_PROMPT_FOR_TRAINING,
)

# ---- Mặc định (override được bằng env hoặc trong UI) -------------------------
DEFAULT_VLLM_URL = os.environ.get("VLLM_URL", "")
DEFAULT_LORA_ID = os.environ.get("LORA_ID", "vi-rewriter")
DEFAULT_BASE_ID = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

BENCH_PATH = ROOT / "data/bench/dialogues_bench_browser.jsonl"
N_PER_PATTERN = 8  # số mẫu lấy mỗi pattern để bảng đa dạng


# ---- Nạp mẫu test thật từ benchmark -----------------------------------------

def _record_to_turns_gold(r: dict) -> tuple[list[dict], str]:
    """LF record → (turns [{'role':'user'|'bot','content':..}], gold rewrite text).

    Bỏ system; lượt gpt cuối là gold (JSON); các lượt trước là hội thoại; gỡ tag
    <REWRITE> khỏi lượt user cuối. Lượt cuối của turns LUÔN là user (câu cần rewrite)."""
    body = [c for c in r["conversations"] if c["from"] != "system"]
    gold = json.loads(body[-1]["value"])["rewrite_message"]
    turns = []
    for c in body[:-1]:
        role = "user" if c["from"] == "human" else "bot"
        content = c["value"]
        if content.startswith(f"{REWRITE_TAG}\n"):
            content = content[len(REWRITE_TAG) + 1:]
        turns.append({"role": role, "content": content})
    return turns, gold


def load_samples() -> list[dict]:
    """Lấy bộ mẫu đa dạng (cân bằng theo pattern) từ benchmark. Fallback nếu thiếu file."""
    if not BENCH_PATH.exists():
        return [
            {"turns": [{"role": "user", "content": "Mở điều hoà."},
                       {"role": "bot", "content": "Bạn muốn đặt bao nhiêu độ?"},
                       {"role": "user", "content": "27 độ."}],
             "gold": "Đặt điều hoà 27 độ.", "pattern": "multi_turn_slot", "domain": "climate"},
        ]
    by_pattern: dict[str, list[dict]] = {}
    for line in BENCH_PATH.open(encoding="utf-8"):
        r = json.loads(line)
        by_pattern.setdefault(r["meta"].get("pattern", "?"), []).append(r)

    rng = random.Random(13)
    out = []
    for pattern in sorted(by_pattern):
        recs = by_pattern[pattern]
        rng.shuffle(recs)
        for r in recs[:N_PER_PATTERN]:
            turns, gold = _record_to_turns_gold(r)
            out.append({
                "turns": turns,
                "gold": gold,
                "pattern": pattern,
                "domain": r["meta"].get("domain", ""),
            })
    rng.shuffle(out)
    return out


SAMPLES = load_samples()
# Mỗi dòng bảng: [pattern, lĩnh vực, câu cuối của người dùng]
SAMPLE_ROWS = [
    [s["pattern"], s["domain"], s["turns"][-1]["content"]]
    for s in SAMPLES
]


# ---- Helpers -----------------------------------------------------------------

def _to_chatbot(turns: list[dict]) -> list[dict]:
    """turns -> messages cho gr.Chatbot ('bot' hiển thị là assistant)."""
    return [
        {"role": "user" if t["role"] == "user" else "assistant", "content": t["content"]}
        for t in turns
    ]


def build_trained_messages(turns: list[dict]) -> list[dict]:
    """Format ĐÚNG như lúc train: system ngắn + multi-turn + tag <REWRITE> ở lượt user cuối."""
    last_user = max(i for i, t in enumerate(turns) if t["role"] == "user")
    msgs = [{"role": "system", "content": SYSTEM_PROMPT_FOR_TRAINING}]
    for i, t in enumerate(turns):
        role = "user" if t["role"] == "user" else "assistant"
        content = t["content"]
        if i == last_user:
            content = f"{REWRITE_TAG}\n{content}"
        msgs.append({"role": role, "content": content})
    return msgs


def build_baseline_messages(turns: list[dict]) -> list[dict]:
    """Prompt cho base model CHƯA train: system chi tiết + 1 user message gồm hội thoại
    dạng nhãn (giống src/eval/eval_bench.lf_to_baseline_query, bỏ few-shot cho gọn)."""
    lines = [
        f"{'Người dùng' if t['role'] == 'user' else 'Trợ lý'}: {t['content']}"
        for t in turns
    ]
    return [
        {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    ]


def parse_rewrite(raw: str) -> str:
    """Lấy câu rewrite từ output (JSON {"rewrite_message": ...} hoặc text thuần)."""
    txt = (raw or "").strip()
    try:
        obj = json.loads(txt)
        if isinstance(obj, dict) and "rewrite_message" in obj:
            return str(obj["rewrite_message"]).strip()
    except (json.JSONDecodeError, ValueError):
        pass
    return txt


def _call(client: OpenAI, model: str, messages: list[dict], temperature: float, max_tokens: int) -> str:
    resp = client.chat.completions.create(
        model=model, messages=messages,
        temperature=float(temperature), max_tokens=int(max_tokens), top_p=0.9,
    )
    return resp.choices[0].message.content.strip()


def check_health(vllm_url):
    if not (vllm_url or "").strip():
        return "ℹ️ Chưa nhập vLLM URL."
    try:
        client = OpenAI(base_url=vllm_url.rstrip("/") + "/v1", api_key="EMPTY")
        ids = [m.id for m in client.models.list().data]
        return "✅ Kết nối OK. Models: " + ", ".join(ids)
    except Exception as e:  # noqa: BLE001
        return f"❌ Không kết nối được: {e}"


# ---- Chạy rewrite ------------------------------------------------------------

def run_rewrite(turns, vllm_url, lora_id, base_id, temperature, max_tokens, do_compare):
    """Trả: (lora_rewrite, lora_raw, base_out_update, base_raw_update)."""
    if not turns:
        raise gr.Error("Chưa có hội thoại — bấm 1 mẫu ở bảng, hoặc tự nhập ở dưới.")
    if turns[-1]["role"] != "user":
        raise gr.Error("Lượt CUỐI phải là của người dùng (đó là câu sẽ được rewrite).")
    if not (vllm_url or "").strip():
        raise gr.Error("Chưa có vLLM URL — dán URL ngrok (từ kaggle_serve.ipynb) ở ô trên cùng.")

    client = OpenAI(base_url=vllm_url.rstrip("/") + "/v1", api_key="EMPTY")
    try:
        raw = _call(client, lora_id, build_trained_messages(turns), temperature, max_tokens)
        lora_rewrite, lora_raw = parse_rewrite(raw), raw
    except Exception as e:  # noqa: BLE001
        lora_rewrite, lora_raw = f"⚠️ Lỗi gọi model: {e}", str(e)

    if not do_compare:
        return lora_rewrite, lora_raw, gr.update(visible=False), gr.update(visible=False)

    try:
        raw_b = _call(client, base_id, build_baseline_messages(turns), temperature, max_tokens)
        base_rewrite, base_raw = parse_rewrite(raw_b), raw_b
    except Exception as e:  # noqa: BLE001
        base_rewrite, base_raw = f"⚠️ Lỗi gọi base: {e}", str(e)

    return (
        lora_rewrite, lora_raw,
        gr.update(value=base_rewrite, visible=True),
        gr.update(value=base_raw, visible=True),
    )


def pick_sample(idx, vllm_url, lora_id, base_id, temperature, max_tokens, do_compare):
    """Bấm 1 dòng trong bảng → nạp hội thoại + gold, rồi tự chạy (nếu đã có URL).

    `idx` đến từ chính gr.Dataset nằm đầu danh sách inputs — tuỳ phiên bản Gradio có
    thể là int (type="index") hoặc cả hàng [pattern, domain, câu cuối]; xử lý cả hai."""
    if isinstance(idx, (list, tuple)):
        row = list(idx)
        idx = SAMPLE_ROWS.index(row) if row in SAMPLE_ROWS else 0
    s = SAMPLES[int(idx)]
    turns = s["turns"]
    chat = _to_chatbot(turns)
    gold = s["gold"]
    if (vllm_url or "").strip():
        lora_rewrite, lora_raw, base_u, base_raw_u = run_rewrite(
            turns, vllm_url, lora_id, base_id, temperature, max_tokens, do_compare
        )
    else:
        lora_rewrite = "⬆️ Dán vLLM URL ở ô trên cùng rồi bấm lại mẫu này để chạy."
        lora_raw = ""
        base_u = gr.update(value="", visible=bool(do_compare))
        base_raw_u = gr.update(value="", visible=bool(do_compare))
    return turns, chat, gold, lora_rewrite, lora_raw, base_u, base_raw_u


# ---- Tự nhập thủ công (mục phụ) ----------------------------------------------

def add_turn(turns, role, content):
    content = (content or "").strip()
    if not content:
        return turns, _to_chatbot(turns), ""
    turns = turns + [{"role": role, "content": content}]
    return turns, _to_chatbot(turns), ""


def undo_turn(turns):
    turns = turns[:-1] if turns else turns
    return turns, _to_chatbot(turns)


def clear_turns():
    return [], [], ""


# ---- UI ----------------------------------------------------------------------

def build_ui():
    with gr.Blocks(title="Vietnamese Dialogue Rewriter — Demo", fill_width=True) as demo:
        gr.Markdown(
            "# 🚗 Vietnamese Dialogue Rewriter — Demo\n"
            "**Cách dùng:** ① dán vLLM URL → ② bấm một dòng trong bảng mẫu. "
            "Model đã train sẽ viết lại **câu cuối của người dùng** và hiện cạnh đáp án chuẩn."
        )

        with gr.Row():
            vllm_url = gr.Textbox(
                label="① vLLM URL (ngrok từ kaggle_serve.ipynb)",
                value=DEFAULT_VLLM_URL, placeholder="https://xxxx.ngrok-free.dev", scale=4,
            )
            health_btn = gr.Button("Kiểm tra kết nối", scale=1)
        health_out = gr.Markdown("")

        with gr.Accordion("⚙️ Tuỳ chọn (model id, so sánh, sampling)", open=False):
            with gr.Row():
                lora_id = gr.Textbox(label="LoRA model id", value=DEFAULT_LORA_ID)
                base_id = gr.Textbox(label="Base model id (để so sánh)", value=DEFAULT_BASE_ID)
            with gr.Row():
                do_compare = gr.Checkbox(value=True, label="So sánh với base model (chưa train)")
                temperature = gr.Slider(0.0, 1.0, value=0.1, step=0.05, label="Temperature")
                max_tokens = gr.Slider(32, 512, value=160, step=16, label="Max tokens")

        turns_state = gr.State([])

        gr.Markdown(f"### ② Chọn một mẫu test — bấm vào dòng bất kỳ ({len(SAMPLES)} mẫu)")
        samples_tbl = gr.Dataset(
            components=[gr.Textbox(visible=False), gr.Textbox(visible=False), gr.Textbox(visible=False)],
            headers=["Pattern", "Lĩnh vực", "Câu cuối của người dùng"],
            samples=SAMPLE_ROWS,
            samples_per_page=12,
            type="index",
            label=None,
        )

        gr.Markdown("### Kết quả")
        with gr.Row():
            with gr.Column(scale=1):
                chat = gr.Chatbot(height=340, label="Hội thoại (lượt cuối = câu được rewrite)")
                gold_box = gr.Textbox(label="✅ Đáp án chuẩn (gold)", lines=2)
            with gr.Column(scale=1):
                lora_out = gr.Textbox(label="🤖 Model đã train → câu rewrite", lines=3)
                base_out = gr.Textbox(label="📦 Base model (chưa train) → câu rewrite", lines=3, visible=True)
                with gr.Accordion("Output thô của model (debug)", open=False):
                    lora_raw = gr.Textbox(label="LoRA raw", lines=2)
                    base_raw = gr.Textbox(label="Base raw", lines=2, visible=True)
                rerun_btn = gr.Button("🔄 Chạy lại mẫu hiện tại", variant="primary")

        with gr.Accordion("✍️ Tự nhập hội thoại (nâng cao)", open=False):
            gr.Markdown("Thêm từng lượt; lượt **cuối phải là user**. Xong bấm *Chạy lại*.")
            with gr.Row():
                role = gr.Radio(["user", "bot"], value="user", label="Vai", scale=1)
                content = gr.Textbox(label="Nội dung lượt", scale=3, placeholder="Nhập rồi Enter / bấm Thêm…")
            with gr.Row():
                add_btn = gr.Button("➕ Thêm lượt")
                undo_btn = gr.Button("↩️ Bỏ lượt cuối")
                clear_btn = gr.Button("🗑️ Xoá hết")

        run_io = [turns_state, vllm_url, lora_id, base_id, temperature, max_tokens, do_compare]
        result_out = [lora_out, lora_raw, base_out, base_raw]

        # --- wiring ---
        health_btn.click(check_health, [vllm_url], [health_out])
        samples_tbl.click(
            pick_sample,
            [samples_tbl, vllm_url, lora_id, base_id, temperature, max_tokens, do_compare],
            [turns_state, chat, gold_box, *result_out],
        )
        rerun_btn.click(run_rewrite, run_io, result_out)
        add_btn.click(add_turn, [turns_state, role, content], [turns_state, chat, content])
        content.submit(add_turn, [turns_state, role, content], [turns_state, chat, content])
        undo_btn.click(undo_turn, [turns_state], [turns_state, chat])
        clear_btn.click(clear_turns, None, [turns_state, chat, gold_box])

    return demo


def _free_port(preferred: int = 7860) -> int:
    """Trả `preferred` nếu trống, ngược lại để OS cấp 1 cổng trống bất kỳ."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s2:
                s2.bind(("127.0.0.1", 0))
                return s2.getsockname()[1]


if __name__ == "__main__":
    port = _free_port(int(os.environ.get("PORT", 7860)))
    print(f"\n>> Mở UI tại http://127.0.0.1:{port}\n")
    # ssr_mode=False: Gradio 5/6 bật SSR (cần Node) mặc định → hay rớt websocket
    # "Connection to the server was lost". Tắt đi cho ổn định trên máy local.
    build_ui().queue().launch(
        server_name="127.0.0.1", server_port=port, ssr_mode=False, show_error=True,
    )
