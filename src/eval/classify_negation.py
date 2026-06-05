"""Classify each pattern=negation bench gold's negative clause as no_op vs load_bearing.

no_op       — negates an INDEPENDENT on/off action whose target is off-by-default and
              uncoupled, so omitting the clause changes neither the tool-call set nor the
              device state (e.g. "bật A, đừng bật B" with B already off & independent).
load_bearing— the negation is (a) a PARAMETER/MODE of an action being executed
              ("đừng dùng sạc nhanh" while charging), (b) a CONSTRAINT ("không đi qua cầu X"),
              or (c) a CANCEL/undo of an already-done or coupled action ("đừng mở nữa").

Used to estimate the *real* negation regression after forgiving no_op clause-drops.
Cached to a jsonl so a quota wall doesn't lose work — re-run to resume.
"""
from __future__ import annotations
import asyncio, json, os, re
from pathlib import Path
from openai import AsyncOpenAI
from src.eval.eval_bench import _extract_json, lf_to_old_turns, format_old_user_msg

BENCH = Path("data/bench/dialogues_bench_browser.jsonl")
CACHE = Path("data/bench/eval_results/full_gptoss/negation_types.jsonl")
MODEL = "gpt-oss:120b-cloud"

SYS = """Bạn phân loại MỆNH ĐỀ PHỦ ĐỊNH trong câu lệnh 'gold'.
BỐI CẢNH: câu lệnh (rewrite) được đưa cho một model nhỏ để GỌI TOOL. Một thiết bị/hành động chỉ thay đổi khi có LỆNH DƯƠNG tương ứng; không nhắc tới thì không gọi tool và trạng thái giữ nguyên.

Phân loại vế phủ định của gold:
- "no_op": phủ định một HÀNH ĐỘNG bật/mở/phát ĐỘC LẬP mà mục tiêu vốn ĐANG TẮT và không bị ràng buộc/đính kèm. Bỏ vế này thì tập tool-call và trạng thái KHÔNG đổi (vì vốn không có lệnh dương cho nó). VD: "bật giữ làn, đừng bật auto pilot" (auto pilot vốn tắt, độc lập); "phát bản acoustic, đừng phát bản gốc" (chỉ phát được 1 bản).
- "load_bearing": phủ định là MỘT trong:
   (a) THAM SỐ/CHẾ ĐỘ của hành động đang thực hiện — vd "đừng dùng sạc nhanh" khi đang sạc, "không dùng điều hoà ngoài";
   (b) RÀNG BUỘC — vd "không đi qua cầu Nhật Tân", "tránh đường X";
   (c) HỦY/đảo một hành động VỪA xảy ra hoặc bị COUPLING — vd cửa vừa mở → "đừng mở nữa".
  Bỏ vế này SẼ làm sai tool-call hoặc tham số.
- "none": gold không chứa phủ định.

Quyết định dựa trên dialogue (để biết thiết bị đang bật/tắt, có coupling không).
Trả về DUY NHẤT JSON: {"type":"no_op"|"load_bearing"|"none","reason":"1 câu ngắn"}"""


async def classify(client, dialogue, gold, sem):
    user = f"dialogue:\n{dialogue}\n\ngold:\n{gold}\n\nPhân loại vế phủ định."
    async with sem:
        for attempt in range(6):
            try:
                resp = await client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "system", "content": SYS}, {"role": "user", "content": user}],
                    temperature=0.0, max_tokens=512, reasoning_effort="low",
                )
                d = _extract_json(resp.choices[0].message.content)
                t = d.get("type", "")
                if t not in ("no_op", "load_bearing", "none"):
                    t = "load_bearing"  # safe default: don't forgive ambiguous
                return {"type": t, "reason": d.get("reason", "")}
            except Exception as e:
                if attempt == 5:
                    return {"type": "__ERROR__", "reason": str(e)[:160]}
                await asyncio.sleep(min(2 ** attempt, 5))


async def main():
    records = [json.loads(l) for l in BENCH.open(encoding="utf-8")]
    neg_idx = [i for i, r in enumerate(records) if r["meta"].get("pattern") == "negation"]
    print(f"{len(neg_idx)} negation rows")

    cache = {}
    if CACHE.exists():
        for l in CACHE.open(encoding="utf-8"):
            o = json.loads(l)
            if o["type"] != "__ERROR__":
                cache[o["idx"]] = o
    todo = [i for i in neg_idx if i not in cache]
    print(f"cached {len(cache)}, to classify {len(todo)}")

    client = AsyncOpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
    sem = asyncio.Semaphore(4)

    async def one(i):
        turns, gold = lf_to_old_turns(records[i])
        res = await classify(client, format_old_user_msg(turns), gold, sem)
        return {"idx": i, "gold": gold, **res}

    results = await asyncio.gather(*(one(i) for i in todo))
    with CACHE.open("a", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"appended {len(results)} → {CACHE}")


if __name__ == "__main__":
    asyncio.run(main())
