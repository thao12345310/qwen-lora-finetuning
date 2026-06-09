"""GPT-4o-mini paraphrase augmentation for Llama Factory training data.

Reads template-converted LF jsonl, asks gpt-4o-mini for 1 paraphrased variant
per sample with strict slot preservation, and writes the combined originals +
variants out.

Usage:
    export OPENAI_API_KEY=sk-...
    python -m src.data.paraphrase_with_llm \\
        --input data/train/dialogues_train_template.jsonl \\
        --output data/train/dialogues_train.jsonl \\
        --concurrency 12

The script caches per-sample results in `<output>.cache.jsonl` for resumability.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm import tqdm

from src.data.prompts import REWRITE_TAG, SYSTEM_PROMPT_FOR_TRAINING

load_dotenv()


META_SYSTEM = """Bạn là chuyên gia paraphrase tiếng Việt cho dữ liệu huấn luyện task rewrite hội thoại xe ô tô.

Bạn nhận một mẫu LF gồm:
- "dialogue": list các turn xen kẽ "human"/"gpt", kết thúc bằng "human" (chứa <REWRITE>).
- "rewrite": câu rewrite chuẩn gói toàn bộ slot.

Sinh 1 BIẾN THỂ paraphrase mới — cùng số turn, cùng vai trò, cùng slot, nhưng diễn đạt khác hẳn.

YÊU CẦU TUYỆT ĐỐI:
1. Giữ NGUYÊN VẸN slot: số (digit, kể cả "27 độ" → vẫn "27 độ"), tên riêng (Bún Đậu Ngon, anh Nam, EBOOST, Em Của Ngày Hôm Qua), đơn vị (km/h, độ, %). KHÔNG thay slot.
2. Giữ NGUYÊN số lượng turn và vai trò (human/gpt). Turn cuối vẫn là "human" và VẪN bắt đầu bằng "<REWRITE>\\n".
3. Đa dạng surface form:
   - Đổi opener của rewrite — không dùng cùng cấu trúc với gốc. Ví dụ gốc "Tôi muốn ..." → biến thể có thể "Hãy ...", "Làm ơn ...", "Bạn ... giúp tôi", "Cho tôi ...", "Mình cần ...".
   - Đổi từ vựng đồng nghĩa cho động từ, liên từ.
   - Có thể lược chủ ngữ, dùng tiếng nói nhanh / slang nhẹ trong human turn.
   - Bot turn có thể đổi cách hỏi nhưng giữ cùng ý.
4. KHÔNG copy y nguyên bất kỳ turn nào từ gốc.
5. KHÔNG thêm intent mới hoặc slot mới. KHÔNG bỏ slot.

Trả về JSON: {"variant": {"dialogue": [{"from":"human","value":"..."},{"from":"gpt","value":"..."},...], "rewrite": "..."}}"""


SLOT_NUMBER_RE = re.compile(r"\d+")


def extract_slots(text: str) -> set[str]:
    """Numeric slots — digit runs. Cheap proxy for 'slot preserved'."""
    return set(SLOT_NUMBER_RE.findall(text))


def lf_to_compact(record: dict) -> tuple[list[dict], str]:
    """Strip system + final gpt → return (dialogue_turns_without_rewrite_tag, gold_rewrite)."""
    convs = [c for c in record["conversations"] if c["from"] != "system"]
    last_gpt = convs[-1]
    assert last_gpt["from"] == "gpt"
    rewrite = json.loads(last_gpt["value"])["rewrite_message"]
    dialogue = convs[:-1]
    # Strip <REWRITE> tag from final human for cleaner LLM input
    cleaned = []
    for c in dialogue:
        v = c["value"]
        if v.startswith(REWRITE_TAG + "\n"):
            v = v[len(REWRITE_TAG) + 1:]
        cleaned.append({"from": c["from"], "value": v})
    return cleaned, rewrite


def compact_to_lf(dialogue: list[dict], rewrite: str, meta: dict) -> dict:
    convs = [{"from": "system", "value": SYSTEM_PROMPT_FOR_TRAINING}]
    for c in dialogue[:-1]:
        convs.append({"from": c["from"], "value": c["value"]})
    last = dialogue[-1]
    assert last["from"] == "human"
    convs.append({"from": "human", "value": f"{REWRITE_TAG}\n{last['value']}"})
    answer = json.dumps({"rewrite_message": rewrite}, ensure_ascii=False)
    convs.append({"from": "gpt", "value": answer})
    return {"conversations": convs, "meta": {**meta, "augmented": True}}


def validate_variant(orig_dialogue: list[dict], orig_rewrite: str,
                     variant: dict) -> tuple[list[dict], str] | None:
    if not isinstance(variant, dict):
        return None
    v_dialogue = variant.get("dialogue")
    v_rewrite = (variant.get("rewrite") or "").strip()
    if not isinstance(v_dialogue, list) or not v_rewrite:
        return None
    if len(v_dialogue) != len(orig_dialogue):
        return None
    cleaned: list[dict] = []
    for o, v in zip(orig_dialogue, v_dialogue):
        if not isinstance(v, dict):
            return None
        if v.get("from") != o["from"]:
            return None
        val = (v.get("value") or "").strip()
        if not val:
            return None
        cleaned.append({"from": v["from"], "value": val})
    if cleaned[-1]["from"] != "human":
        return None

    # Slot integrity: every digit-run in the original rewrite must appear in variant
    # rewrite (slots can shift inside the dialogue but must survive in the final answer).
    orig_slots = extract_slots(orig_rewrite)
    var_slots = extract_slots(v_rewrite)
    if not orig_slots.issubset(var_slots):
        return None

    # Don't accept identical paraphrase (waste).
    if v_rewrite == orig_rewrite and all(
        c["value"] == o["value"] for c, o in zip(cleaned, orig_dialogue)
    ):
        return None
    return cleaned, v_rewrite


async def paraphrase_one(
    client: AsyncOpenAI,
    model: str,
    record: dict,
    sem: asyncio.Semaphore,
    max_retries: int = 3,
) -> dict | None:
    dialogue, rewrite = lf_to_compact(record)
    user_msg = (
        f"Mẫu gốc:\n"
        f"dialogue:\n{json.dumps(dialogue, ensure_ascii=False, indent=2)}\n\n"
        f"rewrite: {rewrite}\n\n"
        f"Sinh 1 biến thể paraphrase theo schema."
    )
    async with sem:
        for attempt in range(max_retries):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": META_SYSTEM},
                        {"role": "user", "content": user_msg},
                    ],
                    response_format={"type": "json_object"},
                    temperature=1.1,
                    max_tokens=800,
                )
                payload = json.loads(resp.choices[0].message.content)
                v = payload.get("variant")
                validated = validate_variant(dialogue, rewrite, v)
                if validated is None:
                    return None
                new_dialogue, new_rewrite = validated
                return compact_to_lf(new_dialogue, new_rewrite, record.get("meta", {}))
            except Exception as e:
                if attempt == max_retries - 1:
                    return None
                await asyncio.sleep(2 ** attempt)
        return None


async def run(args):
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set")

    records = [json.loads(l) for l in args.input.open(encoding="utf-8")]
    print(f"Loaded {len(records)} input samples from {args.input}")
    if args.limit:
        records = records[: args.limit]
        print(f"--limit applied: processing only {len(records)} samples")

    cache_path = args.output.with_suffix(".cache.jsonl")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done: dict[int, dict | None] = {}
    if cache_path.exists():
        for line in cache_path.open(encoding="utf-8"):
            rec = json.loads(line)
            done[rec["idx"]] = rec["variant"]
        print(f"Resumed {len(done)} cached entries from {cache_path}")

    todo = [(i, r) for i, r in enumerate(records) if i not in done]
    print(f"Calling {args.model} for {len(todo)} samples (concurrency={args.concurrency})")

    client = AsyncOpenAI()
    sem = asyncio.Semaphore(args.concurrency)
    tasks = [paraphrase_one(client, args.model, r, sem) for _, r in todo]

    cache_fh = cache_path.open("a", encoding="utf-8")
    pbar = tqdm(total=len(tasks), desc="paraphrase")

    async def wrap(idx, coro):
        v = await coro
        cache_fh.write(json.dumps({"idx": idx, "variant": v}, ensure_ascii=False) + "\n")
        cache_fh.flush()
        pbar.update(1)
        done[idx] = v

    await asyncio.gather(*(wrap(i, t) for (i, _), t in zip(todo, tasks)))
    pbar.close()
    cache_fh.close()

    # Assemble final: originals + valid variants, dedup.
    out: list[dict] = []
    seen: set[tuple] = set()

    def key_of(rec):
        last_human = next(c["value"] for c in reversed(rec["conversations"]) if c["from"] == "human")
        last_gpt = rec["conversations"][-1]["value"]
        return (last_human, last_gpt)

    for r in records:
        k = key_of(r)
        if k in seen:
            continue
        seen.add(k)
        out.append(r)

    n_variants_kept = 0
    n_variants_dropped = 0
    for i in range(len(records)):
        v = done.get(i)
        if v is None:
            n_variants_dropped += 1
            continue
        k = key_of(v)
        if k in seen:
            n_variants_dropped += 1
            continue
        seen.add(k)
        out.append(v)
        n_variants_kept += 1

    with args.output.open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(out)} samples to {args.output}")
    print(f"  originals : {len(records)}")
    print(f"  variants kept : {n_variants_kept}")
    print(f"  variants dropped : {n_variants_dropped}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path,
                        default=Path("data/train/dialogues_train_template.jsonl"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/train/dialogues_train.jsonl"))
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
