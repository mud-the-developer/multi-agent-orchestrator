import json
import re
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# =====================
# config
# =====================
BASE_ID = "google/gemma-4-31B-it"
ADAPTER_ID = "kai-os/gemma4-opus-reasoning-adapter-v1"

MMLU_SUBSET = "test[:100]"     # 빠르게 보려면 100, 좀 더 보려면 500
HUMANEVAL_LIMIT = None         # 예: 20 으로 두면 20개만 생성
MAX_NEW_TOKENS_MMLU = 16
MAX_NEW_TOKENS_HE = 256

# 공개 비교 숫자
PUBLIC_COMPARE = {
    "Gemma 4 31B (official)": {
        "MMLU-Pro": 85.2,
        "GPQA Diamond": 84.3,
        "LiveCodeBench v6": 80.0,
    }
}

# =====================
# load model
# =====================
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(BASE_ID)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

base = AutoModelForCausalLM.from_pretrained(
    BASE_ID,
    device_map="auto",
    quantization_config=bnb,
    torch_dtype=torch.bfloat16,
)
model = PeftModel.from_pretrained(base, ADAPTER_ID)
model.eval()

device = next(model.parameters()).device


# =====================
# helpers
# =====================
def generate_text(user_prompt: str, max_new_tokens: int = 64, do_sample: bool = False) -> str:
    messages = [{"role": "user", "content": user_prompt}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=0.7 if do_sample else None,
            top_p=0.95 if do_sample else None,
            pad_token_id=tokenizer.eos_token_id,
        )

    gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    return text


def extract_choice(text: str):
    m = re.search(r"\b([A-J])\b", text.upper())
    return m.group(1) if m else None


def strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```python"):
        text = text[len("```python"):].strip()
    elif text.startswith("```"):
        text = text[len("```"):].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text


def postprocess_humaneval_completion(text: str) -> str:
    text = strip_code_fence(text)

    stop_tokens = [
        "\nclass ",
        "\ndef ",
        "\nif __name__",
        "\nprint(",
        "\n#",
        "\n```",
    ]
    end = len(text)
    for tok in stop_tokens:
        idx = text.find(tok)
        if idx != -1:
            end = min(end, idx)
    return text[:end].rstrip()


# =====================
# MMLU-Pro eval
# =====================
def eval_mmlu_pro():
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split=MMLU_SUBSET)

    correct = 0
    total = 0
    rows = []

    for row in ds:
        choices = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(row["options"])])
        prompt = f"""Answer the multiple choice question.
Return only one letter: A, B, C, D, E, F, G, H, I, or J.

Question:
{row["question"]}

Choices:
{choices}
"""

        out = generate_text(prompt, max_new_tokens=MAX_NEW_TOKENS_MMLU, do_sample=False)
        pred = extract_choice(out)
        gold = chr(65 + row["answer_index"])

        ok = pred == gold
        total += 1
        correct += int(ok)

        rows.append({
            "question": row["question"],
            "pred": pred,
            "gold": gold,
            "raw_output": out,
            "correct": ok,
        })

        print(f"[MMLU-Pro] {total}/{len(ds)} pred={pred} gold={gold} correct={ok}")

    acc = correct / total if total else 0.0

    with open("mmlu_pro_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "subset": MMLU_SUBSET,
            "correct": correct,
            "total": total,
            "accuracy": acc,
            "details": rows,
        }, f, ensure_ascii=False, indent=2)

    return {
        "correct": correct,
        "total": total,
        "accuracy": acc,
    }


# =====================
# HumanEval sample generation
# =====================
def build_humaneval_prompt(prompt: str) -> str:
    return (
        "Complete the following Python function.\n"
        "Return only valid Python code for the missing implementation.\n\n"
        f"{prompt}"
    )


def generate_humaneval_samples():
    ds = load_dataset("openai/openai_humaneval", split="test")
    if HUMANEVAL_LIMIT is not None:
        ds = ds.select(range(min(HUMANEVAL_LIMIT, len(ds))))

    samples = []

    for i, row in enumerate(ds):
        prompt = build_humaneval_prompt(row["prompt"])
        out = generate_text(prompt, max_new_tokens=MAX_NEW_TOKENS_HE, do_sample=False)
        completion = postprocess_humaneval_completion(out)

        samples.append({
            "task_id": row["task_id"],
            "completion": completion,
        })

        print(f"[HumanEval] {i+1}/{len(ds)} {row['task_id']}")

    with open("humaneval_samples.jsonl", "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    return {
        "num_samples": len(samples),
        "output_file": "humaneval_samples.jsonl",
    }


# =====================
# main
# =====================
if __name__ == "__main__":
    print("\n=== Running MMLU-Pro subset eval ===")
    mmlu = eval_mmlu_pro()

    print("\n=== Generating HumanEval samples ===")
    he = generate_humaneval_samples()

    summary = {
        "local_model": f"{BASE_ID} + {ADAPTER_ID}",
        "local_results": {
            "MMLU-Pro subset accuracy": mmlu["accuracy"],
            "MMLU-Pro subset correct": mmlu["correct"],
            "MMLU-Pro subset total": mmlu["total"],
            "HumanEval samples generated": he["num_samples"],
        },
        "public_compare": PUBLIC_COMPARE,
        "files": [
            "mmlu_pro_results.json",
            "humaneval_samples.jsonl",
        ],
    }

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))