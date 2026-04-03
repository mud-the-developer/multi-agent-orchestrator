from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset
import torch

base_id = "google/gemma-4-31B-it"
adapter_id = "kai-os/gemma4-opus-reasoning-adapter-v1"

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(base_id)
base = AutoModelForCausalLM.from_pretrained(
    base_id,
    device_map="auto",
    quantization_config=bnb,
    torch_dtype=torch.bfloat16,
)
model = PeftModel.from_pretrained(base, adapter_id)
model.eval()

# 예시 데이터셋: GSM8K
ds = load_dataset("openai/gsm8k", "main", split="test[:5]")

for i, row in enumerate(ds):
    question = row["question"]

    messages = [
        {"role": "user", "content": question}
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = text[len(prompt):].strip()

    print(f"\n=== Example {i+1} ===")
    print("Q:", question)
    print("GT:", row["answer"])
    print("Model:", answer)
