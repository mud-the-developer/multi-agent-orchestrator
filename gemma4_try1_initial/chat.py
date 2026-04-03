import time
import threading
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
from peft import PeftModel

base_id = "google/gemma-4-31B-it"
adapter_id = "kai-os/gemma4-opus-reasoning-adapter-v1"

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(base_id)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

base = AutoModelForCausalLM.from_pretrained(
    base_id,
    device_map="auto",
    quantization_config=bnb,
    torch_dtype=torch.bfloat16,
)
model = PeftModel.from_pretrained(base, adapter_id)
model.eval()

history = []

while True:
    user_input = input("\nYou> ").strip()

    if user_input.lower() in {"exit", "quit"}:
        break
    if user_input.lower() == "/clear":
        history = []
        print("history cleared")
        continue
    if not user_input:
        continue

    history.append({"role": "user", "content": user_input})

    prompt = tokenizer.apply_chat_template(
        history,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    gen_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
    )

    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)

    start = time.time()
    thread.start()

    print("Model> ", end="", flush=True)
    chunks = []

    for chunk in streamer:
        print(chunk, end="", flush=True)
        chunks.append(chunk)

    thread.join()
    print()

    answer = "".join(chunks).strip()
    history.append({"role": "assistant", "content": answer})

    elapsed = time.time() - start
    gen_tokens = len(tokenizer(answer, add_special_tokens=False)["input_ids"])
    tps = gen_tokens / elapsed if elapsed > 0 else 0.0

    print(f"[{gen_tokens} tokens | {elapsed:.2f}s | {tps:.2f} tok/s]")