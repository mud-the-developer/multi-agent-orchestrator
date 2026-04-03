from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
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

