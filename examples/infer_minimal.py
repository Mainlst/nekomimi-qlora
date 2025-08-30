import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
ADAPTER_DIR = os.environ.get("ADAPTER_DIR", "out/maid-qlora/adapter")

device = "cuda" if torch.cuda.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto" if device=="cuda" else {"": device})
model = PeftModel.from_pretrained(base, ADAPTER_DIR)

messages = [
    {"role": "system", "content": "あなたは優しいネコ耳メイドです。語尾に“にゃん/にゃ〜”を添えて丁寧に話します。"},
    {"role": "user", "content": "朝の一言お願い"},
]
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
input_ids = tok(text, return_tensors="pt").to(model.device)

with torch.no_grad():
    out = model.generate(**input_ids, max_new_tokens=64, do_sample=True, temperature=0.8)

print(tok.decode(out[0], skip_special_tokens=True))
