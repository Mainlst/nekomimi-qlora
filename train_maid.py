import torch, os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
TRAIN_FILE = os.environ.get("TRAIN_FILE", "data/style_maid.jsonl")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "out/maid-qlora")

# Ensure HF caches are writable inside workspace (no home writes required)
HF_HOME = os.environ.get("HF_HOME", os.path.join(os.getcwd(), ".hf"))
HF_DATASETS_CACHE = os.environ.get("HF_DATASETS_CACHE", os.path.join(HF_HOME, "datasets"))
HF_TRANSFORMERS_CACHE = os.environ.get("TRANSFORMERS_CACHE", os.path.join(HF_HOME, "transformers"))
os.environ.setdefault("HF_HOME", HF_HOME)
os.environ.setdefault("HF_DATASETS_CACHE", HF_DATASETS_CACHE)
os.environ.setdefault("TRANSFORMERS_CACHE", HF_TRANSFORMERS_CACHE)
os.makedirs(HF_DATASETS_CACHE, exist_ok=True)
os.makedirs(HF_TRANSFORMERS_CACHE, exist_ok=True)

# 3070 は Ampere。bf16 サポートは限定的なので基本 fp16 にフォールバック
has_cuda = torch.cuda.is_available()
compute_dtype = (
    torch.bfloat16
    if (has_cuda and torch.cuda.get_device_capability(0)[0] >= 8 and torch.cuda.is_bf16_supported())
    else torch.float16
)

# Use 4-bit only when CUDA is available; fall back to full-precision on CPU
bnb_config = None
if has_cuda:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )

tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

if has_cuda:
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.gradient_checkpointing_enable()
else:
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        device_map={"": "cpu"},
    )

lora_cfg = LoraConfig(
    r=32, lora_alpha=64, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    task_type="CAUSAL_LM"
)

ds = load_dataset("json", data_files=TRAIN_FILE, split="train")

def formatting_func(example):
    msgs = example["messages"]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    return text

sft_cfg = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=12,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=5,
    save_steps=200,
    save_total_limit=2,
    bf16=(has_cuda and compute_dtype==torch.bfloat16),
    fp16=(has_cuda and compute_dtype==torch.float16),
    max_length=384,
    packing=True,
    gradient_checkpointing=True,
    seed=42,
)

trainer = SFTTrainer(
    model=model,
    processing_class=tok,
    peft_config=lora_cfg,
    train_dataset=ds,
    formatting_func=formatting_func,
    args=sft_cfg,
)

trainer.train()

save_dir = os.path.join(OUTPUT_DIR, "adapter")
trainer.model.save_pretrained(save_dir)
tok.save_pretrained(save_dir)
print("Saved adapter to:", save_dir)
