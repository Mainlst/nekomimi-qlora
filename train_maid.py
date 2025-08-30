import torch, os, json, argparse, logging, tarfile
from datetime import datetime
from pathlib import Path

# Use non-interactive backend for headless environments
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

def load_config(path: str | None):
    if not path:
        return None
    try:
        import yaml
    except Exception:
        print("[warn] PyYAML not installed; ignoring --config and using defaults.")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default=None, help="YAML config path")
parser.add_argument("--dry-run", action="store_true", help="Do not download/train; just validate config and create placeholders")
args = parser.parse_args()

cfg = load_config(args.config) if args.config else {}

BASE_MODEL = (
    (cfg.get("model", {}) or {}).get("base")
    if cfg else None
)
if not BASE_MODEL:
    BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

TRAIN_FILE = (
    (cfg.get("dataset", {}) or {}).get("path")
    if cfg else None
)
if not TRAIN_FILE:
    TRAIN_FILE = os.environ.get("TRAIN_FILE", "data/style_maid.jsonl")

OUTPUT_DIR = os.environ.get("OUTPUT_DIR")
if not OUTPUT_DIR:
    # if provided via config, derive default under out/<date>-<expName> if path includes experiments
    if args.config:
        exp_dir = Path(args.config).parent
        # use experiments/<name>/ as base when available, else default out/maid-qlora
        if "experiments" in str(exp_dir):
            OUTPUT_DIR = str(exp_dir / "out")
        else:
            OUTPUT_DIR = "out/maid-qlora"
    else:
        OUTPUT_DIR = "out/maid-qlora"

# Determine experiment directory and prepare folders early (for logging)
exp_dir = Path(args.config).parent if args.config else Path(OUTPUT_DIR)
artifacts_dir = exp_dir / "artifacts"
logs_dir = exp_dir / "logs"
artifacts_dir.mkdir(parents=True, exist_ok=True)
logs_dir.mkdir(parents=True, exist_ok=True)

# Configure logging to both console and file
log_file = logs_dir / "train.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file, encoding="utf-8"),
    ],
)
logger = logging.getLogger("train_maid")
logger.info("Starting training with config: %s", args.config)

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

lorac = (cfg.get("model", {}) or {}).get("lora", {}) if cfg else {}
lora_cfg = LoraConfig(
    r=int(lorac.get("r", 32)),
    lora_alpha=int(lorac.get("alpha", 64)),
    lora_dropout=float(lorac.get("dropout", 0.05)),
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    task_type="CAUSAL_LM"
)

if args.dry_run:
    ds = None
else:
    ds = load_dataset("json", data_files=TRAIN_FILE, split="train")

def formatting_func(example):
    msgs = example["messages"]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    return text

trainc = (cfg.get("train", {}) or {}) if cfg else {}
sft_cfg = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=int(trainc.get("epochs", 1)),
    per_device_train_batch_size=int(trainc.get("batch_size", 1)),
    gradient_accumulation_steps=int(trainc.get("grad_accum", 12)),
    learning_rate=float(trainc.get("lr", 2e-4)),
    lr_scheduler_type=str(trainc.get("scheduler", "cosine")),
    warmup_ratio=float(trainc.get("warmup_ratio", 0.03)),
    logging_steps=int(trainc.get("logging_steps", 5)),
    save_steps=int(trainc.get("save_steps", 200)),
    save_total_limit=int(trainc.get("save_total_limit", 2)),
    bf16=(has_cuda and compute_dtype==torch.bfloat16),
    fp16=(has_cuda and compute_dtype==torch.float16),
    max_length=int(trainc.get("max_length", 384)),
    packing=bool(trainc.get("packing", True)),
    gradient_checkpointing=True,
    seed=int((cfg or {}).get("seed", 42)),
)

if args.dry_run:
    trainer = None
    train_result = None
else:
    trainer = SFTTrainer(
        model=model,
        processing_class=tok,
        peft_config=lora_cfg,
        train_dataset=ds,
        formatting_func=formatting_func,
        args=sft_cfg,
    )
    train_result = trainer.train()

save_dir = os.path.join(OUTPUT_DIR, "adapter")
os.makedirs(save_dir, exist_ok=True)
if not args.dry_run and trainer is not None:
    trainer.model.save_pretrained(save_dir)
    tok.save_pretrained(save_dir)
logger.info("Saved adapter to: %s", save_dir)

# Save lightweight metrics
metrics = {
    "train_runtime": getattr(train_result, "metrics", {}).get("train_runtime", None) if train_result else None,
    "train_samples_per_second": getattr(train_result, "metrics", {}).get("train_samples_per_second", None) if train_result else None,
    "seed": (cfg or {}).get("seed", 42),
    "base_model": BASE_MODEL,
    "train_file": TRAIN_FILE,
    "output_dir": OUTPUT_DIR,
    "timestamp": datetime.utcnow().isoformat() + "Z",
}

with open(exp_dir / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)
logger.info("Saved metrics to: %s", str(exp_dir / "metrics.json"))

# Save log history to JSON/CSV and plot learning curve
history = []
if not args.dry_run and trainer is not None and getattr(trainer, "state", None):
    history = trainer.state.log_history or []
else:
    # Minimal synthetic history for dry-run example
    history = [
        {"step": 0, "loss": 2.0},
        {"step": 10, "loss": 1.8},
        {"step": 20, "loss": 1.6},
    ]

with open(logs_dir / "history.json", "w", encoding="utf-8") as f:
    json.dump(history, f, ensure_ascii=False, indent=2)

# CSV for quick inspection
try:
    import csv
    keys = sorted({k for row in history for k in row.keys()})
    with open(logs_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in history:
            w.writerow(row)
except Exception as e:
    logger.warning("Failed to write history.csv: %s", e)

# Plot learning curve if 'loss' present
if plt is not None:
    try:
        steps = [h.get("step") for h in history if "loss" in h]
        losses = [h.get("loss") for h in history if "loss" in h]
        if steps and losses:
            plt.figure(figsize=(6, 4))
            plt.plot(steps, losses, marker="o")
            plt.xlabel("step")
            plt.ylabel("loss")
            plt.title("Training Loss")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(artifacts_dir / "learning_curve.png", dpi=150)
            logger.info("Saved learning curve to: %s", str(artifacts_dir / "learning_curve.png"))
    except Exception as e:
        logger.warning("Failed to plot learning curve: %s", e)
else:
    logger.info("matplotlib not available; skipping learning_curve.png generation")

# Archive adapter for convenient artifact snapshot
try:
    tar_path = artifacts_dir / "adapter.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(save_dir, arcname="adapter")
    logger.info("Archived adapter to: %s", str(tar_path))
except Exception as e:
    logger.warning("Failed to create adapter archive: %s", e)

# Persist resolved config snapshot
try:
    import yaml
    resolved = {
        "seed": (cfg or {}).get("seed", 42),
        "dataset": {"path": TRAIN_FILE},
        "model": {
            "base": BASE_MODEL,
            "lora": {
                "r": lora_cfg.r,
                "alpha": lora_cfg.lora_alpha,
                "dropout": lora_cfg.lora_dropout,
            },
        },
        "train": {
            "epochs": sft_cfg.num_train_epochs,
            "batch_size": sft_cfg.per_device_train_batch_size,
            "grad_accum": sft_cfg.gradient_accumulation_steps,
            "lr": sft_cfg.learning_rate,
            "scheduler": str(sft_cfg.lr_scheduler_type),
            "warmup_ratio": sft_cfg.warmup_ratio,
            "max_length": sft_cfg.max_length,
            "packing": sft_cfg.packing,
        },
        "runtime": {
            "cuda": has_cuda,
            "compute_dtype": str(compute_dtype),
            "output_dir": OUTPUT_DIR,
        },
    }
    with open(exp_dir / "config_resolved.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(resolved, f, allow_unicode=True, sort_keys=False)
    logger.info("Saved resolved config snapshot.")
except Exception as e:
    logger.warning("Failed to save resolved config: %s", e)
