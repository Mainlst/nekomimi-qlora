[English] | [日本語README](README.md)
# Nekomimi-QLoRA 🐾✨

Cat-Ear Maid Style LoRA – Lightweight QLoRA implementation running on RTX 3070/4070/T4

![demo](screenshots/sample.png)

---

## ✨ Project Overview

This repository is based on **Qwen2.5-1.5B-Instruct**,
fine-tuned with LoRA to learn a “gentle cat-ear maid speaking style.”
It demonstrates that character traits can be added even with a small dataset (around 100 examples).

---

## 🚀 Setup (lora-local env; prefer mamba/micromamba)

1) Clone
```bash
git clone https://github.com/Mainlst/nekomimi-qlora.git
cd nekomimi-qlora
```

2) Create environment (auto: micromamba/mamba → conda → venv)
```bash
bash scripts/setup_env.sh
```

Manual example (micromamba)
```bash
micromamba create -n lora-local -f environment.yml -y
micromamba run -n lora-local python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision --upgrade  # optional
```

Using venv (minimal)
```bash
python -m venv .venv-lora-local
source .venv-lora-local/bin/activate
python -m pip install -r requirements.txt
```

---

## 🧪 Training (8GB-safe baseline and dry-run)

Minimal config for 8GB:
```bash
# Dry-run (skips heavy download/training; produces placeholder artifacts)
bash scripts/train.sh configs/maid_1p5b_stable.yaml --dry-run

# Actual training
bash scripts/train.sh configs/maid_1p5b_stable.yaml
```

Notes
- Settings are managed in `configs/*.yaml` (e.g., stable/std/3b_edge)
- `train_maid.py --config <yaml>` loads YAML (seq→max_length, LoRA r/alpha/dropout, target_modules)
- Adapter output: `out/maid-qlora/adapter`
- Logs/metrics/plots are saved next to the config file (e.g., `configs/metrics.json`, `configs/artifacts/learning_curve.png`)

---

## 💬 Inference (presets: mild/sweet/ultra)

Simple CLI with presets:
```bash
# Run with the sweet preset
bash scripts/infer.sh "Give me three small morning tasks" sweet

# Direct (optionally switch base/adapter)
micromamba run -n lora-local python -u chat_maid.py \
	--prompt "A one-line encouragement" --preset mild \
	--base Qwen/Qwen2.5-1.5B-Instruct \
	--adapter out/maid-qlora/adapter
```

Preset definitions: `presets/infer.json`
- mild: temperature=0.6, top_p=0.9, repetition_penalty=1.05
- sweet: 0.7, 0.9, 1.05
- ultra: 0.8, 0.92, 1.1 (may go wild)

---

## 📂 Structure (key parts)

- `configs/` … training configs (8GB minimal/standard, 3B edge)
- `presets/infer.json` … inference presets mild/sweet/ultra
- `scripts/` … helpers for env/training/inference
	- `setup_env.sh` … builds lora-local env (micromamba/mamba/conda/venv)
	- `train.sh` / `infer.sh`
- `data/`
	- `style_maid_100.jsonl` … sample SFT data
	- `make_style_maid_200.py` … expand to 200 (prepend [STYLE=maid], balance categories)
	- `validate.py` … JSONL validator (count/dup ratio/avg length)
- `eval/prompts_100.txt` … fixed evaluation prompts (skeleton)
- `Makefile` … `make setup/train/infer/validate/data200`
- `results/metrics/` / `reports/` / `assets/screenshots/` … output folders
- `train_maid.py` … QLoRA training (--config, dry-run)
- `chat_maid.py` … LoRA inference (--preset/--prompt)
- `examples/infer_minimal.py` … ultra-minimal inference

### 🔁 Experiments lifecycle (brief)
- Branches: `main` (stable), `exp/*` (experiments), `demo/*` (demos)
- Outputs: `reports/` (tables & notes), `results/metrics/` (JSON logs), `assets/screenshots/` (images)
- Legacy flow (`scripts/new_exp.sh`, `scripts/run_exp.sh`) still available

---

## 🔮 Applications / Ideas

* Other character-style LoRA (Butler, Kansai dialect, English version)
* Integration into ChatBots or Discord Bots
* NPC dialogue or personalized Game AI

---

## 📜 License

MIT License

---

## 📖 Citation

```bibtex
@software{maid_qlora,
	author = {Your Name},
	title = {Maid-QLoRA: Cat-Ear Maid Style LoRA},
	year = {2025},
	url = {https://github.com/yourname/nekomimi-qlora},
}