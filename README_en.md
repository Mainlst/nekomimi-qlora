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

## 🚀 Setup

```bash
git clone https://github.com/Mainlst/nekomimi-qlora.git
cd nekomimi-qlora
pip install -r requirements.txt
```

---

## 🧪 Training

```bash
python train_maid.py
```

Output: `out/maid-qlora/adapter`

---

## 💬 Inference

```bash
python chat_maid.py
```bibtex
@software{maid_qlora,
	author = {Your Name},
	title = {Maid-QLoRA: Cat-Ear Maid Style LoRA},
	year = {2025},
	url = {https://github.com/yourname/nekomimi-qlora},
}
```
```

Example output:

```
Master〜😊 Today’s first task is “3 minutes of desk cleaning, nya!”  
```

---

## 📂 Structure

* `train_maid.py` – Training script with QLoRA
* `chat_maid.py` – Inference with the trained LoRA
* `data/style_maid.jsonl` – Sample training data (15–100 examples)
* `Dockerfile` – For reproducible environment setup
* `requirements.txt` – List of required packages
* `examples/infer_minimal.py` – Ultra-minimal inference example

### 🔁 Experiments lifecycle (quick)

- Branches: `main` (stable), `exp/<date>-<short>`, `feat/<goal>`, `fix/<what>`
- Template: copy `experiments/_template/` to `experiments/2025-08-30-lora-qlora/`
- Run: `bash scripts/run_exp.sh experiments/2025-08-30-lora-qlora/config.yaml`
- Save: results are stored under the experiment folder (`metrics.json`, `artifacts/`, `logs/`)

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