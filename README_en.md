# nekomimi-qlora
[English] | [日本語README](README.md)

A project for training and inference of a maid-style chatbot using QLoRA.

## Features
- Training and inference scripts for a maid-style chatbot
- Example usage scripts
- Data files for style adaptation

## Getting Started

### Requirements
Install the required Python packages:

```bash
pip install -r requirements.txt
```

### Training
To train the model, run:

```bash
python train_maid.py
```

### Inference
To run inference with the trained model:

```bash
python chat_maid.py
```

You can also use the example script:

```bash
python examples/infer_minimal.py
```

## Data
- `data/style_maid.jsonl` and `data/style_maid_100.jsonl`: Maid-style dialogue data for training and evaluation.

## Screenshots
See the `screenshots/` directory for sample outputs and UI images.

## License
See `LICENSE` for details.

## Citation
If you use this project, please cite it using the information in `CITATION.cff`.
