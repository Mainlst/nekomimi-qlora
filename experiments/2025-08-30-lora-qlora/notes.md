# Experiment Notes

- Goal: 何を確かめたい？（例：QLoRAで学習安定性の確認）
- Hypothesis: 学習率2e-4, cosineで安定化するはず
- Setup:
  - GPU: <your GPU>
  - Commit: <hash>
  - Command: `python train_maid.py --config experiments/2025-08-30-lora-qlora/config.yaml`
- Results (要点):
  - val_acc = N/A, f1 = N/A
- Learnings / Next:
  - 例）バッチを増やす / warmup を導入