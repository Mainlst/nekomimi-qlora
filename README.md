# 日本語README | [English README](README_en.md)
# Nekomimi-QLoRA 🐾✨
ネコ耳メイド口調LoRA – RTX 3070/4070/T4 で動く軽量QLoRA実装

![demo](screenshots/sample.png)

---

## ✨ プロジェクト概要
このリポジトリは、Qwen2.5-1.5B-Instruct をベースに、
LoRA ファインチューニングで「優しいネコ耳メイド口調」を学習させた最小実装です。小規模データ（100例程度）でもキャラクター性を付与できることを示します。

---

## 🚀 セットアップ（neko-lora 環境・mamba/micromamba 推奨）

1) 取得
```bash
git clone https://github.com/Mainlst/nekomimi-qlora.git
cd nekomimi-qlora
```

2) 環境作成（micromamba/mamba → conda → venv の順で自動）
```bash
bash scripts/setup_env.sh
```

手動の例（micromambaがある場合）
```bash
micromamba create -n neko-lora -f environment.yml -y
micromamba run -n neko-lora python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision --upgrade  # 任意
```

venv を使う場合（最小）
```bash
python -m venv .venv-neko-lora
source .venv-neko-lora/bin/activate
python -m pip install -r requirements.txt
```

---

## 🧪 学習（8GB安全ラインとドライラン）

最小構成（8GB想定）:
```bash
# ドライラン（重いDL/学習をスキップ、成果物の雛形を出力）
bash scripts/train.sh configs/maid_1p5b_stable.yaml --dry-run

# 実学習
bash scripts/train.sh configs/maid_1p5b_stable.yaml
```

ポイント
- 設定は `configs/*.yaml` で管理（例: `maid_1p5b_stable.yaml`/`maid_1p5b_std.yaml`/`maid_3b_edge.yaml`）
- `train_maid.py --config <yaml>` で読み込み（seq→max_length、LoRA r/alpha/dropout、target_modules も反映）
- 既定のアダプタ保存先: `out/maid-qlora/adapter`
- ログ/メトリクス/図は「configファイルのあるディレクトリ直下」に保存（例: `configs/metrics.json`, `configs/artifacts/learning_curve.png` など）

---

## 💬 推論（プリセット mild/sweet/ultra）

プリセット付きの簡易CLI:
```bash
# sweet プリセットでプロンプトを実行
bash scripts/infer.sh "明日の朝やるべきことを3つだけ教えて" sweet

# 直接指定（任意でベース/アダプタの切替も可能）
micromamba run -n neko-lora python -u chat_maid.py \
  --prompt "短い応援を一言" --preset mild \
  --base Qwen/Qwen2.5-1.5B-Instruct \
  --adapter out/maid-qlora/adapter
```

プリセット定義: `presets/infer.json`
- mild: temperature=0.6, top_p=0.9, repetition_penalty=1.05
- sweet: 0.7, 0.9, 1.05
- ultra: 0.8, 0.92, 1.1（暴走注意）

---

## 📂 構成（主要）

- `configs/` … 学習設定（8GB最小/推奨/3Bエッジ）
- `presets/infer.json` … 推論プリセット mild/sweet/ultra
- `scripts/` … 環境準備・学習/推論のヘルパ
  - `setup_env.sh` … neko-lora 環境を micromamba/mamba/conda/venv の順で構築
  - `train.sh` / `infer.sh`
- `data/`
  - `style_maid_100.jsonl` … サンプルSFTデータ
  - `make_style_maid_200.py` … 100→200件へ拡張（[STYLE=maid] 前置、カテゴリ均等化）
  - `validate.py` … JSONL検証（件数/重複率/平均長さ）
- `eval/prompts_100.txt` … 評価用固定プロンプト（雛形）
- `Makefile` … `make setup/train/infer/validate/data200`
- `results/metrics/` / `reports/` / `assets/screenshots/` … 成果物の置き場
- `train_maid.py` … QLoRA学習（--config対応、dry-run対応）
- `chat_maid.py` … LoRA推論（--preset/--prompt 対応）
- `examples/infer_minimal.py` … 最小推論サンプル

---

## 実験運用ガイド 🧪（簡易）

推奨ブランチ運用
- `main`（安定） / `exp/*`（実験） / `demo/*`（作品）

成果物の置き場
- `reports/`（表・所感）
- `results/metrics/`（JSONログ）
- `assets/screenshots/`（UIや比較のスクショ）

補足（旧ワークフロー）
- `scripts/new_exp.sh` と `scripts/run_exp.sh` も併用可能です（experiments/配下に雛形生成）。

### Windows（PowerShell）

```powershell
./scripts/run_exp.ps1 -Config "experiments/2025-08-30-lora-qlora/config.yaml"
```

### 開発の足場（任意）

- プリコミット
  - 設定: `.pre-commit-config.yaml`
  - 有効化:
    ```bash
    pip install pre-commit
    pre-commit install
    ```
- 最小CI（GitHub Actions）
  - ワークフロー: `.github/workflows/ci.yml`
  - 目的: 依存インストールとスタイルチェックのスモーク

### 運用ルール（推奨）

- ブランチ: `main`（安定）, `exp/<日付>-<短名>`, `demo/<名前>`, `feat/<目的>`, `fix/<内容>`
- コミット例: `exp: run qlora on 3e-4 with cosine schedule`
- 節目はタグ/Release、`CHANGELOG.md`更新

---

## 🔮 応用アイデア

* 他のキャラ口調 LoRA（執事、関西弁、英語版）
* ChatBot や Discord Bot への組み込み
* NPC 対話やゲーム AI のパーソナライズ

---

## 📜 License

MIT License

---

## 📖 Citation

```bibtex
@software{maid_qlora,
	author = {Your Name},
	title = {Maid-QLoRA: ネコ耳メイド口調LoRA},
	year = {2025},
	url = {https://github.com/yourname/nekomimi-qlora},
}
```
