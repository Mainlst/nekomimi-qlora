# 日本語README | [English README](README_en.md)
# Nekomimi-QLoRA 🐾✨
ネコ耳メイド口調LoRA – RTX 3070/4070/T4 で動く軽量QLoRA実装

![demo](screenshots/sample.png)

---

## ✨ プロジェクト概要
このリポジトリは、Qwen2.5-1.5B-Instruct をベースに、
LoRA ファインチューニングで「優しいネコ耳メイド口調」を学習させた最小実装です。小規模データ（100例程度）でもキャラクター性を付与できることを示します。

---

## 🚀 セットアップ
```bash
git clone https://github.com/Mainlst/nekomimi-qlora.git
cd nekomimi-qlora
pip install -r requirements.txt
```

---

## 🧪 学習

```bash
python train_maid.py
```

出力: `out/maid-qlora/adapter`

---

## 💬 推論

```bash
python chat_maid.py
```

出力例:

```
ご主人様〜😊 今日の最初の一手は“机を拭く3分”ですにゃん！
```

---

## 📂 構成

* `train_maid.py` – QLoRA での学習スクリプト
* `chat_maid.py` – 学習済み LoRA を載せた推論
* `data/style_maid.jsonl` – サンプル学習データ（15〜100例）
* `Dockerfile` – 再現性の高い環境構築用
* `requirements.txt` – 必要パッケージ一覧
* `examples/infer_minimal.py` – 超ミニマルな推論サンプル

---

## 実験運用ガイド 🧪

このリポジトリは「実験や検証を続ける“育つレポジトリ”」を目指し、テンプレとスクリプトで反復を支援します。

### 1) 実験の始め方（テンプレから作成）

```bash
bash scripts/new_exp.sh lora-qlora
# => experiments/YYYY-MM-DD-lora-qlora/ が生成されます
```

- 編集ポイント:
  - `experiments/<日付>-<短名>/config.yaml`（データパスや学習ハイパラ）
  - `experiments/<日付>-<短名>/notes.md`（目的/仮説/セットアップ/学び）

### 2) 実行方法（本番/ドライラン）

- 本番実行（学習を実行）
```bash
bash scripts/run_exp.sh experiments/YYYY-MM-DD-lora-qlora/config.yaml
```

- 軽量テスト（--dry-run: 重い処理スキップ、成果物の雛形のみ生成）
```bash
bash scripts/run_exp.sh experiments/YYYY-MM-DD-lora-qlora/config.yaml --dry-run
```

引数はそのまま `train_maid.py` に渡されます（例: `--dry-run`）。

### 3) 成果物の場所（自動出力）

各実験フォルダ内に以下を自動保存します。

- ルート
  - `metrics.json`: 主要メトリクスの要約
  - `config_resolved.yaml`: 実行時点の解決済み設定（再現用）
- `artifacts/`
  - `learning_curve.png`: 学習曲線（matplotlib 未導入ならスキップ）
  - `adapter.tar.gz`: 学習済みアダプタ（`out/adapter/` のスナップショット）
- `logs/`
  - `train.log`: 進行ログ
  - `history.json`, `history.csv`: ログ履歴（ステップ毎の loss 等）

学習済みアダプタは通常 `experiments/<日付>-<短名>/out/adapter/` に保存されます。

### 4) Windows（PowerShell）

```powershell
./scripts/run_exp.ps1 -Config "experiments/2025-08-30-lora-qlora/config.yaml"
```

### 5) 開発の足場（任意）

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

### 6) 運用ルール（推奨）

- ブランチ: `main`（安定）, `exp/<日付>-<短名>`, `feat/<目的>`, `fix/<内容>`
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
