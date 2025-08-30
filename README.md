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
