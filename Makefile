.PHONY: setup train infer validate data200

PYTHON ?= python

setup:
	bash scripts/setup_env.sh

train:
	$(PYTHON) -u train_maid.py --config configs/maid_1p5b_stable.yaml

infer:
	$(PYTHON) -u chat_maid.py --prompt "明日の朝やるべきことを3つだけ教えて" --preset sweet

validate:
	$(PYTHON) -u data/validate.py data/style_maid.jsonl

data200:
	$(PYTHON) -u data/make_style_maid_200.py --src data/style_maid_100.jsonl --dst data/style_maid_200.jsonl
