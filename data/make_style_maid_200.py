#!/usr/bin/env python
"""
簡易ビルダー: style_maid_100.jsonl → style_maid_200.jsonl
- 既存100件をコピーして拡張（重複は軽微に許容）
- 各レコードに 'style': 'maid' と '[STYLE=maid]' をsystemへ前置
- 'category' を簡易ヒューリスティックで6カテゴリにアサイン
  {greeting, encouragement, todo, template, short, tech}
"""
import json, argparse, re
from pathlib import Path

CATS = ['greeting','encouragement','todo','template','short','tech']

KEYMAP = {
  'greeting': ['朝','おは','挨拶','はじめ','一言'],
  'encouragement': ['励ま','応援','がんば','勇気','元気'],
  'todo': ['ToDo','タスク','やること','チェック','一覧'],
  'template': ['テンプレ','雛形','フォーマット','例','書き方'],
  'short': ['短く','手短','一行','短文','ミニ'],
  'tech': ['コード','Git','Docker','推論','学習','VRAM','エラー']
}

def guess_category(text: str) -> str:
    t = text.lower()
    for cat, kws in KEYMAP.items():
        for kw in kws:
            if kw.lower() in t:
                return cat
    return 'short'


def build(src: Path, dst: Path):
    # Load either JSONL or JSON array
    src_text = src.read_text(encoding='utf-8').lstrip()
    rows = []
    if src_text.startswith('['):
        data = json.loads(src_text)
        if not isinstance(data, list):
            raise ValueError('Top-level JSON must be an array')
        rows = data
    else:
        for line in src_text.splitlines():
            if not line.strip():
                continue
            rows.append(json.loads(line))

    out = []
    # 1) 原本100件を整形
    for r in rows:
        msgs = r.get('messages', [])
        if msgs and msgs[0].get('role') == 'system':
            if '[STYLE=' not in msgs[0].get('content',''):
                msgs[0]['content'] = '[STYLE=maid]\n' + msgs[0]['content']
        user_txt = ''
        for m in msgs:
            if m.get('role')=='user':
                user_txt = m.get('content','')
                break
        cat = guess_category(user_txt)
        out.append({'messages': msgs, 'style': 'maid', 'category': cat})

    # 2) 100件を単純複製して200件化（categoryをラウンドロビンで調整）
    i = 0
    while len(out) < 200:
        base = out[i % len(out)]
        new = json.loads(json.dumps(base, ensure_ascii=False))  # deepcopy via json
        # ラベルのみ回す
        new['category'] = CATS[len(out) % len(CATS)]
        out.append(new)
        i += 1

    with dst.open('w', encoding='utf-8') as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    print(f"wrote {len(out)} lines to {dst}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='data/style_maid_100.jsonl')
    ap.add_argument('--dst', default='data/style_maid_200.jsonl')
    args = ap.parse_args()
    build(Path(args.src), Path(args.dst))
