#!/usr/bin/env python
import argparse, json, sys, re
from pathlib import Path

def iter_records(path: Path):
    """Yield (ln, obj) from either JSONL or JSON array files.
    ln is 1-based line number for JSONL; for array JSON it is the index+1.
    """
    text = path.read_text(encoding='utf-8').lstrip()
    # Detect array JSON
    if text.startswith('['):
        try:
            data = json.loads(text)
        except Exception as e:
            print(f"[ERR] JSON parse error: {e}")
            raise
        if not isinstance(data, list):
            print("[ERR] Top-level JSON must be an array when not JSONL")
            raise SystemExit(3)
        for i, obj in enumerate(data, 1):
            yield i, obj
        return
    # Fallback: JSONL line-delimited
    for ln, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            yield ln, obj
        except Exception as e:
            print(f"[ERR] L{ln}: JSON parse error: {e}")
            raise

def text_stats(messages):
    # concatenate all assistant+user contents
    txts = [m.get('content','') for m in messages if isinstance(m, dict)]
    text = '\n'.join(txts)
    toks = re.findall(r"\w+|\S", text)
    return {
        'chars': len(text),
        'tokens_like': len(toks),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('file', help='path to jsonl')
    args = ap.parse_args()
    p = Path(args.file)
    if not p.exists():
        print(f"not found: {p}")
        sys.exit(2)

    seen = set()
    N = 0
    char_sum = 0
    tok_sum = 0
    dup = 0

    for ln, obj in iter_records(p):
        if 'messages' not in obj or not isinstance(obj['messages'], list):
            print(f"[ERR] L{ln}: missing 'messages' list")
            sys.exit(3)
        key = json.dumps(obj, ensure_ascii=False, sort_keys=True)
        if key in seen:
            dup += 1
        seen.add(key)
        st = text_stats(obj['messages'])
        char_sum += st['chars']
        tok_sum += st['tokens_like']
        N += 1

    avg_chars = char_sum / N if N else 0
    avg_toks = tok_sum / N if N else 0

    print(json.dumps({
        'file': str(p), 'count': N,
        'dup_count': dup, 'dup_ratio': (dup / N) if N else 0.0,
        'avg_chars': round(avg_chars, 2), 'avg_tokens_like': round(avg_toks, 2)
    }, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
