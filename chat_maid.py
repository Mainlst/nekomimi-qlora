import os, json, argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
ADAPTER_DIR = os.environ.get("ADAPTER_DIR", "out/maid-qlora/adapter")

compute_dtype = torch.float16
bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=compute_dtype
)

tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
mdl = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb, device_map="auto")
mdl = PeftModel.from_pretrained(mdl, ADAPTER_DIR)

def chat(user_text, system_text="あなたは優しいネコ耳メイドのAIアシスタントです。", preset: dict | None = None):
    msgs = [{"role":"system","content":system_text},
            {"role":"user","content":user_text}]
    # chat テンプレートからトークナイズ済みの input_ids を取得
    input_ids = tok.apply_chat_template(
        msgs,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(mdl.device)
    attention_mask = torch.ones_like(input_ids)
    gen_kwargs = {
        'max_new_tokens': 200,
        'do_sample': True,
        'temperature': 0.7,
        'top_p': 0.9,
        'repetition_penalty': 1.05,
    }
    if preset:
        gen_kwargs.update({k: v for k, v in preset.items() if k in ('temperature','top_p','repetition_penalty')})
    with torch.no_grad():
        out = mdl.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_kwargs,
        )
    # 生成された新規トークンのみをデコード
    gen_ids = out[0][input_ids.shape[-1]:]
    return tok.decode(gen_ids, skip_special_tokens=True).strip()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--prompt', type=str, default="明日の朝やるべきことを3つだけ教えて")
    ap.add_argument('--system', type=str, default="あなたは優しいネコ耳メイドのAIアシスタントです。")
    ap.add_argument('--preset', type=str, default='sweet', help='mild/sweet/ultra or path to JSON')
    ap.add_argument('--base', type=str, default=None)
    ap.add_argument('--adapter', type=str, default=None)
    args = ap.parse_args()

    if args.base:
        BASE = args.base
    else:
        BASE = BASE_MODEL
    if args.adapter:
        ADPT = args.adapter
    else:
        ADPT = ADAPTER_DIR

    # reload if different from defaults
    if BASE != BASE_MODEL or ADPT != ADAPTER_DIR:
        tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
        mdl_base = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map="auto")
        mdl = PeftModel.from_pretrained(mdl_base, ADPT)

    preset_obj = None
    preset_path = None
    if args.preset in ("mild","sweet","ultra"):
        try:
            with open("presets/infer.json", "r", encoding="utf-8") as f:
                allp = json.load(f)
            preset_obj = allp.get(args.preset)
        except Exception:
            preset_obj = None
    else:
        preset_path = args.preset
        try:
            with open(preset_path, "r", encoding="utf-8") as f:
                preset_obj = json.load(f)
        except Exception:
            preset_obj = None

    print(chat(args.prompt, args.system, preset_obj))
