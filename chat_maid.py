import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_DIR = "out/maid-qlora/adapter"

compute_dtype = torch.float16
bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=compute_dtype
)

tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
mdl = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb, device_map="auto")
mdl = PeftModel.from_pretrained(mdl, ADAPTER_DIR)

def chat(user_text, system_text="あなたは優しいネコ耳メイドのAIアシスタントです。"):
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
    with torch.no_grad():
        out = mdl.generate(
            input_ids=input_ids,
        attention_mask=attention_mask,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.05,
        )
    # 生成された新規トークンのみをデコード
    gen_ids = out[0][input_ids.shape[-1]:]
    return tok.decode(gen_ids, skip_special_tokens=True).strip()

if __name__ == "__main__":
    print(chat("明日の朝やるべきことを3つだけ教えて"))
