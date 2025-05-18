# taigi_translator_service/main.py
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, TextGenerationPipeline, AutoModelForCausalLM
import torch, accelerate, os, traceback

MODEL_ID = os.getenv("TAIGI_MODEL", "Bohanlu/Taigi-Llama-2-Translator-7B")
DEVICE   = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE    = torch.float16 if torch.cuda.is_available() else torch.float32

app = Flask(__name__)

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
    accelerator = accelerate.Accelerator()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="cuda:0", torch_dtype=DTYPE, low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    pipe = TextGenerationPipeline(
        model=model, tokenizer=tokenizer, pad_token_id=tokenizer.pad_token_id,
        eos_token_id=[tokenizer.eos_token_id, tokenizer.pad_token_id],
        num_workers=accelerator.state.num_processes * 4
    )
    PROMPT_TPL = "[TRANS]\n{src}\n[/TRANS]\n[{tgt}]\n"
except Exception as e:
    traceback.print_exc()
    raise RuntimeError("模型載入失敗，無法啟動翻譯服務")

@app.route("/translate", methods=["POST"])
def translate():
    try:
        raw_data = request.data
        print(f"[DEBUG] Raw request data bytes: {raw_data}")
        print(f"[DEBUG] Raw request data decoded: {raw_data.decode('utf-8', errors='replace')}")
        
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"status":"error","message":"無法解析 JSON 請求"}), 400

        text = data.get("text", "").strip()
        target = data.get("target", "ZH")

        if not text:
            return jsonify({"status":"error", "message":"缺少 text"}), 400

        prompt = PROMPT_TPL.format(src=text, tgt=target)
        print(f"[DEBUG] Translate request received. Prompt:\n{prompt}")

        out = pipe(
            prompt,
            do_sample=False,
            temperature=1.0,    
            top_p=1.0,
            repetition_penalty=1.1,
            return_full_text=False,
            max_new_tokens=256
        )[0]['generated_text']

        ans = out.split("[/")[0].strip()
        print(f"[DEBUG] Translation output:\n{ans}")

        return jsonify({"status":"success", "translation": ans})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status":"error", "message": f"翻譯失敗: {str(e)}"}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status":"ok", "device": DEVICE, "model": MODEL_ID})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5021, debug=True)
