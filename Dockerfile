# taigi_translator_service/Dockerfile
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime  
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./model /app/model
COPY main.py .

# 預先下載權重，避免 container 啟動時才拉
RUN python - <<'PY'
from transformers import AutoTokenizer, AutoModelForCausalLM
model_id = "./model"
AutoTokenizer.from_pretrained(model_id, use_fast=False)
AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, device_map="auto")
PY

ENV TAIGI_MODEL="./model"
EXPOSE 5021
CMD ["python", "main.py"]
