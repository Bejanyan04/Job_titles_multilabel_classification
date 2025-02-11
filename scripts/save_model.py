from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import os
import tarfile

from src.inference import get_model, get_tokenizer
model = get_model()
tokenizer = get_tokenizer()

save_dir = "zipped_model"
os.makedirs(save_dir, exist_ok=True)

model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

# Create tar.gz file for SageMaker
with tarfile.open(f"{save_dir}/model.tar.gz", "w:gz") as tar:
    tar.add(save_dir, arcname=".")
