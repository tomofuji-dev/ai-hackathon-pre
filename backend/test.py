from transformers import LlavaProcessor, LlavaForConditionalGeneration
import torch
import uuid
import requests
import cv2
from PIL import Image

model_id = "llava-hf/llava-interleave-qwen-0.5b-hf"

processor = LlavaProcessor.from_pretrained(model_id)
model = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16)
print("torch.cuda.is_available() = " + str(torch.cuda.is_available()))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
