from fastapi import FastAPI, UploadFile, File
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import librosa
import torch
import numpy as np

app = FastAPI()

# モデルと特徴量抽出器のロード
model_id = "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"
model = AutoModelForAudioClassification.from_pretrained(model_id)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, do_normalize=True)
id2label = model.config.id2label
print("Model and feature extractor are loaded.")

def preprocess_audio(audio_path, feature_extractor, max_duration=30.0):
    audio_array, sampling_rate = librosa.load(audio_path, sr=feature_extractor.sampling_rate)
    max_length = int(feature_extractor.sampling_rate * max_duration)
    if len(audio_array) > max_length:
        audio_array = audio_array[:max_length]
    else:
        audio_array = np.pad(audio_array, (0, max_length - len(audio_array)))

    inputs = feature_extractor(
        audio_array,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    return inputs

def predict_emotion(audio_path, model, feature_extractor, id2label, max_duration=30.0):
    inputs = preprocess_audio(audio_path, feature_extractor, max_duration)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_id = torch.argmax(logits, dim=-1).item()
    predicted_label = id2label[predicted_id]
    return logits[0][predicted_id].item(), predicted_label

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 送信されたファイルを一時的に保存
    contents = await file.read()
    temp_filename = "temp_audio.wav"
    with open(temp_filename, "wb") as f:
        f.write(contents)
    
    # 感情予測の実行
    score, label = predict_emotion(temp_filename, model, feature_extractor, id2label)
    
    return {"score": score, "label": label}
