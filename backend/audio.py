from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import librosa
import torch
import numpy as np
import os

class Audio:
    def __init__(self):
        self.model_id = "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"
        self.model = AutoModelForAudioClassification.from_pretrained(self.model_id)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_id, do_normalize=True)
        self.id2label = self.model.config.id2label
        print("Audio model and feature extractor are loaded.")
    
    def preprocess(self, audio_path, max_duration=30.0):
        audio_array, sampling_rate = librosa.load(audio_path, sr=self.feature_extractor.sampling_rate)
        max_length = int(self.feature_extractor.sampling_rate * max_duration)
        if len(audio_array) > max_length:
            audio_array = audio_array[:max_length]
        else:
            audio_array = np.pad(audio_array, (0, max_length - len(audio_array)))
        
        inputs = self.feature_extractor(
            audio_array,
            sampling_rate=self.feature_extractor.sampling_rate,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        return inputs

    def predict(self, audio_path, max_duration=30.0):
        inputs = self.preprocess(audio_path, max_duration)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        predicted_id = torch.argmax(logits, dim=-1).item()
        predicted_label = self.id2label[predicted_id]
        return logits[0][predicted_id].item(), predicted_label

    async def execute(self, file):
        contents = await file.read()
        temp_filename = "temp_audio.webm"
        with open(temp_filename, "wb") as f:
            f.write(contents)
        
        score, label = self.predict(temp_filename)
        
        os.remove(temp_filename)
        return {"score": score, "label": label}
