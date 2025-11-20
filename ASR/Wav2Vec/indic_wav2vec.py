import os
import time
import torch
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

class AudioToText:
    DEFAULT_SAMPLING_RATE = os.getenv("DEFAULT_SAMPLING_RATE", 16000)

    def __init__(self, model_id="ai4bharat/indicwav2vec-hindi"):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.cpu = torch.device('cpu')
        self.model_id = model_id
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(self.model_id).eval().to(self.device)

    def pre_print(self, log: str = "", prefix: str = ""):
        print(f"{prefix} ====================> {log}")

    def audio_to_text(self, audio_path: str):
        speech, _ = librosa.load(audio_path, sr=self.DEFAULT_SAMPLING_RATE)
        input_values = self.processor(speech, sampling_rate=16000, return_tensors='pt').input_values
        input_values = input_values.to(self.device)

        start = time.time()
        with torch.no_grad():
            logits = self.model(input_values).logits.to(self.device)
        time_taken = time.time() - start

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.decode(predicted_ids[0])

        return transcription, time_taken

if __name__ == "__main__":
    audio_to_text_instance = AudioToText()        # Replace your audio file here
    transcription, time_taken = audio_to_text_instance.audio_to_text("./samples/modi_legit.wav")
    print(transcription, time_taken)