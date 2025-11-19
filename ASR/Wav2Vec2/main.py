import sys
import torch
# import intel_extension_for_pytorch as ipex
import fairseq
import librosa
import soundfile
import torch.nn.functional as F
import torchaudio.sox_effects as ta_sox
import os
import logging
import time
import numpy as np

torch.serialization.add_safe_globals([fairseq.data.dictionary.Dictionary])


class AudioToText:
    DEFAULT_SAMPLING_RATE = int(os.getenv("DEFAULT_SAMPLING_RATE", "16000"))
    PAD_OFFSET = int(os.getenv("ASR_PAD_OFFSET", "200000"))

    def __init__(self, model_path="hindi.pt", warmup_iterations=1, sample_audio_path="./samples/hindi.wav", batch_size=5):
        self.batch_size = batch_size

        self.model, self.cfg, self.task = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_path])
        self.model = self.model[0]
        self.model.to(torch.bfloat16) 
        self.model.eval()

        self.effects = [["gain", "-n"]]

        self.token = self.task.target_dictionary
        self.warmup_audio_path = sample_audio_path
        self.warmup(warmup_iterations)
    
    def warmup(self, warmup_iters: int):
        for _ in range(warmup_iters):
            _ = self.transcribe(self.sample_audio_path)

    def transcribe(self, path, sample_rate=16000):
        audio, sr = librosa.load(path, sr=sample_rate)
        input_sample, rate = ta_sox.apply_effects_tensor(
            torch.tensor(audio[0]), sample_rate, self.effects)
        input_sample = input_sample.float()
    
        with torch.no_grad():
            input_sample = F.layer_norm(input_sample, input_sample.shape)

        logits = self.model(source=input_sample, padding_mask=None)['encoder_out']
        predicted_ids = torch.argmax(logits, axis=-1)
        predicted_ids = torch.unique_consecutive(predicted_ids.T, dim=1).tolist()

        transcriptions = []
        for ids in predicted_ids:
            transcription = self.token.string(ids)
            transcription = transcription.replace(
                ' ', "").replace('|', " ").strip()
            transcriptions.append(transcription)

        return transcriptions
    

    def audio_to_text(self, audio_path: str):
        transcriptions, time_taken = self.multi_audio_to_text(audio_paths=[
                                                              audio_path])
        return transcriptions[0], time_taken

    def transcribe_batch(self, path, sample_rate=16000):
        audio, sr = librosa.load(path, sr=sample_rate)
        input_sample, rate = ta_sox.apply_effects_tensor(
            torch.tensor(audio[0]), sample_rate, self.effects)
        input_sample = input_sample.float()
    
        with torch.no_grad():
            input_sample = F.layer_norm(input_sample, input_sample.shape)

        logits = self.model(source=input_sample, padding_mask=None)['encoder_out']
        predicted_ids = torch.argmax(logits, axis=-1)
        predicted_ids = torch.unique_consecutive(predicted_ids.T, dim=1).tolist()

        transcriptions = []
        for ids in predicted_ids:
            transcription = self.token.string(ids)
            transcription = transcription.replace(
                ' ', "").replace('|', " ").strip()
            transcriptions.append(transcription)

        return transcriptions

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Audio to Text Transcription")
    parser.add_argument("--model_path", type=str, default="hindi.pt", help="Path to the ASR model file")
    parser.add_argument("--sample_audio_path", type=str, default="./samples/hindi.wav", help="Path to a sample audio file for warmup")
    parser.add_argument("--warmup_iterations", type=int, default=3, help="Number of warmup iterations")
    args = parser.parse_args()
    
    audio_to_text = AudioToText(
        model_path=args.model_path,
        warmup_iterations=args.warmup_iterations,
        sample_audio_path=args.sample_audio_path
    )
    
    transcription, _ = audio_to_text.audio_to_text(args.sample_audio_path)
    print("Transcription:", transcription)