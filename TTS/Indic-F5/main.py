import torch
import time
import numpy as np
import soundfile as sf
from transformers import AutoModel

# Load INF5 from Hugging Face
repo_id = "ai4bharat/IndicF5"
model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)


class TTS:
    def __init__(self, model_name: str = "ai4bharat/IndicF5"):
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    
    def generate_audio(self, text: str, ref_audio_path: str, ref_text: str):
        st = time.perf_counter()
        audio = self.model(
            text,
            ref_audio_path=ref_audio_path,
            ref_text=ref_text
        )
        total_time = time.perf_counter() - st
        return audio, total_time
    
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TTS using IndicF5")
    parser.add_argument("--text", type=str, default="नमस्ते! संगीत की तरह जीवन भी खूबसूरत होता है, बस इसे सही ताल में जीना आना चाहिए.", help="Input text to synthesize")
    parser.add_argument("--ref_audio", type=str, default="sample_voice.wav", help="Reference audio path")
    parser.add_argument("--ref_text", type=str, default="नमस्ते मै अभिलाष बोल रहा हूँ  क्या आप मुझे जानते है ? मै हिन्दी में बात करने में सक्षम हूँ ")
    parser.add_argument("--warmup_text", type=str, default="ये एरर इसलिए आ रहा है क्योंकि गितहब बिना लॉगिन के किसी को भी कोड पुश करने नहीं देता", help="Text for warmup iterations")
    parser.add_argument("--output_path", type=str, default=None, help="Output path to save synthesized audio")
    parser.add_argument("--warmup_iters", type=int, default=3, help="Number of warmup iterations")
    args = parser.parse_args()
    
    model = TTS(model_name="ai4bharat/IndicF5")
    
    for _ in range(args.warmup_iters):
        audio, _ = model.generate_audio(args.text, args.ref_audio, args.warmup_text)
        
    audio, total_time = model.generate_audio(args.text, args.ref_audio, args.ref_text)

    if args.output_path:
        # Normalize and save output
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        sf.write(args.output_path, np.array(audio, dtype=np.float32), samplerate=24000)
        print(f"Synthesized audio saved at {args.output_path}")