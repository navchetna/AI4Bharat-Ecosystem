import io
import os
import time
import scipy.io.wavfile
from datetime import datetime

from TTS.utils.synthesizer import Synthesizer
# from src.inference import TextToSpeechEngine
from src.inference import TextToSpeechEngine

SUPPORTED_LANGUAGES = {
    # 'as' : "Assamese - অসমীয়া",
    'bn' : "Bangla - বাংলা",
    # 'brx': "Boro - बड़ो",
    # 'en' : "English (Indian accent)",
    # 'en+hi': "English+Hindi (Hinglish code-mixed)",
    # 'gu' : "Gujarati - ગુજરાતી",
    # 'hi' : "Hindi - हिंदी",
    # 'kn' : "Kannada - ಕನ್ನಡ",
    # 'ml' : "Malayalam - മലയാളം",
    # 'mni': "Manipuri - মিতৈলোন",
    # 'mr' : "Marathi - मराठी",
    # 'or' : "Oriya - ଓଡ଼ିଆ",
    # 'pa' : "Panjabi - ਪੰਜਾਬੀ",
    # 'raj': "Rajasthani - राजस्थानी",
    # 'ta' : "Tamil - தமிழ்",
    # 'te' : "Telugu - తెలుగు",
}

lang_models_path_prefix = os.getenv("MODEL_PATH_PREFIX", ".")
lang_model_path = f"{lang_models_path_prefix}/checkpoints"

class Text2Speech:
    def __init__(self, lang=None):
        self.lang = lang or os.getenv("LANG", "bn")
        self.default_sampling_rate = int(os.getenv("DEFAULT_SAMPLING_RATE", 16000))
        models = {}
        for lang in SUPPORTED_LANGUAGES:
            models[lang]  = Synthesizer(
                tts_checkpoint=f'{lang_model_path}/{lang}/fastpitch/best_model.pth',
                tts_config_path=f'{lang_model_path}/{lang}/fastpitch/config.json',
                tts_speakers_file=f'{lang_model_path}/{lang}/fastpitch/speakers.pth',
                # tts_speakers_file=None,
                tts_languages_file=None,
                vocoder_checkpoint=f'{lang_model_path}/{lang}/hifigan/best_model.pth',
                vocoder_config=f'{lang_model_path}/{lang}/hifigan/config.json',
                encoder_checkpoint="",
                encoder_config="",
                use_cuda=False,
            )
            print(f"Synthesizer loaded for {lang}.")
        self.engine = TextToSpeechEngine(models)

    def synthesize(self, text, speaker_gender="male"):
        start_time = time.time()
        raw_audio = self.engine.infer_from_text(input_text=text, lang=self.lang, speaker_name=speaker_gender)
        end_time = time.time()
        print(f"Synthesis took {end_time - start_time:.2f} seconds")
        
        byte_io = io.BytesIO()
        scipy.io.wavfile.write(byte_io, self.default_sampling_rate, raw_audio)
        
        return byte_io, (end_time - start_time)

    def save_to_file(self, byte_io, file_path):
        with open(file_path, "wb") as f:
            f.write(byte_io.read())
        print(f"Audio saved to {file_path}")
    
    def convert_and_save(self, text, speaker_gender="male", output_file_dir: str = "/outputs"):
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        output_file = f"{output_file_dir}/audio_{timestamp}.wav"

        byte_io, time_taken = self.synthesize(text, speaker_gender)
        self.save_to_file(byte_io=byte_io, file_path=output_file)
        
        return time_taken, output_file 
        
if __name__ == "__main__":
    tts = Text2Speech("bn")
    tts.convert_and_save("ছোটবেলায় আমি প্রতিদিন পার্কে যেতাম।" )