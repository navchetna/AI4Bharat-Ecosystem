from text_preprocess_for_inference import (
    TTSDurAlignPreprocessor,
    CharTextPreprocessor,
    TTSPreprocessor,
)
from espnet2.bin.tts_inference import Text2Speech
from scipy.io.wavfile import write
from viztracer import VizTracer
import json
import torch
import yaml
import sys
from utilities import SAMPLING_RATE, WARMUP_PARAGRAPHS
from datetime import datetime
import os
import time
import numpy as np

sys.path.append(os.getenv("HIFIGAN_PATH", f"hifigan"))
from hifigan.env import AttrDict
from hifigan.models import Generator
from hifigan.meldataset import MAX_WAV_VALUE
from typing import List

import nltk

nltk.download("averaged_perceptron_tagger_eng")

if torch.cuda.is_available():
    device = "cuda"
elif torch.xpu.is_available():
    device = "xpu"
else:
    device = "cpu"

print(f"Using device: {device}")


def load_hifigan_vocoder(
    language: str, gender: str, device: str, dtype: str = "float32"
):
    """
    Loads HiFi-GAN vocoder configuration file and generator model.
    """
    vocoder_config = f"vocoder/{gender}/{language}/config.json"
    vocoder_generator = f"vocoder/{gender}/{language}/generator"

    if not os.path.exists(vocoder_config) or not os.path.exists(vocoder_generator):
        raise FileNotFoundError(
            f"Vocoder files not found. Expected config: {vocoder_config}, generator: {vocoder_generator}"
        )

    with open(vocoder_config, "r") as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    torch.manual_seed(h.seed)
    device = torch.device(device)
    generator = Generator(h).to(device)
    state_dict_g = torch.load(vocoder_generator, map_location=device)
    generator.load_state_dict(state_dict_g["generator"])
    generator.eval()
    generator.remove_weight_norm()

    if dtype == "bfloat16":
        generator = generator.to(torch.bfloat16)

    return generator


def load_fastspeech2_model(
    language: str, gender: str, device: str, dtype: str = "float32"
):
    """
    Loads FastSpeech2 model and updates its configuration with absolute paths.
    """
    config_path = f"{language}/{gender}/model/config.yaml"
    tts_model_path = f"{language}/{gender}/model/model.pth"

    if not os.path.exists(config_path) or not os.path.exists(tts_model_path):
        raise FileNotFoundError(
            f"FastSpeech2 model files not found. Expected config: {config_path}, model: {tts_model_path}"
        )

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    current_working_directory = os.getcwd()
    feat_rel_path = "model/feats_stats.npz"
    pitch_rel_path = "model/pitch_stats.npz"
    energy_rel_path = "model/energy_stats.npz"

    feat_path = os.path.join(current_working_directory, language, gender, feat_rel_path)
    pitch_path = os.path.join(
        current_working_directory, language, gender, pitch_rel_path
    )
    energy_path = os.path.join(
        current_working_directory, language, gender, energy_rel_path
    )

    config["normalize_conf"]["stats_file"] = feat_path
    config["pitch_normalize_conf"]["stats_file"] = pitch_path
    config["energy_normalize_conf"]["stats_file"] = energy_path

    # Temporarily write the modified config to a new file or use a BytesIO object if preferred
    with open(config_path, "w") as file:
        yaml.dump(config, file)

    model = Text2Speech(
        train_config=config_path,
        model_file=tts_model_path,
        device=device,
        vocoder_config=None,
        vocoder_file=None,
    )
    model.vocoder = None

    if dtype == "bfloat16":
        model.model = model.model.to(torch.bfloat16)
    return model


def split_into_chunks(text: str, words_per_chunk: int = 100):
    """Splits text into chunks of specified words_per_chunk."""
    words = text.split()
    chunks = [
        words[i : i + words_per_chunk] for i in range(0, len(words), words_per_chunk)
    ]
    return [" ".join(chunk) for chunk in chunks]


class Text2SpeechApp:
    def __init__(
        self,
        language: str,
        batch_size: str = 1,
        alpha: float = 1,
        dtype: str = "bfloat16",
    ):
        self.alpha = alpha
        self.lang = language
        self.dtype = dtype
        self.vocoder_model = {}
        self.fastspeech2_model = {}
        self.supported_genders = []
        assert dtype in [
            "bfloat16",
            "float32",
        ], f"Unsupported dtype: {dtype}. Must be 'bfloat16' or 'float32'"
        self.preprocessor = TTSDurAlignPreprocessor()

        genders = ["male", "female"]
        for gender in genders:
            try:
                self.vocoder_model[gender] = load_hifigan_vocoder(
                    f"{language}_latest", gender, device, self.dtype
                )
                print(f"Loaded HiFi-GAN vocoder for {language}-{gender}")

                self.fastspeech2_model[gender] = load_fastspeech2_model(
                    f"{language}_latest", gender, device, self.dtype
                )
                print(f"Loaded FastSpeech2 model for {language}-{gender}")
                self.supported_genders.append(gender)
            except FileNotFoundError as e:
                print(
                    f"Error loading model for {language}-{gender}: {e}. This model key will not be available."
                )
            except Exception as e:
                print(
                    f"An unexpected error occurred while loading model for {language}-{gender}: {e}. This model key will not be available."
                )
        self.warmup()

    def pre_print(self, print_str: str):
        print("=================================================")
        print(print_str)
        print("=================================================")

    def warmup(self):
        self.pre_print("TTS Warming up!")

        lang = self.lang.lower()
        text = WARMUP_PARAGRAPHS.get(lang)

        if not text:
            print(f"No warmup paragraph available for language: {lang}")
            return

        # Ensure warmup output directory exists
        output_dir = "./warmup_outputs"
        os.makedirs(output_dir, exist_ok=True)

        print(f"Running warmup for language: {lang}")
        print(f"Warmup text length: {len(text.split())} words")

        total_start_time = time.time()

        for gender in ["male", "female"]:
            if gender not in self.fastspeech2_model:
                print(f"Skipping warmup for {gender} - model not loaded.")
                continue

            print(f"Starting warmup for {lang}-{gender}")
            gender_start_time = time.time()
            for i in range(2):  # Run twice; adjust as needed
                print(f"Warmup iteration {i + 1} for {gender}")
                time_taken, _ = self.warmup_specific_run(texts=text, speaker_gender=gender, save_file=True, out_dir=output_dir)
                print(f"Iteration {i + 1} for {gender} completed in {time_taken:.2f} seconds")
            gender_total_time = time.time() - gender_start_time
            print(f"Total warmup time for {gender}: {gender_total_time:.2f} seconds")

        total_time = time.time() - total_start_time
        print(f"Total TTS warmup completed in {total_time:.2f} seconds")
        self.pre_print("TTS Warming finished!")

    def generate_audio_bytes(self, text: str, speaker_gender="male"):
        # Preprocess text
        preprocessed_text, _ = self.preprocessor.preprocess(
            text, self.lang, speaker_gender
        )
        preprocessed_text = " ".join(preprocessed_text)

        with torch.no_grad():
            # Generate mel-spectrograms
            specific_model = self.fastspeech2_model[speaker_gender]
            mel = specific_model(preprocessed_text, decode_conf={"alpha": self.alpha})
            final_mel = mel["feat_gen_denorm"].T.unsqueeze(0) * 2.3262
            # Convert mel-spectrograms to raw audio waveforms
            y_g_hat = self.vocoder_model[speaker_gender](final_mel)
            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
        return audio

    def warmup_specific_run(
        self,
        texts: List[str] | str,
        speaker_gender="male",
        save_file: bool = False,
        out_dir: str = "./outputs",
        output_file: str = "result",
    ):
        start = time.time()
        audio_arr = []
        result_chunks = split_into_chunks(texts)
        print(f"Total chunks to process: {len(result_chunks)}")
        for chunk_text in result_chunks:
            audio = self.generate_audio_bytes(chunk_text)
            audio_arr.append(audio)
        final_audio = torch.cat(audio_arr, dim=0)

        if save_file:
            self.save_to_file(
                audio=final_audio, folder_path=out_dir, file_name=output_file
            )
        time_taken = time.time() - start

        return time_taken, output_file

    def save_to_file(self, audio, folder_path, file_name):
        if audio.dtype == torch.bfloat16:
            audio = audio.to(torch.float32)
        audio = audio.cpu().numpy().astype("int16")

        # Save as numpy file
        os.makedirs(f"{folder_path}/numpy_files", exist_ok=True)
        np_output_file = f"{folder_path}/numpy_files/{file_name}.npy"
        np.save(np_output_file, audio)

        # Save as wav file
        os.makedirs(f"{folder_path}/audio_files", exist_ok=True)
        audio_file_path = f"{folder_path}/audio_files/{file_name}.wav"
        with open(audio_file_path, "wb") as f:
            write(f, SAMPLING_RATE, audio)
        print(f"Audio saved to {audio_file_path}")

    def generate_batched_audio_bytes(self, texts: list, batch_size=1, speaker_gender="male"):
        print(f"\nTotal T2S to be done: {len(texts)}\n")
        preprocessed_texts= []
        for text in texts:
            preprocessed_text, _ = self.preprocessor.preprocess(
                text, self.lang, speaker_gender)
            preprocessed_texts.append(" ".join(preprocessed_text))
        batched_audios = []
        batched_texts = [preprocessed_texts[i:i+batch_size] for i in range(0, len(preprocessed_texts), batch_size)]

        for texts in batched_texts:
            with torch.no_grad():
                specific_model = self.fastspeech2_model[speaker_gender]
                if batch_size == 1:
                    # Single text processing
                    out = specific_model(texts[0], decode_conf={"alpha": self.alpha})
                    mel = out["feat_gen_denorm"].T.unsqueeze(0) * 2.3262
                    y_g_hat = self.vocoder_model[speaker_gender](mel)
                    audio = y_g_hat.squeeze()
                    audio = audio * MAX_WAV_VALUE
                    batched_audios.append(audio)
                else:
                    # Batch processing
                    out_batched = specific_model.batch_call(texts)
                    # Prepare mel-spectrograms for batching
                    mel_spectrograms = []
                    mel_lengths = []
                    for i in range(len(out_batched["feat_gen_denorm"])):
                        mel = out_batched["feat_gen_denorm"][i].T * 2.3262  # Shape: (160, time_steps)
                        mel_spectrograms.append(mel)
                        mel_lengths.append(mel.shape[1])
                    # Pad mel-spectrograms to the same length for batching
                    max_length = max(mel_lengths)
                    padded_mels = []
                    masks = []
                    for mel in mel_spectrograms:
                        if mel.shape[1] < max_length:
                            # Pad with zeros on the time dimension
                            padding = torch.zeros((mel.shape[0], max_length - mel.shape[1]), device=mel.device)
                            mel = torch.cat([mel, padding], dim=1)
                        masks.append(mel.shape[1])
                        padded_mels.append(mel)
                    # Stack into a batch: (batch_size, 160, max_length)
                    mels = torch.stack(padded_mels, dim=0)
                    # Convert mel-spectrograms to raw audio waveforms in one batch
                    y_g_hat = self.vocoder_model[speaker_gender](mels, mel_lengths)
                    # Process each audio output and trim to original length
                    for i, original_length in enumerate(mel_lengths):
                        audio = y_g_hat[i].squeeze()
                        # Trim audio to match the original mel length (approximate audio length)
                        # Each mel frame corresponds to hop_size samples (typically 1024)
                        audio_length = original_length * 1024  # Adjust hop_size if different
                        audio = audio[:audio_length]
                        audio = audio * MAX_WAV_VALUE
                        batched_audios.append(audio)
                
        return batched_audios
    
    def evaluate_performance(
        self,
        texts: List[str],
        batch_size: int = 1,
        save_file: bool = False,
        out_dir: str = "./outputs",
    ):
        audios =  self.generate_batched_audio_bytes(texts, batch_size=batch_size)
        if save_file:
            for idx, audio in enumerate(audios):
                self.save_to_file(
                    audio=audio,
                    folder_path=out_dir,
                    file_name=f"result_{idx}",
                )



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Text to Speech benchmarking")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for TTS inference")
    parser.add_argument("--language", type=str, default="hindi", help="Language for TTS")
    parser.add_argument("--alpha", type=float, default=1.0, help="Alpha value for FastSpeech2 decoding")
    parser.add_argument("--dtype", type=str, default="float32", help="Data type for model inference")
    parser.add_argument("--save_file", action="store_true", help="Save audio files to disk")
    args = parser.parse_args()
    language = args.language
    alpha = args.alpha
    tts = Text2SpeechApp(alpha=alpha, language=language, dtype=args.dtype)
    st = time.perf_counter()
    texts = [
        "जीवन में सफलता पाने के लिए केवल सपने देखना ही नहीं, बल्कि उन्हें पूरा करने के लिए निरंतर प्रयास और आत्मविश्वास भी ज़रूरी होता है।",
        "कठिन परिस्थितियाँ हमें तोड़ने नहीं आतीं, बल्कि हमें मज़बूत बनाकर जीवन के असली अर्थ से परिचित कराती हैं।",
        "जो व्यक्ति समय का सम्मान करता है, समय भी उसके जीवन को सही दिशा और उज्ज्वल भविष्य देता है।",
        "सकारात्मक सोच और सही दृष्टिकोण के साथ किया गया हर छोटा प्रयास भी एक दिन बड़ी उपलब्धि में बदल जाता है।",
        "जब हम निस्वार्थ भाव से दूसरों की मदद करते हैं, तब हमारे अपने जीवन में भी शांति और संतुलन अपने आप आ जाता है।",
        "मैं आज सुबह से बहुत व्यस्त हूँ, इसलिए थोड़ा टाइम नहीं मिल रहा।",
        "भारत एक विविधताओं से भरा देश है, जहाँ विभिन्न भाषाएँ, संस्कृतियाँ और परंपराएँ मिलकर एक अनोखा संगम बनाती हैं।",
        "प्रकृति की सुंदरता को देखकर मन आनंदित हो जाता है और जीवन में एक नई ऊर्जा का संचार होता है।",
        "शिक्षा ही वह माध्यम है जो व्यक्ति को अज्ञानता के अंधकार से निकालकर ज्ञान के प्रकाश की ओर ले जाती है।",
        "सपनों को साकार करने के लिए मेहनत और समर्पण की आवश्यकता होती है, केवल इच्छाशक्ति ही पर्याप्त नहीं होती।",
        "समय का सदुपयोग करना जीवन में सफलता पाने की कुंजी है।",
        "सच्ची मित्रता जीवन का सबसे बड़ा उपहार है, जो कठिन समय में भी साथ निभाती है।",
        "स्वास्थ्य ही सबसे बड़ा धन है, इसलिए अपनी सेहत का ध्यान रखना अत्यंत आवश्यक है।",
        "परिवर्तन जीवन का नियम है, इसे स्वीकार करना और उसके अनुसार खुद को ढालना ही बुद्धिमानी है।",
        "संगीत आत्मा का आहार है, जो हमारे मन को शांति और आनंद प्रदान करता है।",
        "प्रत्येक दिन एक नया अवसर लेकर आता है, इसे सकारात्मक दृष्टिकोण से देखना चाहिए।",
        "किसी भी कार्य में सफलता पाने के लिए धैर्य और बुद्धिमानी की आवश्यकता होती है।",
        "सहयोग और एकता से ही समाज में प्रगति संभव है।",
        "प्रकृति के साथ सामंजस्य स्थापित करके ही हम एक स्वस्थ और खुशहाल जीवन जी सकते हैं।",
        "सपनों को पूरा करने के लिए साहस और आत्मविश्वास की आवश्यकता होती है।",
        "जीवन में चुनौतियाँ आती हैं, लेकिन उन्हें पार करके ही हम मजबूत बनते हैं।",
        "सच्चाई और ईमानदारी जीवन के सबसे महत्वपूर्ण मूल्य में आवश्यकता इच्छाशक्ति होती है।",
        "सकारात्मक सोच से ही हम जीवन में सफलता और खुशहाली प्राप्त कर सकते हैं।",
        "जब हम दूसरों की मदद करते हैं, तो हमारे अपने जीवन में भी खुशियाँ आती हैं।",
        "प्रकृति की सुंदरता को देखकर मन आनंदित हो जाता है और जीवन में एक नई ऊर्जा का संचार होता है।",
        "शिक्षा ही वह माध्यम है जो व्यक्ति को अज्ञानता के अंधकार से निकालकर ज्ञान के प्रकाश की ओर ले जाती है।",
        "सपनों को साकार करने के लिए मेहनत और समर्पण की आवश्यकता होती है, केवल इच्छाशक्ति ही पर्याप्त नहीं होती।",
        "समय का सदुपयोग करना जीवन में सफलता पाने की कुंजी है।",
        "सच्ची मित्रता जीवन का सबसे बड़ा उपहार है, जो कठिन समय में भी साथ निभाती है।",
        "स्वास्थ्य ही सबसे बड़ा धन है, इसलिए अपनी सेहत का ध्यान रखना अत्यंत आवश्यक है।",
        "परिवर्तन जीवन का नियम है, इसे स्वीकार करना और उसके अनुसार खुद को ढालना ही बुद्धिमानी है।",
        "संगीत आत्मा का आहार है, जो हमारे मन को शांति और आनंद प्रदान करता है।",
    ]

    tts.evaluate_performance(texts, batch_size=args.batch_size, save_file=args.save_file)
    et = time.perf_counter()
    print(
        f"Total time for evaluating {len(texts)} sentences: {(et - st)*1000:.0f} milliseconds"
    )
    print(f"Average time per sentence: {(et - st)/len(texts)*1000:.0f} milliseconds")
