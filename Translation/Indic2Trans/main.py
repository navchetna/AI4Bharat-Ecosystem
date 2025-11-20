import os 
import torch
import time

import intel_extension_for_pytorch as ipex
from IndicTransToolkit.processor import IndicProcessor
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

sample_hi_sents = [
    "जब मैं छोटा था, तो मैं हर दिन पार्क में जाता था।",
    "उसके पास कई पुरानी किताबें हैं, जो उसने अपने पूर्वजों से विरासत में पाई हैं।",
    "मैं समझ नहीं पा रहा हूँ कि अपनी समस्या कैसे हल करूँ।",
    "वह बहुत मेहनती और बुद्धिमान है, इसी कारण उसे सभी अच्छे अंक मिले हैं।",
    "हमने पिछले हफ्ते एक नई फिल्म देखी, जो बहुत प्रेरणादायक थी।",
    "अगर तुम उस समय मुझसे मिले होते, तो हम बाहर खाने चले जाते।",
    "वह अपनी बहन के साथ एक नई साड़ी खरीदने बाजार गई।",
    "राज ने मुझे बताया कि वह अगले महीने अपनी दादी के घर जा रहा है।",
    "सभी बच्चे पार्टी में मज़े कर रहे थे और बहुत सारी मिठाइयाँ खा रहे थे।",
    "मेरे दोस्त ने मुझे अपने जन्मदिन की पार्टी में आमंत्रित किया है, और मैं उसे एक उपहार दूँगा।"
]


class IndicTranslator:
    def __init__(self, direction: str = "indic-indic", checkpoint_dir: str = "ai4bharat/indictrans2-indic-indic-1B", batch_size: int = 5):
        self.direction = direction
        self.checkpoint_dir = checkpoint_dir
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_dir, trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.checkpoint_dir,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        self.ip = IndicProcessor(inference=True)
        self.BATCH_SIZE = batch_size
        self.model.eval()
        self.model = ipex.optimize(self.model, weights_prepack=False)
        self.model = torch.compile(self.model, backend="ipex",)
        
    def pre_print(self, print_str: str):
        print("=================================================")
        print(print_str)
        print("=================================================")

    def preprocess_input(self, sentences, src_lang, tgt_lang):
        preprocessed = self.ip.preprocess_batch(sentences, src_lang=src_lang, tgt_lang=tgt_lang)
        inputs = self.tokenizer(preprocessed, max_length=128, padding='max_length', return_tensors="pt", return_attention_mask=True)
        return inputs

    def profile_model(self, inputs, tgt_lang, verbose=True):
        """Profile the models."""
        times = {}
        tokens = {}
        
        t0 = time.perf_counter()

        with torch.no_grad():
            generated_tokens = self.model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=128,
                num_beams=5,
                num_return_sequences=1,
            )
        pad_token_id = self.tokenizer.pad_token_id
        times["total_time"] = time.perf_counter() - t0
        tokens['input_tokens'] = (inputs["input_ids"] != pad_token_id).sum(dim=1).tolist()
        tokens["output_tokens"] = (generated_tokens != pad_token_id).sum(dim=1).tolist()
        tokens['total_input_tokens'] = sum(tokens['input_tokens'])
        tokens['total_output_tokens'] = sum(tokens['output_tokens'])
        
        decoded_tokens = self.tokenizer.batch_decode(generated_tokens.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
        translation = self.ip.postprocess_batch(decoded_tokens, lang=tgt_lang)
        return translation, times, tokens

    def translate(self, inputs, tgt_lang):
        with torch.no_grad():
            generated_tokens = self.model.generate(**inputs, use_cache=True, min_length=0, max_length=128, num_beams=5, num_return_sequences=1)

        decoded_tokens = self.tokenizer.batch_decode(generated_tokens.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True,)
        
        translation = self.ip.postprocess_batch(decoded_tokens, lang=tgt_lang)
        return translation

    def batch_translate(self, input_sentences, src_lang="hin_Deva", tgt_lang="ben_Beng"):
        benchmark_data = {}
        for i in range(0, len(input_sentences), self.BATCH_SIZE):
            batch = input_sentences[i:i+self.BATCH_SIZE]
            inputs = self.preprocess_input(batch, src_lang, tgt_lang)
            translations, times, tokens = self.profile_model(inputs, tgt_lang)
            benchmark_data[i//self.BATCH_SIZE] = {
                "input_sentences": batch,
                "translations": translations,
                "times": times,
                "tokens": tokens
            }
            del inputs
        
        return benchmark_data

    def single_translate(self, sentence, src_lang, tgt_lang):
        start_time = time.time()
        translation = self.batch_translate([sentence], src_lang, tgt_lang)[0]
        time_taken = time.time() - start_time
        return translation, time_taken
    
    def warmup(self):
        self.pre_print("Warming up!")
        src_lang, tgt_lang = "hin_Deva", "ben_Beng"
        _ = self.batch_translate(sample_hi_sents, src_lang, tgt_lang)
        self.pre_print("Warmup finished!")

  
# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Indic2Indic Translation Benchmarking")
    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup iterations")
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size for translation")
    parser.add_argument("--src_lang", type=str, default="hin_Deva", help="Source language code")
    parser.add_argument("--tgt_lang", type=str, default="ben_Beng", help="Target language code")
    args = parser.parse_args()
    
    translator = IndicTranslator("indic-indic", "ai4bharat/indictrans2-indic-indic-1B", batch_size=args.batch_size)

    src_lang, tgt_lang = args.src_lang, args.tgt_lang         # Change your language
    
    for _ in range(args.warmup):
        _ = translator.batch_translate(sample_hi_sents, src_lang, tgt_lang)
    
    benchmark_data = translator.batch_translate(sample_hi_sents, src_lang, tgt_lang)
    # Check the "times" and "tokens" keys in benchmark_data for profiling information
    print("Benchmark Data:", benchmark_data)

