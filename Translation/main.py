import os 
import torch
from transformers import AutoModelForSeq2SeqLM
from IndicTransTokenizer import IndicProcessor, IndicTransTokenizer
import time

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
    def __init__(self, direction: str = "indic-indic", checkpoint_dir: str = "ai4bharat/indictrans2-indic-indic-1B"):
        self.direction = direction
        self.checkpoint_dir = checkpoint_dir
        self.tokenizer = IndicTransTokenizer(direction=self.direction)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.checkpoint_dir, trust_remote_code=True, output_attentions=True)
        self.ip = IndicProcessor(inference=True)
        self.BATCH_SIZE = 10
        self.DEVICE = torch.device('cpu')
        self.model.to(self.DEVICE)
        self.model.eval()
        self.warmup()
        
    def pre_print(self, print_str: str):
        print("=================================================")
        print(print_str)
        print("=================================================")

    def preprocess_input(self, sentences, src_lang, tgt_lang):
        preprocessed = self.ip.preprocess_batch(sentences, src_lang=src_lang, tgt_lang=tgt_lang)
        inputs = self.tokenizer(preprocessed, src=True, truncation=True, padding="longest", return_tensors="pt", return_attention_mask=True).to(self.DEVICE)
        return inputs

    def translate(self, inputs, tgt_lang):
        with torch.no_grad():
            generated_tokens = self.model.generate(**inputs, use_cache=True, min_length=0, max_length=256, num_beams=5, num_return_sequences=1)

        decoded_tokens = self.tokenizer.batch_decode(generated_tokens.detach().cpu().tolist(), src=False)
        
        translation = self.ip.postprocess_batch(decoded_tokens, lang=tgt_lang)
        return translation

    def batch_translate(self, input_sentences, src_lang = "hin_Deva", tgt_lang = "ben_Beng"):
        translations = []
        for i in range(0, len(input_sentences), self.BATCH_SIZE):
            batch = input_sentences[i:i+self.BATCH_SIZE]
            inputs = self.preprocess_input(batch, src_lang, tgt_lang)
            translated_batch = self.translate(inputs, tgt_lang)
            translations.extend(translated_batch)
            del inputs
        
        return translations

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
    translator = IndicTranslator("indic-indic", "ai4bharat/indictrans2-indic-indic-1B")

    src_lang, tgt_lang = "hin_Deva", "ben_Beng"         # Change your language
    hi_translations = translator.batch_translate(sample_hi_sents, src_lang, tgt_lang)

    print(f"\n{src_lang} - {tgt_lang}")
    for input_sentence, translation in zip(sample_hi_sents, hi_translations):
        print(f"{src_lang}: {input_sentence}")
        print(f"{tgt_lang}: {translation}")