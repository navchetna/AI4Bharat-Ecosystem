import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import time

device = "cpu"

model = ParlerTTSForConditionalGeneration.from_pretrained("ai4bharat/indic-parler-tts").to(device)
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)

prompt = "नमस्ते! संगीत की तरह जीवन भी खूबसूरत होता है, बस इसे सही ताल में जीना आना चाहिए."
description = "A Indian male voice with deep tone"

st = time.perf_counter()
description_input_ids = description_tokenizer(description, return_tensors="pt").to(device)
prompt_input_ids = tokenizer(prompt, return_tensors="pt").to(device)

generation = model.generate(input_ids=description_input_ids.input_ids, attention_mask=description_input_ids.attention_mask, prompt_input_ids=prompt_input_ids.input_ids, prompt_attention_mask=prompt_input_ids.attention_mask)
audio_arr = generation.cpu().numpy().squeeze()
print("Total time for generation: ", time.perf_counter() - st)
sf.write("indic_tts_out.wav", audio_arr, model.config.sampling_rate)
