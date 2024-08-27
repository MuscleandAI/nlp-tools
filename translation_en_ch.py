import torch
modchoice = 'Helsinki-NLP/opus-mt-en-zh'
# -*- coding: utf-8 -*-
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained(modchoice)

model = AutoModelForSeq2SeqLM.from_pretrained(modchoice)

text = "Display of test results of Chinese-English translation model"
# Tokenize the text
batch = tokenizer.prepare_seq2seq_batch(src_texts=[text])
for k, v in batch.items():
    batch[k] = torch.tensor([w[:512] for w in v])
translation = model.generate(**batch)
result = tokenizer.batch_decode(translation, skip_special_tokens=True)
print('ori:', text)
print('result:', result)