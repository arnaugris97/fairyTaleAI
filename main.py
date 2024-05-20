from dataloader.dataloader import Dataloader
from tokenizer.wordPieceTokenizer import WordPieceTokenizer, mask_tokens
import pandas as pd

dataset = pd.read_csv('tokenizer/dataset/merged_stories(1).csv')

dataloader = Dataloader(dataset, 6)

print(dataloader.__getitem__(0))
print(len(dataloader.__getitem__(0)[1]))
print(len(dataloader.__getitem__(0)[2]))

sentences = dataloader.__getitem__(0)[1]
tokenizer = WordPieceTokenizer()
tokenizer.load('tokenizer/wordPieceVocab.json')

# Do a loop that provide two senentences at each step
for i in range(0, len(sentences), 2):
    sentence1 = sentences[i]
    sentence2 = sentences[i+1]
    token_ids_sentence1 = tokenizer.encode(sentence1)
    token_ids_sentence2 = tokenizer.encode(sentence2)
    token_ids, attention_mask, token_type_ids = tokenizer.add_special_tokens(token_ids_sentence1, token_ids_sentence2, max_length=512)
    print(f'token_ids:', token_ids)
    print(f'attention_mask:', attention_mask)
    print(f'token_type_ids', token_type_ids)
    masked_input_ids, labels = mask_tokens(token_ids, tokenizer)
    print(f'masked_input_ids:', masked_input_ids)
    print(f'labels:', labels)