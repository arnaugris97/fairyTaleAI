from BERT.BERT_model import BERT
from dataloader.dataloader import Dataloader
from transformers import AdamW
from tokenizer.wordPieceTokenizer import WordPieceTokenizer, mask_tokens
import pandas as pd

dataset = pd.read_csv('dataset/merged_stories(1).csv')

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
    input_ids, attention_mask, segment_ids = tokenizer.add_special_tokens(token_ids_sentence1, token_ids_sentence2, max_length=512)
    print(f'token_ids:', input_ids)
    print(f'attention_mask:', attention_mask)
    print(f'token_type_ids', segment_ids)
    masked_input_ids, labels = mask_tokens(input_ids, tokenizer)
    print(f'masked_input_ids:', masked_input_ids)
    print(f'labels:', labels)

   # Initialize the model
    model = BERT(vocab_size=30522, max_seq_len=512, hidden_size=768, segment_vocab_size=2, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Training loop
    model.train()
    for epoch in range(3):  # Number of epochs
         # Move to GPU if available
        input_ids, attention_mask, segment_ids, masked_lm_labels, next_sentence_labels
        
        optimizer.zero_grad()
        loss, mlm_logits, nsp_logits = model(input_ids, attention_mask, segment_ids, masked_lm_labels, next_sentence_labels)
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch}, Loss {loss.item()}")