import pandas as pd
import torch
import random
from BERT.BERT_model import BERT
from dataloader.dataloader import Dataloader
from transformers import AdamW
from tokenizer.wordPieceTokenizer import WordPieceTokenizer, mask_tokens
from helper import prepare_tensor, create_sentence_pair  # Import helper functions


# Afegeixo chequeo de GPU (si ho probem a collab)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    
   # Movem model a GPU si está disponible
    model.to(device)  
    
    optimizer = AdamW(model.parameters(), lr=5e-5)


'''
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
'''


# Número de passos per l'accumulació de gradients
accumulation_steps = 4

# Training loop
model.train()  #engeguem modo training
for epoch in range(3):  # Epochs
    for step, (title, text, sentences) in enumerate(dataloader):  # Iterar sobre el total de batches del data loader
        for i in range(0, len(sentences), 2):  # Per cada batch, iterar sobre les frases
            if i + 1 < len(sentences):  
                #definim les frases
                sentence1, sentence2, nsp_label = create_sentence_pair(sentences, i)

                # Tokenitzem
                token_ids_sentence1 = tokenizer.encode(sentence1)  
                token_ids_sentence2 = tokenizer.encode(sentence2)  
                input_ids, attention_mask, segment_ids = tokenizer.add_special_tokens(
                    token_ids_sentence1, token_ids_sentence2, max_length=512
                )  # Afegim tokens especials i attention mask

                # Emmascarem tokens per MLM
                masked_input_ids, labels = mask_tokens(input_ids, tokenizer)

                # Prepare tensors by adding batch dimension (unsqueeze) and moving to device
                input_ids = prepare_tensor(input_ids, device)  
                attention_mask = prepare_tensor(attention_mask, device)  
                segment_ids = prepare_tensor(segment_ids, device)  
                masked_lm_labels = prepare_tensor(labels, device)  
                next_sentence_labels = prepare_tensor([nsp_label], device)  

                # Forward pass 
                outputs = model(input_ids, attention_mask, segment_ids, masked_lm_labels, next_sentence_labels)
                loss = outputs[0]

                # Normalitzem tenint en compte l'acumulació de gradients
                loss = loss / accumulation_steps
                loss.backward()  # Backpropaguem

                # Optimizer step cada X accumulation Steps
                if (step * len(sentences) + i // 2 + 1) % accumulation_steps == 0:
                    optimizer.step()  
                    optimizer.zero_grad()  # Resetegem gradients després

                    # Print loss for monitoring
                    print(f"Epoch {epoch}, Step {step * len(sentences) + i // 2}, Loss {loss.item()}")


    # Final step to update weights after the last accumulation step
    if (step * len(sentences) + i // 2 + 1) % accumulation_steps != 0:
        optimizer.step()  # Final optimizer step to update model parameters
        optimizer.zero_grad()
