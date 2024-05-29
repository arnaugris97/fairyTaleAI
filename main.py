import pandas as pd
import torch
from BERT.BERT_model import BERT
from dataloader.dataloader import Dataloader
import torch.optim as optim
from tokenizer.wordPieceTokenizer import WordPieceTokenizer, mask_tokens
from helper import prepare_tensor, create_sentence_pair  # Import helper functions

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
dataset = pd.read_csv('dataset/merged_stories_full.csv')

# Initialize dataloader
dataloader = Dataloader(dataset, 6)

# Initialize tokenizer
tokenizer = WordPieceTokenizer()
tokenizer.load('tokenizer/wordPieceVocab.json')

# Initialize BERT model
model = BERT(vocab_size=tokenizer.vocab_size, max_seq_len=512, hidden_size=768, segment_vocab_size=2, num_hidden_layers=2, num_attention_heads=2, intermediate_size=3072)
model.to(device)

# Initialize optimizer
optimizer = optim.AdamW(model.parameters(), lr=5e-7)

# Number of steps for gradient accumulation
accumulation_steps = 3

# Training loop
model.train()  # Set model to training mode
for epoch in range(3):  # Epochs
    print(f"Epoch {epoch + 1} Start")
    for step, (title, text, sentences) in enumerate(dataloader):  # Iterate over all batches in the data loader
        for i in range(0, len(sentences), 2):  # For each batch, iterate over the sentences
            if i + 1 < len(sentences):
                # Define the sentences
                sentence1, sentence2, nsp_label = create_sentence_pair(sentences, i)

                # Tokenize
                token_ids_sentence1 = tokenizer.encode(sentence1)
                token_ids_sentence2 = tokenizer.encode(sentence2)
                input_ids, attention_mask, segment_ids = tokenizer.add_special_tokens(
                    token_ids_sentence1, token_ids_sentence2, max_length=512
                )  # Add special tokens and attention mask

                # Mask tokens for MLM
                masked_input_ids, labels = mask_tokens(input_ids, tokenizer)

                # Prepare tensors and move to GPU if available
                input_ids = prepare_tensor(input_ids, device)
                attention_mask = prepare_tensor(attention_mask, device)
                segment_ids = prepare_tensor(segment_ids, device)
                masked_lm_labels = prepare_tensor(labels, device)
                next_sentence_labels = prepare_tensor([nsp_label], device)
                

                # If input_ids length is more than 512, skip the batch
                if input_ids.shape[1] > 512:
                    continue

                
                # Forward pass
                outputs = model(input_ids, attention_mask, segment_ids)
                nsp_logits, mlm_logits = outputs
                print(f'nsp_logits: {nsp_logits.shape}')
                print(f'mlm_logits: {mlm_logits.shape}')


                # Calculate MLM loss separately
                mlm_loss = torch.nn.functional.cross_entropy(mlm_logits.view(-1, mlm_logits.size(-1)), masked_lm_labels.view(-1))
                nsp_loss = torch.nn.functional.cross_entropy(nsp_logits.view(-1, 2), next_sentence_labels.view(-1))
                print("Loss components:")
                print(f"  Masked LM Loss: {mlm_loss.item()}")
                print(f"  Next Sentence Loss: {nsp_loss.item()}")
                loss = mlm_loss + nsp_loss
                print(f"  Total Loss: {loss.item()}")

                # Check for NaN values in loss components
                if torch.isnan(mlm_loss) or torch.isnan(nsp_loss):
                    print("NaN detected in loss components")
                    continue

                # Normalize considering gradient accumulation
                loss = loss / accumulation_steps
                loss.backward()  # Backpropagate

                if (step * len(sentences) + i // 2 + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()  # Reset gradients

                    # Print loss for monitoring
                    print(f"Epoch {epoch}, Step {step * len(sentences) + i // 2}, Loss: {loss.item()}")

        # Update weights at the end of training
        if (step * len(sentences) + i // 2 + 1) % accumulation_steps != 0:
            optimizer.step()  # Final optimizer step to update model parameters
            optimizer.zero_grad()
