import torch
from tokenizer.wordPieceTokenizer import WordPieceTokenizer
from BERT.BERT_model import BERT
from transformers import BertTokenizer
from inference_utils import load_model, preprocess_input, generate_embeddings, search_chromadb, generate_output
import chromadb
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import time

init_time = time.time() 

csv_file = './dataset/dataset_sentences_cleaned.csv'
df = pd.read_csv(csv_file)
sentence_list = df['Sentence'].tolist()
#print('len', len(sentence_list))

sentence_list = sentence_list[:18000] # select only the x first elements

model_path = 'Checkpoints/GoodCheckpoint.pt'
tokenizer_path = 'tokenizer/wordPieceVocab.json'

# Load the model and tokenizer
model, tokenizer = load_model(model_path, tokenizer_path)
time_load = time.time() - init_time
print('load done', time_load)


# Initialize lists to store embeddings and processed inputs
embeddings = []
processed_inputs = []
batch_size = 64

# Generate embeddings for the batch
for i in range(0, len(sentence_list), batch_size):
    batch = sentence_list[i:i + batch_size]
    
    # Preprocess batch
    inputs = [preprocess_input(text, tokenizer) for text in batch]
    input_ids = torch.cat([x['input_ids'] for x in inputs])
    attention_mask = torch.cat([x['attention_mask'] for x in inputs])
    token_type_ids = torch.cat([x['token_type_ids'] for x in inputs])

    batch_inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'token_type_ids': token_type_ids
    }
    
    with torch.no_grad():  # Disable gradient calculation for faster processing
        batch_embeddings = generate_embeddings(batch_inputs, model)
    
    # Append embeddings to the list
    embeddings.append(batch_embeddings.numpy())

# Convert list of embeddings to a numpy array
embeddings_time = time.time() - init_time
print('embeddings generated', embeddings_time)

# Flatten and detach the list of embeddings
#embeddings_flat = np.concatenate([emb.detach().numpy() for emb in embeddings])
embeddings_flat = np.vstack(embeddings)
print('embedding_flattened')

if len(sentence_list) < 30:
    # Choose a perplexity value less than the number of embeddings (e.g., 30)
    perplexity_value = len(sentence_list) -1

    # Apply t-SNE with the adjusted perplexity
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_value)
    embeddings_2d = tsne.fit_transform(embeddings_flat)
else:
    # Apply t-SNE to reduce dimensions to 2D
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings_flat)

embed2dtime = time.time() - init_time
print('embeddingd 2d', embed2dtime)

# Plotting the t-SNE embeddings
plt.figure(figsize=(10, 6))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])


plt.title("t-SNE visualization of BERT embeddings")
plt.xlabel("t-SNE component 1")
plt.ylabel("t-SNE component 2")

os.makedirs('figs_tsne', exist_ok=True)
plt.savefig('figs_tsne/tsne_visualization.png')

#plt.show()
print('done')