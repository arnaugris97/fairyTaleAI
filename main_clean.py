from BERT.BERT_model import BERT
from dataloader.custom_dataset import Custom_Dataset
from transformers import AdamW
from tokenizer.wordPieceTokenizer import WordPieceTokenizer, mask_tokens
import pandas as pd
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_csv = pd.read_csv('dataset/merged_stories_full.csv')
tokenizer = WordPieceTokenizer()
tokenizer.load('tokenizer/wordPieceVocab.json')


dataset = Custom_Dataset(dataset_csv, 2,tokenizer)
dataloader = torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False)
model = BERT(vocab_size=30522, max_seq_len=512, hidden_size=768, segment_vocab_size=2, num_hidden_layers=2, num_attention_heads=1, intermediate_size=3072)

optimizer = AdamW(model.parameters(), lr=5e-7)
loss_fct = torch.nn.CrossEntropyLoss()

epochs = 1
accumulation_steps = 3

model.train()


for epoch in range(epochs):
    for step,data in enumerate(dataloader):
        title,input_ids,attention_mask,segment_ids,labels = data[0],data[1],data[2],data[3],data[4]

        outputs = model(input_ids.to(device), attention_mask.to(device), segment_ids.to(device)) #Hauria de retornar dos tensors

    
        mlm_loss = loss_fct(outputs[1].view(-1,outputs[1].size(-1)), labels[1].view(-1))
        nsp_loss = loss_fct(outputs[0].view(-1, 2), outputs[0].view(-1))
        total_loss = mlm_loss + nsp_loss

        # Normalize considering gradient accumulation
        loss = loss / accumulation_steps
        loss.backward()  # Backpropagate



        # Print loss for monitoring
        print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")

        # Update weights at the end of training
        if step % accumulation_steps == 0:
            optimizer.step()  
            optimizer.zero_grad()
            
    optimizer.step()  
    optimizer.zero_grad()