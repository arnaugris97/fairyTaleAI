from BERT.BERT_model import BERT
from dataloader.custom_dataset import Custom_Dataset
from transformers import AdamW
from tokenizer.wordPieceTokenizer import WordPieceTokenizer, mask_tokens
import pandas as pd
import torch
from numpy.random import RandomState

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_csv = pd.read_csv('dataset/merged_stories_full.csv')
tokenizer = WordPieceTokenizer()
tokenizer.load('tokenizer/wordPieceVocab.json')

seed = RandomState()
train_frac = 0.6
test_frac = 0.2
val_frac = 0.2

train_val = dataset_csv.sample(frac=train_frac+val_frac, random_state=seed)
test = dataset_csv.loc[~dataset_csv.index.isin(train_val.index)]

train = train_val.sample(frac=train_frac/(train_frac+val_frac), random_state=seed)
val = train_val.loc[~train_val.index.isin(train.index)]

train_dataset = Custom_Dataset(train, 2,tokenizer)
test_dataset = Custom_Dataset(test, 2,tokenizer)
val_dataset = Custom_Dataset(val, 2,tokenizer)

train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=1,shuffle=False)
test_dataloader = torch.utils.data.DataLoader(test_dataset,batch_size=1,shuffle=False)
val_dataloader = torch.utils.data.DataLoader(val_dataset,batch_size=1,shuffle=False)

model = BERT(vocab_size=tokenizer.vocab_size, max_seq_len=512, hidden_size=768, segment_vocab_size=2, num_hidden_layers=2, num_attention_heads=2, intermediate_size=3072)
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-7)
loss_fct = torch.nn.CrossEntropyLoss()

epochs = 1
accumulation_steps = 3

for epoch in range(epochs):
    model.train()
    for step,data in enumerate(train_dataloader):
        title,input_ids,attention_mask,segment_ids,next_sentence_labels,masked_lm_labels = data[0],data[1],data[2],data[3].to(device),data[4].to(device),data[5].to(device)

        if input_ids.shape[1] > 512:
            continue
        outputs = model(input_ids.to(device), attention_mask.to(device), segment_ids.to(device)) #Hauria de retornar dos tensors
        nsp_logits, mlm_logits = outputs

    
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

        # Print loss for monitoring
        print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")

        # Update weights at the end of training
        if step % accumulation_steps == 0:
            optimizer.step()  
            optimizer.zero_grad()
            
    optimizer.step()  
    optimizer.zero_grad()
    model.eval()
    for step,data in enumerate(val_dataloader):
        title,input_ids,attention_mask,segment_ids,next_sentence_labels,masked_lm_labels = data[0],data[1],data[2],data[3].to(device),data[4].to(device),data[5].to(device)

        if input_ids.shape[1] > 512:
            continue
        outputs = model(input_ids.to(device), attention_mask.to(device), segment_ids.to(device)) #Hauria de retornar dos tensors
        nsp_logits, mlm_logits = outputs

        ## ADD METRICS (ACCURACY,PREC@k)
