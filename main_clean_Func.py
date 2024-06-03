from BERT.BERT_model import BERT
from dataloader.custom_dataset import Custom_Dataset
from transformers import AdamW
from tokenizer.wordPieceTokenizer import WordPieceTokenizer, mask_tokens
import pandas as pd
import torch
from numpy.random import RandomState
from sklearn.metrics import accuracy_score

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            print('*')
            if self.counter >= self.patience:
                return True
        return False
    
#######################################################################

def training_step(model,optimizer,train_dataloader,device,accumulation_steps):
    model.train()
    for step,data in enumerate(train_dataloader):
        title,input_ids,attention_mask,segment_ids,next_sentence_labels,masked_lm_labels = data[0],data[1],data[2],data[3].to(device),data[4].to(device),data[5].to(device)

        if input_ids.shape[1] > 512:
            print(title,str(input_ids.shape[1]))

            continue
        outputs = model(input_ids.to(device), attention_mask.to(device), segment_ids.to(device)) #Hauria de retornar dos tensors
        nsp_logits, mlm_logits = outputs

    
        mlm_loss = torch.nn.functional.cross_entropy(mlm_logits.view(-1, mlm_logits.size(-1)), masked_lm_labels.view(-1))
        nsp_loss = torch.nn.functional.cross_entropy(nsp_logits.view(-1, 2), next_sentence_labels.view(-1))
        loss = mlm_loss + nsp_loss

        # Check for NaN values in loss components
        if torch.isnan(mlm_loss) or torch.isnan(nsp_loss):
            print("NaN detected in loss components")
            continue

        loss.backward()  # Backpropagate

        # Update weights at the end of training
        if step % accumulation_steps == 0:
            optimizer.step()  
            optimizer.zero_grad()

    
    optimizer.step()  
    optimizer.zero_grad()

    return(model,loss)

def validation_step(model,val_dataloader,device):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model.eval()

    predicted_nsp = []
    y_nsp = []
    with torch.no_grad():
        for step,data in enumerate(val_dataloader):
            title,input_ids,attention_mask,segment_ids,next_sentence_labels,masked_lm_labels = data[0],data[1],data[2],data[3].to(device),data[4].to(device),data[5].to(device)

            if input_ids.shape[1] > 512:
                print(title,str(input_ids.shape[1]))
                continue
            outputs = model(input_ids.to(device), attention_mask.to(device), segment_ids.to(device)) #Hauria de retornar dos tensors
            nsp_logits, mlm_logits = outputs

            mlm_loss = torch.nn.functional.cross_entropy(mlm_logits.view(-1, mlm_logits.size(-1)), masked_lm_labels.view(-1))
            nsp_loss = torch.nn.functional.cross_entropy(nsp_logits.view(-1, 2), next_sentence_labels.view(-1))
            
            loss = mlm_loss + nsp_loss

            p_nsp = torch.max(nsp_logits, 1)[1]
            predicted_nsp+=p_nsp.tolist()
            y_nsp+=next_sentence_labels.view(-1).to('cpu').tolist()

    accuracy_val = accuracy_score(y_nsp,predicted_nsp)

    return (loss,accuracy_val)
def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    savedir = config["path_savedir"]
    
    dataset_csv = pd.read_csv(config["path_dataset"])
    tokenizer = WordPieceTokenizer()
    tokenizer.load(config["path_tokenizer"])

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

    train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=config["batch_size"],shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,batch_size=1,shuffle=False)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,batch_size=1,shuffle=False)

    model = BERT(vocab_size=tokenizer.vocab_size, max_seq_len=512, hidden_size=config['BERT_hidden_size'],
                 segment_vocab_size=2, num_hidden_layers=config['BERT_num_hidden_layers'], 
                 num_attention_heads=config['BERT_att_heads'], intermediate_size=4*config['BERT_hidden_size'], batch_size=config['batch_size'])
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=config['lr'])


    early_stopper = EarlyStopper(patience=config['stopper_patience'], min_delta=0)

    for epoch in range(config['epochs']):
        model,loss_train = training_step(model,optimizer,train_dataloader,device,config['accumulation_steps'])
        loss_val,accuracy_val = validation_step(model,val_dataloader,device)

        print(f"Epoch {epoch}, Train Loss: {loss_train.item()}, Val Loss: {loss_val.item()}, Acc NSP Loss: {accuracy_val}")

        if epoch == 0:
            checkpoint_loss = loss_val

        if loss_val <= checkpoint_loss:
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "BERT_hidden_size": config['BERT_hidden_size'],
                "BERT_num_hidden_layers":config["BERT_num_hidden_layers"],
                "BERT_att_heads":config["BERT_att_heads"],
                "Train_Loss":loss_train,
                "Val_Loss":loss_val,
                "Val_Acc":accuracy_val,
            }
            torch.save(checkpoint,savedir)
            checkpoint_loss = loss_val

        if early_stopper.early_stop(loss_val):
            print(' --- TRAINING STOPPED --- ')
            break

if __name__ == "__main__":

    config = {
        "lr": 5e-4,
        "batch_size": 1,
        "epochs": 3,
        "accumulation_steps": 5,
        "stopper_patience": 5,
        "path_dataset": "dataset/merged_stories_full.csv",
        "path_savedir": "Checkpoints/checkpoint.pt",
        "path_tokenizer": 'tokenizer/wordPieceVocab.json',
        "BERT_hidden_size": 256,
        "BERT_num_hidden_layers":1,
        "BERT_att_heads":1,
    }

    train_model(config)