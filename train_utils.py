# train_utils.py
import torch
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorboard_logger import TensorBoardLogger
from tokenizer.wordPieceTokenizer import WordPieceTokenizer
from BERT.BERT_model import BERT
from dataloader.custom_dataset import Custom_Dataset
from torch.optim import AdamW
from numpy.random import RandomState


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

def training_step(model, optimizer, train_dataloader, device, accumulation_steps, logger, epoch):
    model.train()
    total_loss = 0
   
    total_steps = len(train_dataloader)
   
    for step, data in enumerate(train_dataloader):
        title, input_ids, attention_mask, segment_ids, next_sentence_labels, masked_lm_labels = data[0], data[1], data[2], data[3].to(device), data[4].to(device), data[5].to(device)

        if input_ids.size(1) != 512:
            continue

        outputs = model(input_ids.to(device), attention_mask.to(device), segment_ids.to(device))
        nsp_logits, mlm_logits = outputs

        mlm_loss = torch.nn.functional.cross_entropy(mlm_logits.view(-1, mlm_logits.size(-1)), masked_lm_labels.view(-1))
        nsp_loss = torch.nn.functional.cross_entropy(nsp_logits.view(-1, 2), next_sentence_labels.view(-1))
        loss = mlm_loss + nsp_loss

        total_loss += loss.item()
        print(f"Epoch {epoch}, Training Step {step}, Loss: {loss.item()}")

        if torch.isnan(mlm_loss) or torch.isnan(nsp_loss):
            print("NaN detected in loss components")
            continue

        loss.backward()

        # Clip the gradients to prevent them from exploding
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)


        if step % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        logger.log_training_loss(loss.item(), epoch, step, total_steps)

    avg_loss = total_loss / total_steps
    logger.log_average_training_loss(avg_loss, epoch)

    optimizer.step()
    optimizer.zero_grad()

    return model, avg_loss

def validation_step(model, val_dataloader, device, logger, epoch):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model.eval()

    total_loss = 0
    predicted_nsp = []
    y_nsp = []
    with torch.no_grad():
        for step, data in enumerate(val_dataloader):
            title, input_ids, attention_mask, segment_ids, next_sentence_labels, masked_lm_labels = data[0], data[1], data[2], data[3].to(device), data[4].to(device), data[5].to(device)

            if input_ids.size(1) != 512:
                continue


            outputs = model(input_ids.to(device), attention_mask.to(device), segment_ids.to(device))
            nsp_logits, mlm_logits = outputs

            mlm_loss = torch.nn.functional.cross_entropy(mlm_logits.view(-1, mlm_logits.size(-1)), masked_lm_labels.view(-1))
            nsp_loss = torch.nn.functional.cross_entropy(nsp_logits.view(-1, 2), next_sentence_labels.view(-1))

            loss = mlm_loss + nsp_loss
            total_loss += loss.item()
            print(f"Epoch {epoch}, Validation Step {step}, Loss: {loss.item()}")


            p_nsp = torch.max(nsp_logits, 1)[1]
            predicted_nsp += p_nsp.tolist()
            y_nsp += next_sentence_labels.view(-1).to('cpu').tolist()

    avg_loss = total_loss / len(val_dataloader)
    accuracy_val = accuracy_score(y_nsp, predicted_nsp)

    logger.log_validation_loss(avg_loss, epoch)

    return avg_loss, accuracy_val

def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    savedir = config["path_savedir"]
    logger = TensorBoardLogger()

    # Read the dataset
    dataset_csv = pd.read_csv(config["path_dataset"])

    # Initialize the tokenizer
    tokenizer = WordPieceTokenizer()
    tokenizer.load(config["path_tokenizer"])

    random_state = config['random_state']
    test_size = config['test_size']
    val_size = config['val_size']

    # Split the dataset into train, validation, and test sets
    train_val, test = train_test_split(dataset_csv, test_size=test_size, random_state=random_state)
    train, val = train_test_split(train_val, test_size=val_size/(1-test_size), random_state=random_state)

    train_dataset = Custom_Dataset(train, 2, tokenizer)
    test_dataset = Custom_Dataset(test, 2, tokenizer)
    val_dataset = Custom_Dataset(val, 2, tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=False, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, drop_last=True)

    model = BERT(vocab_size=tokenizer.vocab_size, max_seq_len=512, hidden_size=config['BERT_hidden_size'],
                 segment_vocab_size=2, num_hidden_layers=config['BERT_num_hidden_layers'],
                 num_attention_heads=config['BERT_att_heads'], intermediate_size=4 * config['BERT_hidden_size'], batch_size=config['batch_size'])
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=config['lr'])

    early_stopper = EarlyStopper(patience=config['stopper_patience'], min_delta=0)

    for epoch in range(config['epochs']):
        model, loss_train = training_step(model, optimizer, train_dataloader, device, config['accumulation_steps'], logger, epoch)
        loss_val, accuracy_val = validation_step(model, val_dataloader, device, logger, epoch)

        print(f"Epoch {epoch}, Train Loss: {loss_train}, Val Loss: {loss_val}, Acc NSP Loss: {accuracy_val}")

        if epoch == 0:
            checkpoint_loss = loss_val

        if loss_val <= checkpoint_loss:
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "BERT_hidden_size": config['BERT_hidden_size'],
                "BERT_num_hidden_layers": config["BERT_num_hidden_layers"],
                "BERT_att_heads": config["BERT_att_heads"],
                "Train_Loss": loss_train,
                "Val_Loss": loss_val,
                "Val_Acc": accuracy_val,
            }
            torch.save(checkpoint, savedir)
            checkpoint_loss = loss_val

        if early_stopper.early_stop(loss_val):
            print(' --- TRAINING STOPPED --- ')
            break

    logger.close()
