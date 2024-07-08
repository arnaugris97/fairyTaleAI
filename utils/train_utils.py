# train_utils.py
import torch
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from Optimizer.scheduler_optim import ScheduledOptim
from tensorboard_logger import TensorBoardLogger
from tokenizer.wordPieceTokenizer import WordPieceTokenizer
from BERT.BERT_model import BERT, BERTLM
from dataloader.custom_dataset import Custom_Dataset
from torch.optim import Adam


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
    

def calculate_mlm_accuracy_top_k(logits, labels, k=5):
    """
    Calculate the accuracy for the masked language modeling task, considering the top-k predictions.

    Args:
    logits (torch.Tensor): Logits from the model of shape (batch_size, sequence_length, vocab_size).
    labels (torch.Tensor): Ground truth labels of shape (batch_size, sequence_length), 
                           where labels are 0 for non-masked tokens.
    k (int): Number of top predictions to consider for accuracy calculation.

    Returns:
    float: The top-k accuracy of the MLM task.
    """
    # Identify the positions of the masked tokens
    mask_positions = labels != 0

    # Extract logits at the masked positions
    masked_logits = logits[mask_positions]  # Shape: (num_masked_tokens, vocab_size)

    # Extract the true labels at the masked positions
    masked_labels = labels[mask_positions]  # Shape: (num_masked_tokens,)

    # Compute the top-k predicted tokens
    top_k_predictions = masked_logits.topk(k, dim=-1).indices  # Shape: (num_masked_tokens, k)

    # Check if the true labels are in the top-k predictions
    correct_predictions = masked_labels.unsqueeze(1) == top_k_predictions  # Shape: (num_masked_tokens, k)
    correct_predictions = correct_predictions.sum(dim=1)  # Sum over the second dimension to get bool for each token

    # Calculate the number of correct predictions
    num_correct_predictions = correct_predictions.sum().item()

    # Calculate the total number of masked tokens
    total_masked_tokens = mask_positions.sum().item()

    # Calculate the top-k accuracy
    top_k_accuracy = num_correct_predictions / total_masked_tokens

    return top_k_accuracy

def training_steps(model, scheduler, train_dataloader, device, accumulation_steps, logger, epoch, loss_fn, config):
    model.train()
    total_loss = 0
   
    total_steps = len(train_dataloader)
   
    for step, data in enumerate(train_dataloader):
        input_ids, segment_ids, next_sentence_labels, masked_lm_labels = data[1], data[3].to(device), data[4].to(device), data[5].to(device)

        next_sent_output, mask_lm_output = model.forward(input_ids,segment_ids)

        nsp_loss = loss_fn(next_sent_output, next_sentence_labels.view(-1))
        mlm_loss = loss_fn(mask_lm_output.transpose(1, 2), masked_lm_labels)

        loss = mlm_loss * config['mlm_weight'] + nsp_loss * config['nsp_weight']
        total_loss += loss.item()

        print(f"Epoch {epoch}, Training Step {step}, Loss: {loss.item()}")

        if torch.isnan(mlm_loss) or torch.isnan(nsp_loss):
            print("************* NaN detected in loss components")

            continue

        if step % accumulation_steps == 0:
            scheduler.zero_grad()
            loss.backward()
            scheduler.step_and_update_lr()
            
        logger.log_training_loss(loss.item(), epoch, step, total_steps)

    
    avg_loss = total_loss / total_steps
    logger.log_average_training_loss(avg_loss, epoch)

    scheduler.zero_grad()
    scheduler.step_and_update_lr()

    return model, avg_loss

def validation_step(model, val_dataloader, device, logger, epoch, loss_fn, config):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model.eval()

    total_loss = 0

    avg_loss = 0.0
    total_correct = 0
    total_element = 0
        

    with torch.no_grad():
        for step, data in enumerate(val_dataloader):
            input_ids, segment_ids, next_sentence_labels, masked_lm_labels = data[1], data[3].to(device), data[4].to(device), data[5].to(device)
            next_sentence_labels = next_sentence_labels.view(-1)

            next_sent_output, mask_lm_output = model.forward(input_ids,segment_ids)

            nsp_loss = loss_fn(next_sent_output, next_sentence_labels)
            mlm_loss = loss_fn(mask_lm_output.transpose(1, 2), masked_lm_labels)

            loss = mlm_loss * config['mlm_weight'] + nsp_loss * config['nsp_weight']
            total_loss += loss.item()
            
            print(f"Epoch {epoch}, Validation Step {step}, Loss: {loss.item()}")

            if torch.isnan(mlm_loss) or torch.isnan(nsp_loss):
                print("********* NaN detected in loss components")
                continue

            correct = next_sent_output.argmax(dim=-1).eq(next_sentence_labels).sum().item()
            avg_loss += loss.item()
            total_correct += correct
            total_element += next_sentence_labels.nelement()
            
            accuracy = calculate_mlm_accuracy_top_k(mask_lm_output, masked_lm_labels, k=1)
            accuracy_top5 = calculate_mlm_accuracy_top_k(mask_lm_output, masked_lm_labels, k=5)
            accuracy_top10 = calculate_mlm_accuracy_top_k(mask_lm_output, masked_lm_labels, k=10)
            logger.log_validation_accuracy_mlm_top5(accuracy_top5, epoch)
            logger.log_validation_accuracy_mlm_top10(accuracy_top10, epoch)
            logger.log_validation_accuracy_mlm(accuracy, epoch) 

           

    avg_loss = total_loss / len(val_dataloader)

    accuracy_val_nsp = total_correct * 100.0 / total_element
    accuracy_val_mlm = accuracy


    logger.log_validation_loss(avg_loss, epoch)
    logger.log_validation_accuracy_nsp(accuracy_val_nsp, epoch)


    return avg_loss, accuracy_val_nsp, accuracy_val_mlm

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
    train, val_test = train_test_split(dataset_csv, test_size=0.2, random_state=random_state)
    val, test = train_test_split(val_test, test_size=0.5, random_state=random_state)

    train_dataset = Custom_Dataset(train, 2, tokenizer, config['max_seq_len'])
    val_dataset = Custom_Dataset(val, 2, tokenizer, config['max_seq_len'])

    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=False, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, drop_last=True)

    bert_model = BERT(
        vocab_size=tokenizer.vocab_size,
        seq_len=config['max_seq_len'],
        d_model=config['BERT_hidden_size'],
        n_layers=config['BERT_num_hidden_layers'],
        heads=config['BERT_att_heads'],
        dropout=config['dropout']
        )

    model = BERTLM(bert_model, tokenizer.vocab_size)

    model.to(device)

    optimizer = Adam(model.parameters(), lr=config['lr'], betas=config['betas'], weight_decay=config['weight_decay'])
    scheduler = ScheduledOptim(
        optimizer, model.bert.d_model, n_warmup_steps=config['num_warmup_steps']
        )

    # Initialize the loss function
    loss_fn = torch.nn.NLLLoss(ignore_index=0)

    # Initialize the early stopper
    early_stopper = EarlyStopper(patience=config['stopper_patience'], min_delta=0)

    for epoch in range(config['epochs']):
        model, loss_train = training_steps(model,scheduler, train_dataloader, device, config['accumulation_steps'], logger, epoch, loss_fn, config)
        loss_val, accuracy_val_nsp, accuracy_val_mlm = validation_step(model, val_dataloader, device, logger, epoch, loss_fn, config)

        print(f"Epoch {epoch}, Train Loss: {loss_train}, Val Loss: {loss_val}, Acc NSP Loss: {accuracy_val_nsp}, Acc MLM Loss: {accuracy_val_mlm * 100:.2f}")

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
                "Val_Acc_nsp": accuracy_val_nsp,
                "Val_Acc_mlm": accuracy_val_mlm,
            }
            torch.save(checkpoint, savedir)
            checkpoint_loss = loss_val

        if early_stopper.early_stop(loss_val):
            print(' --- TRAINING STOPPED --- ')
            break
    
    logger.close()
