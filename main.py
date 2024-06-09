# main.py
from train_utils import train_model

if __name__ == "__main__":
    config = {
        "lr": 1e-3,
        "num_warmup_steps":12,  #Defined at minibatch scale. It can be estimated: ~(1100*train_size/batch_size)/acc_step * epochs
        "batch_size": 64,
        "epochs": 20,
        "accumulation_steps": 5,
        "stopper_patience": 10,
        "path_dataset": "dataset/merged_stories_full.csv",
        "path_savedir": "Checkpoints/checkpoint.pt",
        "path_tokenizer": 'tokenizer/wordPieceVocab.json',
        "random_state": 123,
        "test_size": 0.1,
        "val_size": 0.1,
        "BERT_hidden_size": 768,
        "BERT_num_hidden_layers": 12,
        "BERT_att_heads": 12,
        "mlm_weight": 0.1,
        "nsp_weight": 0.9
    }

    train_model(config)
