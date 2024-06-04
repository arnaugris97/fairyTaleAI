# main.py
from train_utils import train_model

if __name__ == "__main__":
    config = {
        "lr": 1e-4,
        "batch_size": 64,
        "epochs": 10,
        "accumulation_steps": 5,
        "stopper_patience": 5,
        "path_dataset": "dataset/merged_stories_full.csv",
        "path_savedir": "Checkpoints/checkpoint.pt",
        "path_tokenizer": 'tokenizer/wordPieceVocab.json',
        "random_state": 123,
        "test_size": 0.2,
        "val_size": 0.2,
        "BERT_hidden_size": 256,
        "BERT_num_hidden_layers": 4,
        "BERT_att_heads": 4,
    }

    train_model(config)
