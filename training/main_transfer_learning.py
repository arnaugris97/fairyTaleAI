from train_utils_TL import train_model
if __name__ == "__main__":
    config = {
        "lr": 1e-4,
        "num_warmup_steps": 500,  #Defined at minibatch scale. It can be estimated: ~(1100*train_size/batch_size)/acc_step * epochs (Volem 1/5 de les epocs que faci warmup)
        "batch_size": 100,
        "epochs": 100,
        "accumulation_steps": 6,
        "stopper_patience": 10,
        "path_dataset": "dataset/merged_stories_full.csv",
        "path_savedir": "Checkpoints/checkpoint.pt",
        "path_tokenizer": 'tokenizer/wordPieceVocab.json',
        "random_state": 123,
        "test_size": 0.1,
        "val_size": 0.1,
        "BERT_hidden_size": 256,
        "BERT_num_hidden_layers": 4,
        "BERT_att_heads": 4,
        "max_seq_len": 512,
        "dropout": 0.1,
        "mlm_weight": 1.0,
        "nsp_weight": 1.0,
        "weight_decay": 0.01,
        "betas": (0.9, 0.999)
    }

    train_model(config)