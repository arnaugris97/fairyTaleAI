from train_utils_TL import train_model
if __name__ == "__main__":
    config = {
        "lr": 1e-4,
        "num_warmup_steps": 500,  #Defined at minibatch scale. It can be estimated: ~(1100*train_size/batch_size)/acc_step * epochs (Volem 1/5 de les epocs que faci warmup)
        "batch_size": 100,
        "epochs": 20,
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

# # Prepare the dataset
# train_dataset = ...
# test_dataset = ...

# # Create a custom dataset class
# class SentimentDataset(torch.utils.data.Dataset):
#     def __init__(self, dataset, tokenizer):
#         self.dataset = dataset
#         self.tokenizer = tokenizer

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         text = self.dataset.iloc[idx, 0]
#         label = self.dataset.iloc[idx, 1]
#         inputs = self.tokenizer.encode_plus(
#             text,
#             add_special_tokens=True,
#             max_length=512,
#             return_attention_mask=True,
#             return_tensors="pt"
#         )
#         return {
#             "input_ids": inputs["input_ids"].flatten(),
#             "attention_mask": inputs["attention_mask"].flatten(),
#             "labels": torch.tensor(label, dtype=torch.long)
#         }

# # Create a trainer
# trainer = Trainer(
#     model=model,
#     args=TrainingArguments(
#         output_dir="results",
#         num_train_epochs=3,
#         per_device_train_batch_size=16,
#         per_device_eval_batch_size=64,
#         evaluation_strategy="epoch",
#         learning_rate=2e-5
#     ),
#     train_dataset=train_dataset,
#     eval_dataset=test_dataset
# )

# # Train the model
# trainer.train()

# # Evaluate the model
# trainer.evaluate()

# # Save the model
# model.save("finetuned_model")