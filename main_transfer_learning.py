import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Load the pre-trained model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

# Prepare the dataset
train_dataset = ...
test_dataset = ...

# Create a custom dataset class
class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset.iloc[idx, 0]
        label = self.dataset.iloc[idx, 1]
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,
            return_attention_mask=True,
            return_tensors="pt"
        )
        return {
            "input_ids": inputs["input_ids"].flatten(),
            "attention_mask": inputs["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

# Create a trainer
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        evaluation_strategy="epoch",
        learning_rate=2e-5
    ),
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# Train the model
trainer.train()

# Evaluate the model
trainer.evaluate()

# Save the model
model.save("finetuned_model")