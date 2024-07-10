import chromadb
import pandas as pd

from utils.inference_utils import load_model, generate_embeddings
from torch.utils.data import DataLoader
from dataloader.custom_dataset import Custom_Dataset_DB


class ChromaEmbeddingProcessor:
    def __init__(self, model_path, tokenizer_path, csv_file, storage_path):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.csv_file = csv_file
        self.collection_name = "BERT_embeddings"
        self.chromadb_client = self.initialize_chromadb_client(storage_path)
        self.model, self.tokenizer = self.load_bert_model()
        self.model.eval()

    def initialize_chromadb_client(self, storage_path):
        return chromadb.PersistentClient(storage_path)

    def load_bert_model(self):
        return load_model(self.model_path, self.tokenizer_path)

    def load_sentences_from_csv(self):
        return pd.read_csv(self.csv_file)

    def process_and_store_embeddings(self):
        dataset = Custom_Dataset_DB(self.load_sentences_from_csv(), self.tokenizer, 512)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False, drop_last=False)
        for step, data in enumerate(dataloader):
            sentence_embeddings = generate_embeddings(data, self.model)
            embeddings = [emb.detach().numpy().flatten().tolist() for emb in sentence_embeddings]
            documents = [data['sentence'][i] for i in range(len(sentence_embeddings))]
            titles = [{"title": data['title'][i]} for i in range(len(sentence_embeddings))]
            ids = [f"id{step*64+i}" for i in range(len(sentence_embeddings))]
            collection = self.chromadb_client.get_collection(name=self.collection_name)
            collection.add(documents=documents, embeddings=embeddings, metadatas=titles, ids=ids)


# Example usage:
if __name__ == "__main__":
    model_path = 'Checkpoints/checkpoint20Epochs.pt'
    tokenizer_path = 'tokenizer/wordPieceVocab.json'
    csv_file = 'dataset/dataset_sentences_cleaned.csv'

    processor = ChromaEmbeddingProcessor(model_path, tokenizer_path, csv_file)
    processor.process_and_store_embeddings()
