# create_embedding_DB.py

import pandas as pd
import chromadb
from inference_utils import load_model, preprocess_input, generate_embeddings
from tokenizer.wordPieceTokenizer import WordPieceTokenizer

class ChromaEmbeddingProcessor:
    def __init__(self, model_path, csv_file):
        self.model_path = model_path
        self.csv_file = csv_file
        self.chromadb_client = chromadb.Client()
        self.model, _ = self.load_bert_model()  # Load BERT model
        self.model.eval()  # Set the model to evaluation mode
        self.custom_tokenizer = WordPieceTokenizer()  # Initialize tokenizer

    def load_bert_model(self):
        return load_model(self.model_path, None)

    def load_sentences_from_csv(self):
        df = pd.read_csv(self.csv_file)
        sentences = df['Sentence'].tolist()
        titles = df['Title'].tolist()
        return sentences, titles

    def process_and_store_embeddings(self):
        sentences, titles = self.load_sentences_from_csv()
        embeddings = []

        for sentence, title in zip(sentences, titles):
            inputs = preprocess_input(sentence, self.custom_tokenizer)
            sentence_embeddings = generate_embeddings(inputs, self.model).numpy().tolist()
            embeddings.append((sentence_embeddings, sentence, title))

        collection_name = 'BERT_embeddings'
        for embedding, sentence, title in embeddings:
            try:
                result = self.chromadb_client.get_collection(name=collection_name).insert_one(
                    {
                        'embedding': embedding,
                        'sentence': sentence,
                        'title': title
                    }
                )
                print(f"Inserted document with _id: {result.inserted_id}")
            except Exception as e:
                print(f"Failed to insert document: {e}")

# Example usage:
if __name__ == "__main__":
    model_path = 'Checkpoints/checkpoint.pt'
    csv_file = 'dataset/dataset_sentences.csv'

    processor = ChromaEmbeddingProcessor(model_path, csv_file)
    processor.process_and_store_embeddings()
