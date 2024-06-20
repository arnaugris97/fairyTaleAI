import pandas as pd
import chromadb
from inference_utils import load_model, preprocess_input, generate_embeddings
import numpy as np

class ChromaEmbeddingProcessor:
    def __init__(self, model_path, tokenizer_path, csv_file):
        # Initialize with paths and filenames
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.csv_file = csv_file
        self.collection_name = "BERT_embeddings"  # Name for the collection in ChromaDB
        self.chromadb_client = self.initialize_chromadb_client()  # Initialize ChromaDB client
        self.model, self.tokenizer = self.load_bert_model()  # Load BERT model and tokenizer
        self.model.eval()  # Set BERT model to evaluation mode
        self.titles = []  # List to store titles from CSV

    def initialize_chromadb_client(self):
        # Initialize ChromaDB client and create collection if not exists
        chromadb_client = chromadb.Client()
        collections = chromadb_client.list_collections()
        if self.collection_name not in [col['name'] for col in collections]:
            chromadb_client.create_collection(name=self.collection_name, metadata={"hnsw:space": "cosine"})
        return chromadb_client

    def load_bert_model(self):
        # Load BERT model and tokenizer using provided paths
        return load_model(self.model_path, self.tokenizer_path)

    def load_sentences_from_csv(self):
        # Load sentences and titles from CSV file
        df = pd.read_csv(self.csv_file)
        sentences = df['Sentence'].tolist()
        titles = df['Title'].tolist()
        return sentences, titles

    def process_and_store_embeddings(self):
        # Process sentences from CSV, generate embeddings, and store in ChromaDB
        sentences, titles = self.load_sentences_from_csv()
        embeddings = []
        documents = []
        ids = []
        self.titles = []
        num_sentences = len(sentences)

        for i, (sentence, title) in enumerate(zip(sentences, titles)):
            sentence = str(sentence)  # Ensure sentence is a string
            inputs = preprocess_input(sentence, self.tokenizer)  # Preprocess input for BERT
            sentence_embeddings = generate_embeddings(inputs, self.model).detach().numpy().flatten().tolist()  # Generate embeddings
            embeddings.append(sentence_embeddings)  # Store embeddings
            documents.append(sentence)  # Store original sentences
            ids.append(f"id{i}")  # Assign IDs
            self.titles.append({"title": title})  # Store titles

            if i + 1 >= 30:  # Limit to 30 sentences for demonstration
                break

        collection = self.chromadb_client.get_collection(name=self.collection_name)

        # Insert documents into the collection
        try:
            collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=self.titles,
                ids=ids
            )
            print("Inserted documents successfully")
        except Exception as e:
            print("Failed to insert documents.")

        # Query and print the results for the first embedding
        self.query_embeddings(collection, embeddings[0])

    def query_embeddings(self, collection, query_embedding, top_k=5):
        # Query embeddings in the collection and print results
        try:
            results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
            print(f'IDs: {results["ids"]}')
            print(f'Documents: {results["documents"]}')
            print(f'Distances: {results["distances"]}')
        except Exception as e:
            print("Failed to query embeddings.")

# Example usage:
if __name__ == "__main__":
    model_path = 'Checkpoints/checkpoint.pt'
    tokenizer_path = 'tokenizer/wordPieceVocab.json'
    csv_file = 'dataset/dataset_sentences.csv'

    processor = ChromaEmbeddingProcessor(model_path, tokenizer_path, csv_file)
    processor.process_and_store_embeddings()
