import pandas as pd
import chromadb
from chromadb import Settings
from inference_utils import load_model, preprocess_input, generate_embeddings
import numpy as np
from torch.utils.data import DataLoader
from dataloader.custom_dataset import Custom_Dataset_DB

class ChromaEmbeddingProcessor:
    def __init__(self, model_path, tokenizer_path, csv_file,storage_path):
        # Initialize with paths and filenames
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.csv_file = csv_file
        self.collection_name = "BERT_embeddings"  # Name for the collection in ChromaDB
        self.chromadb_client = self.initialize_chromadb_client(storage_path)  # Initialize ChromaDB client
        self.model, self.tokenizer = self.load_bert_model()  # Load BERT model and tokenizer
        self.model.eval()  # Set BERT model to evaluation mode
        self.titles = []  # List to store titles from CSV

    def initialize_chromadb_client(self,storage_path):
        # Initialize ChromaDB client and create collection if not exists
        chromadb_client = chromadb.PersistentClient(storage_path)
        collections = chromadb_client.list_collections()
        #if self.collection_name not in [col['name'] for col in collections]:
         #   chromadb_client.create_collection(name=self.collection_name, metadata={"hnsw:space": "cosine"})
        return chromadb_client

    def load_bert_model(self):
        # Load BERT model and tokenizer using provided paths
        return load_model(self.model_path, self.tokenizer_path)

    def load_sentences_from_csv(self):
        # Load sentences and titles from CSV file
        df = pd.read_csv(self.csv_file)
        # sentences = df['Sentence'].tolist()
        # titles = df['Title'].tolist()
        # return sentences, titles
        return df

    def process_and_store_embeddings(self):
        #   # Process sentences from CSV, generate embeddings, and store in ChromaDB
        # sentences, titles = self.load_sentences_from_csv()
        embeddings = []
        documents = []
        ids = []
        # num_sentences = len(sentences)
        dataset = Custom_Dataset_DB(self.load_sentences_from_csv(), self.tokenizer, 512)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False, drop_last=False)

        # for i, (sentence, title) in enumerate(zip(sentences, titles)):
            # sentence = str(sentence)  # Ensure sentence is a string
            # inputs = preprocess_input(sentence, self.tokenizer)  # Preprocess input for BERT
            # sentence_embeddings = generate_embeddings(inputs, self.model).detach().numpy().flatten().tolist()  # Generate embeddings
            # embeddings.append(sentence_embeddings)  # Store embeddings
            # documents.append(sentence)  # Store original sentences
            # ids.append(f"id{i}")  # Assign IDs
            # self.titles.append({"title": title})  # Store titles

            # if i + 1 >= 30:  # Limit to 30 sentences for demonstration
            #     break
        j=0
        for step, data in enumerate(dataloader):
            sentence_embeddings = generate_embeddings(data, self.model)  # Generate embeddings
            
            #for i in range(len(sentence_embeddings)):
            embeddings = [sentence_embeddings[i].detach().numpy().flatten().tolist() for i in range(len(sentence_embeddings))]
            documents = [data['sentence'][i] for i in range(len(sentence_embeddings))]   # Store original sentences
            self.titles = [{"title": data['title'][i]} for i in range(len(sentence_embeddings))]   # Store titles
            ids = [f"id{j+i}" for i in range(len(sentence_embeddings))]  # Assign IDs
            j+=64
            collection = self.chromadb_client.get_collection(name=self.collection_name)

            # Insert documents in the batch into the collection
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

            # We initialize again the variables to avoid their storage

            print('Creating DB:',str(j),'/',str(len(dataset)))
            #if j>1000:
             #   break
        # Query and print the results for the first embedding
        # res = self.query_embeddings(collection, embeddings[0])

    def query_embeddings(self, collection, query_embedding, top_k=5):
        # Query embeddings in the collection and print results
        try:
            results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
            # print(f'IDs: {results["ids"]}')
            # print(f'Documents: {results["documents"]}')
            # print(f'Distances: {results["distances"]}')

            return results
        except Exception as e:
            print("Failed to query embeddings.")

    def process_query(self,sentence):
        # Process an input query and return the results from the DB
        collection = self.chromadb_client.get_collection(name=self.collection_name)
        inputs = preprocess_input(str(sentence),self.tokenizer)
        sentence_embeddings = generate_embeddings(inputs, self.model)[0].detach().numpy().flatten().tolist()  # Generate embeddings
        results = self.query_embeddings(collection, sentence_embeddings)
        return results
# Example usage:
if __name__ == "__main__":
    model_path = 'Checkpoints/checkpoint.pt'
    tokenizer_path = 'tokenizer/wordPieceVocab.json'
    csv_file = 'dataset/dataset_sentences.csv'
    storage_path = 'vectorDB/'

    processor = ChromaEmbeddingProcessor(model_path, tokenizer_path, csv_file, storage_path)

    #results = processor.process_query('Magical lights, carrots and happy dreams')
    processor.process_and_store_embeddings()
