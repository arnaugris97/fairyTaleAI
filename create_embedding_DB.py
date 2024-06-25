import pandas as pd
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType, Collection

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
        # if self.collection_name not in [col['name'] for col in collections]:
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
            embeddings = [sentence_embeddings[i].detach().numpy().astype(np.float16).flatten().tolist() for i in range(len(sentence_embeddings))]
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



class MilvusEmbeddingProcessor:
    def __init__(self, model_path, tokenizer_path, csv_file):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.csv_file = csv_file
        self.collection_name = "BERT_embeddings"
        self.model, self.tokenizer = self.load_bert_model()
        self.model.eval()

        # Connect to Milvus
        self.client = MilvusClient(
            uri="https://in03-38809e18a6b7a16.api.gcp-us-west1.zillizcloud.com",
            token="db_38809e18a6b7a16:Ld8)J(@h1EUYw..)",
        )

    def load_bert_model(self):
        return load_model(self.model_path, self.tokenizer_path)

       

    def load_sentences_from_csv(self):
        return pd.read_csv(self.csv_file)

    def process_and_store_embeddings(self):
        dataset = Custom_Dataset_DB(self.load_sentences_from_csv(), self.tokenizer, 512)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False, drop_last=False)
        id_counter = 0
        
        for step, data in enumerate(dataloader):
            sentence_embeddings = generate_embeddings(data, self.model)  # Generate embeddings
            embeddings = [sentence_embeddings[i] for i in range(len(sentence_embeddings))]
            texts = [data['sentence'][i] for i in range(len(sentence_embeddings))]
            titles = [data['title'][i] for i in range(len(sentence_embeddings))]

            entities = [
                {"id": id_counter + i, "embedding": embedding, "text": text, "title": title}
                for i, (embedding, title, text) in enumerate(zip(embeddings, titles, texts))
            ]
            
            id_counter += len(entities)

            self.client.insert(collection_name=self.collection_name, data=entities)
            print(f"Inserted batch {step+1} into Milvus from {len(dataset)/64}.")
            print("Percentage: ", (step+1)/(len(dataset)/64)*100, "%.")

        print("All embeddings inserted successfully.")

    def query_embeddings(self, query_embedding, top_k=5):
        data = [0.33955413,-0.7795727,-0.04268939,2.2367313,-0.7167185,0.6536712,-0.03885134,0.100777075,-1.4931902,-0.82186085,0.8834861,1.3997046,0.40566114,-1.1849066,-0.03835434,-0.60954154,0.09544808,-1.3524605,-1.7171016,-1.2532133,-0.12401919,-1.3354177,1.6055039,1.588439,0.5851139,0.24123745,-0.48519474,-0.88653857,-0.44173428,0.18546754,1.0545595,0.10281303,0.63400126,-1.2048526,-0.65634924,0.18418644,0.94492036,0.90889233,0.53702015,0.06973148,0.12434827,0.54958135,-1.311817,0.88703305,0.28411472,1.7568169,-0.54680836,-0.5229052,-0.5173166,1.5204394,-0.32510325,-0.77098876,-0.017536394,-0.8463769,-0.56019366,0.5988403,1.0315319,0.36137956,0.80721706,1.8852965,-1.203022,0.41935003,-1.1416146,-0.13573678,0.3259468,-0.6035002,1.0654463,-0.022892004,0.56100655,0.45933908,-1.4873414,0.061841805,1.1901394,1.0657413,-0.5082417,-0.006773464,0.15858802,-0.77862597,-1.0778295,-0.9447876,-0.21018967,-0.74530756,0.6148897,-0.40123585,-0.42044958,-0.78814566,0.7748687,1.8419821,-1.1086495,-1.594808,0.6046663,0.8554383,-0.29923582,-0.17774723,0.35097232,1.4852836,0.5520842,1.3270125,-0.21057615,0.9602869,-1.9701815,0.90521705,0.33362135,-0.12791736,0.30821863,-1.1826233,0.49828616,0.45180452,-0.46285507,-1.1134883,0.53781986,1.1725209,-0.40227938,0.03981261,1.0231143,-0.295985,0.040591888,0.23695646,0.22180066,1.8898156,-0.86173046,1.2872055,-1.3269156,-0.6366502,-0.6337743,-0.53691655,0.63777983,-0.8728216,-1.1226203,-0.014087603,1.5961698,-0.34285787,1.1517653,-0.4902211,-1.1095016,1.7742499,0.31743816,-0.08879158,1.5175042,0.2160559,-0.449341,0.5186563,-0.12595367,1.017619,-1.8681577,-1.6629121,-0.413056,-0.5815734,1.2969699,1.4243462,-0.27846175,0.97306126,-1.2670492,-1.1780311,0.034924492,-1.3015524,-2.4450302,0.77427745,2.4876614,-0.88414395,0.5641688,0.6812182,-0.14343299,-0.9196773,-2.061826,-1.4977005,-1.3977216,1.4524975,1.2700464,-1.9649315,-2.1147602,-0.5887813,-0.6472992,1.753552,-0.6291231,-0.11167939,-0.16996221,1.3585443,-1.8042113,1.2453915,-0.7720546,-0.1483678,-0.88215595,0.55278546,-0.023217406,1.0990999,1.7490631,-1.0391357,0.8676903,-0.33102992,-1.4531502,-1.820979,1.3316118,-0.99652076,0.05702858,1.2991079,0.35946524,-0.80934167,-0.86301416,-0.5242116,0.28894478,-0.18645526,-1.112791,0.40853208,-1.1290886,0.50725085,-1.9085095,0.2806237,1.1584535,0.70304155,-1.0958686,-0.12037869,-0.53492045,-1.0435987,-0.73156196,0.9152746,-0.6003658,-0.91678226,-1.6915278,1.2285812,0.5523031,1.1318012,-0.1065765,0.88876474,0.67102563,1.2375066,-0.09033413,2.6716063,1.8718551,-0.6793177,-1.2643145,0.89066404,0.8802402,0.75722134,-0.5780779,-1.1406257,-0.077703975,0.2487223,-0.7060922,0.23014559,-0.69447637,0.057091344,0.6064264,0.5651319,1.0107626,0.9745882,-1.5429325,0.6377671,-1.7012209,-1.9194592,-0.34591264,-0.12857102,0.6453676,1.962037,1.8785232,0.7783842]
        results = self.client.search(collection_name=self.collection_name, data=[query_embedding], limit=top_k, output_fields=["text", "title"])
        return results

    def process_query(self, sentence):
        inputs = preprocess_input(str(sentence), self.tokenizer)
        sentence_embeddings = generate_embeddings(inputs, self.model).squeeze().tolist()
        results = self.query_embeddings(sentence_embeddings, 10)
        return results

# Example usage:
if __name__ == "__main__":
    model_path = 'Checkpoints/checkpoint.pt'
    tokenizer_path = 'tokenizer/wordPieceVocab.json'
    csv_file = 'dataset/dataset_sentences.csv'
    storage_path = 'vectorDB/'

    # processor = ChromaEmbeddingProcessor(model_path, tokenizer_path, csv_file, storage_path)
    processor = MilvusEmbeddingProcessor(model_path, tokenizer_path, csv_file)

    results = processor.process_query('Once upon a time, in a world of wonder and enchantment, there lived a tiny, delicate girl named Thumbelina.')
    for result in results[0]:
        print(result)
    # processor.process_and_store_embeddings()
