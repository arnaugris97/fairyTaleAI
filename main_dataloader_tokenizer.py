from BERT.BERT_model import BERT
from dataloader.custom_dataset import Custom_Dataset
from transformers import AdamW
from tokenizer.wordPieceTokenizer import WordPieceTokenizer, mask_tokens
import pandas as pd
import torch

dataset_csv = pd.read_csv('dataset/merged_stories_full.csv')
tokenizer = WordPieceTokenizer()
tokenizer.load('tokenizer/wordPieceVocab.json')

dataset = Custom_Dataset(dataset_csv, 1,tokenizer)

dataloader = torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False,num_workers=1,pin_memory=True)

for i,data in enumerate(dataloader):
    print('ok')



