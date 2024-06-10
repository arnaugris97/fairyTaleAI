from torch.utils.data import Dataset
import re
import random
from tokenizer.wordPieceTokenizer import mask_tokens
import torch
from transformers import BertTokenizer


def separate_sentences(text):
    text = text.replace('...','#^')
    text = text.replace('.','~.')
    text = text.replace('?','@?')
    text = text.replace('!','%!')
    
    b = re.split('[.?!^]' , text)
    c = [w.replace('~', '.') for w in b]
    c = [w.replace('@', '?') for w in c]
    c = [w.replace('#', '...') for w in c]
    c = [w.replace('%', '!') for w in c]
    
    return(c)

class Custom_Dataset(Dataset):

    def __init__(self, dataset, sentences, tokenizer1):
        super().__init__()
        self.dataset = dataset
        self.sentences = sentences
        self.tokenizer1 = tokenizer1

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        while True:
            title = self.dataset.iloc[idx]['Title']
            text = separate_sentences(self.dataset.iloc[idx]['cleaned_story'])
            list_sentences = [''.join(map(str, text[i:i+self.sentences])) for i in range(0, len(text), self.sentences)]
            
            
            if len(list_sentences[-1]) == 0:
                list_sentences.pop()

            if len(list_sentences) < 2:
                idx = random.randint(0, len(self.dataset) - 1)
                continue

            it = random.randint(0, len(list_sentences) - 2)

            sentence = list_sentences[it]

            if random.random() < 0.5:
                next_sentence = list_sentences[it + 1]
                is_next = 1
            else:
                idx2 = idx
                while idx2 == idx:
                    idx2 = random.randint(0, len(self.dataset) - 1)

                text2 = separate_sentences(self.dataset.iloc[idx2]['cleaned_story'])
                list_sentences2 = [''.join(map(str, text2[i:i + self.sentences])) for i in range(0, len(text2), self.sentences)]
                
                if len(list_sentences2) == 0:
                    continue
                
                if len(list_sentences2[-1]) == 0:
                    list_sentences2.pop()
                
                it = random.randint(0, len(list_sentences2) - 1)
                next_sentence = list_sentences2[it]
                is_next = 0            

            # token_ids_sentence1 = self.tokenizer.encode(sentence)
            token_ids_sentence1 = tokenizer.encode(sentence, add_special_tokens=False)
            # token_ids_sentence2 = self.tokenizer.encode(next_sentence)
            token_ids_sentence2 = tokenizer.encode(next_sentence, add_special_tokens=False)
            input_ids, attention_mask, segment_ids = self.tokenizer1.add_special_tokens(token_ids_sentence1, token_ids_sentence2, max_length=512)
            masked_input_ids, labels = mask_tokens(input_ids, self.tokenizer1)
            

            if len(masked_input_ids) == 512 and (len(token_ids_sentence1) + len(token_ids_sentence2)) < 509:
                break

        return title, torch.tensor(masked_input_ids), torch.tensor(attention_mask), torch.tensor(segment_ids), torch.tensor([is_next]), torch.tensor(labels)