from torch.utils.data import Dataset
import re
import random
from tokenizer.wordPieceTokenizer import mask_tokens, mask_tokens_BERT
import torch


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

    def __init__(self, dataset, sentences, tokenizer, max_seq_len=512):
        super().__init__()
        self.dataset = dataset
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

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

            token_ids_sentence1 = self.tokenizer.encode(sentence)
            token_ids_sentence2 = self.tokenizer.encode(next_sentence)
            input_ids, attention_mask, segment_ids = self.tokenizer.add_special_tokens(token_ids_sentence1, token_ids_sentence2, max_length=self.max_seq_len)
            masked_input_ids, labels = mask_tokens(input_ids, self.tokenizer)
            

            if len(masked_input_ids) == self.max_seq_len and (len(token_ids_sentence1) + len(token_ids_sentence2)) < (self.max_seq_len - 3):
                break

        return title, torch.tensor(masked_input_ids), torch.tensor(attention_mask), torch.tensor(segment_ids), torch.tensor([is_next]), torch.tensor(labels)
    

class Custom_Dataset_DB(Dataset):

    def __init__(self, dataset, tokenizer, max_seq_len=512):
        super().__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

            title = self.dataset.iloc[idx]['Title']
            text = str(self.dataset.iloc[idx]['Sentence'])

            token_ids = self.tokenizer.encode(text)
    
            if len(token_ids) > self.max_seq_len-2:
                token_ids = token_ids[:self.max_seq_len-2]
                token_ids = [self.tokenizer.word2idx[self.tokenizer.cls_token]] + token_ids + [self.tokenizer.word2idx[self.tokenizer.sep_token]]
            else:
                token_ids = [self.tokenizer.word2idx[self.tokenizer.cls_token]] + token_ids + [self.tokenizer.word2idx[self.tokenizer.sep_token]]+[self.tokenizer.word2idx[self.tokenizer.pad_token]] * (self.max_seq_len-2 - len(token_ids))
            
            # token_ids = token_ids +[self.tokenizer.pad_token_id] * (self.max_seq_len - len(token_ids))

    
            attention_mask = [1 if i < len(token_ids) else 0 for i in range(self.max_seq_len)]
            token_type_ids = [0] * self.max_seq_len
            
            # Convert lists to tensors
            input_ids = torch.tensor(token_ids, dtype=torch.long)
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)
            token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)

            return {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'token_type_ids': token_type_ids,
                    'sentence': text,
                    'title':title
                }
    
class Custom_Dataset_TL(Dataset):

    def __init__(self, dataset, sentences, tokenizer, max_seq_len=512):
        super().__init__()
        self.dataset = dataset
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

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

            token_ids_sentence1 = self.tokenizer.encode(sentence)
            token_ids_sentence2 = self.tokenizer.encode(next_sentence)[1:]
            
            tokens_with_special_tokens  = token_ids_sentence1 + token_ids_sentence2 # The tokenizer already puts the CLS and SEP tokens when encoding

            # Create attention mask
            attention_mask = [0] * len(tokens_with_special_tokens)

            # Create token segment type ids
            token_type_ids = [1] * (len(token_ids_sentence1)) + [2] * (len(token_ids_sentence2))
            
            input_ids = tokens_with_special_tokens + [self.tokenizer._convert_token_to_id(self.tokenizer.pad_token)] * (self.max_seq_len - len(tokens_with_special_tokens))
            attention_mask = attention_mask + [1] * (self.max_seq_len - len(attention_mask))
            segment_ids = token_type_ids + [0] * (self.max_seq_len - len(token_type_ids))
            masked_input_ids, labels = mask_tokens_BERT(input_ids, self.tokenizer)

            if len(masked_input_ids) == self.max_seq_len and (len(token_ids_sentence1) + len(token_ids_sentence2)) < (self.max_seq_len - 3):
                break

        return title, torch.tensor(masked_input_ids), torch.tensor(attention_mask), torch.tensor(segment_ids), torch.tensor([is_next]), torch.tensor(labels)