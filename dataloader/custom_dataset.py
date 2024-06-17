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

        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        
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

            token_ids_sentence1 = self.tokenizer1.encode(sentence)
            # token_ids_sentence1 = self.tokenizer1.encode(sentence, add_special_tokens=False)
            token_ids_sentence2 = self.tokenizer1.encode(next_sentence)
            # token_ids_sentence2 = tokenizer.encode(next_sentence, add_special_tokens=False)
            input_ids, attention_mask, segment_ids = self.tokenizer1.add_special_tokens(token_ids_sentence1, token_ids_sentence2, max_length=512)
            masked_input_ids, labels = mask_tokens(input_ids, self.tokenizer1)
            

            if len(masked_input_ids) == 512 and (len(token_ids_sentence1) + len(token_ids_sentence2)) < 509:
                break

        return title, torch.tensor(masked_input_ids), torch.tensor(attention_mask), torch.tensor(segment_ids), torch.tensor([is_next]), torch.tensor(labels)

# from torch.utils.data import Dataset
# import re
# import random
# from tokenizer.wordPieceTokenizer import mask_tokens
# import torch
# from transformers import BertTokenizer

# def separate_sentences(text):
#     text = text.replace('...', '#^')
#     text = text.replace('.', '~.')
#     text = text.replace('?', '@?')
#     text = text.replace('!', '%!')
    
#     b = re.split('[.?!^]', text)
#     c = [w.replace('~', '.') for w in b]
#     c = [w.replace('@', '?') for w in c]
#     c = [w.replace('#', '...') for w in c]
#     c = [w.replace('%', '!') for w in c]
    
#     return c

# class Custom_Dataset(Dataset):

#     def __init__(self, dataset, tokenizer1):
#         super().__init__()
#         self.dataset = dataset
#         self.tokenizer1 = tokenizer1
#         self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        
#         # Combine exactly two sentences to form the single element of the dataset
#         self.title, self.sentence, self.next_sentence, self.is_next = self._create_single_element()

#     def _create_single_element(self):
#         idx = random.randint(0, len(self.dataset) - 1)
#         title = self.dataset.iloc[idx]['Title']
#         text = separate_sentences(self.dataset.iloc[idx]['cleaned_story'])
#         if len(text) < 2:
#             raise ValueError("The text does not contain enough sentences.")
        
#         sentence = text[0]
#         next_sentence = text[1]
#         is_next = 1  # Assuming the two sentences are consecutive from the same text
#         return title, sentence,next_sentence, is_next

#     def __len__(self):
#         return 1

#     def __getitem__(self, idx):
#         token_ids_sentence1 = self.tokenizer.encode(self.sentence, add_special_tokens=False)
            
#         token_ids_sentence2 = self.tokenizer.encode(self.next_sentence, add_special_tokens=False)
#         input_ids, attention_mask, segment_ids = self.tokenizer1.add_special_tokens(token_ids_sentence1, token_ids_sentence2, max_length=512)
#         masked_input_ids, labels = mask_tokens(input_ids, self.tokenizer1)

    

#         return self.title, torch.tensor(masked_input_ids), torch.tensor(attention_mask), torch.tensor(segment_ids), torch.tensor([self.is_next]), torch.tensor(labels)