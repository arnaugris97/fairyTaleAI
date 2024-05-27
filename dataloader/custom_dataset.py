from torch.utils.data import Dataset
import re
import random
from tokenizer.wordPieceTokenizer import mask_tokens
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

    def __init__(self, dataset,sentences,tokenizer):
        super().__init__()
        self.dataset = dataset
        self.sentences = sentences
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        
        title = self.dataset.iloc[idx]['Title']
        text = separate_sentences(self.dataset.iloc[idx]['cleaned_story'])
        list_sentences = [''.join(map(str, text[i:i+self.sentences])) for i in range(0, len(text), self.sentences)]
        # First approach: for each batch, take one random sentence from each story.
        # The next sentence (own text or another) is defined randomly.

        it = random.randint(0,len(list_sentences)-2)
        sentence = list_sentences[it]

        if random.random()<0.5:
            next_sentence = list_sentences[it+1]
            is_next = True

        else:
            idx2 = idx
            while idx2 == idx:
                idx2 = random.randint(0,len(self.dataset)-1)

            text2 = separate_sentences(self.dataset.iloc[idx2]['cleaned_story'])
            list_sentences2 = [''.join(map(str, text2[i:i+self.sentences])) for i in range(0, len(text2), self.sentences)]

            it = random.randint(0,len(list_sentences2)-1)
            next_sentence = list_sentences2[it]
            is_next = False            

        token_ids_sentence1 = self.tokenizer.encode(sentence)
        token_ids_sentence2 = self.tokenizer.encode(next_sentence)
        input_ids, attention_mask, segment_ids = self.tokenizer.add_special_tokens(token_ids_sentence1, token_ids_sentence2, max_length=512)
        masked_input_ids, labels = mask_tokens(input_ids, self.tokenizer)
    
        return title, torch.tensor(masked_input_ids).unsqueeze(0), torch.tensor(attention_mask).unsqueeze(0), torch.tensor(segment_ids).unsqueeze(0), [is_next, labels]
    
    # HE LLEGIT EN EL LINK DE BAIX QUE FAN EL BATCH A TEXT LEVEL. LLAVORS RECORREN CADA PARÀGRAF DEL TEXT I RETORNEN UNA PARELLA DE
    # PARÀGRAF+NEXT_SENTENCE PER CADA UN DELS PARÀGRAFS. AQUESTA IMPLEMENTACIÓ ÉS STRAIGHTFORWARD AMB UN BUCLE, PERÒ NO HO IMPLEMENTO
    # ENCARA FINS QUE HO COMENTEM.

    # REF: https://d2l.ai/chapter_natural-language-processing-pretraining/bert-dataset.html 