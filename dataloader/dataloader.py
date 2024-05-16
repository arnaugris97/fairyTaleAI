from torch.utils.data import Dataset
import re

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

class Dataloader(Dataset):

    def __init__(self, dataset,sentences):
        super().__init__()
        self.dataset = dataset
        self.sentences = sentences
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        
        title = self.dataset.iloc[idx]['Title']
        text = separate_sentences(self.dataset.iloc[idx]['cleaned_story'])[:-1]
        list_sentences = [''.join(map(str, text[i:i+self.sentences])) for i in range(0, len(text), self.sentences)]

        return title,text,list_sentences