from transformers import BertTokenizer, BertModel
import torch

# Initialize a BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

sentence = "I love to play football"


# Tokenize a text
tokens = tokenizer.tokenize(sentence)
tokens = ['[CLS]'] + tokens + ['[SEP]']
T=15
padded_tokens=tokens +['[PAD]' for _ in range(T-len(tokens))]
print("Padded tokens are \n {} ".format(padded_tokens))
attn_mask=[ 1 if token != '[PAD]' else 0 for token in padded_tokens  ]
print("Attention Mask are \n {} ".format(attn_mask))

seg_ids=[0 for _ in range(len(padded_tokens))]
print("Segment Tokens are \n {}".format(seg_ids))

sent_ids=tokenizer.convert_tokens_to_ids(padded_tokens)
print("Sentence indexes \n {} ".format(sent_ids))
token_ids = torch.tensor(sent_ids).unsqueeze(0) 
attn_mask = torch.tensor(attn_mask).unsqueeze(0) 
seg_ids   = torch.tensor(seg_ids).unsqueeze(0)

outputs = model(token_ids, attention_mask = attn_mask,token_type_ids = seg_ids)
hidden_states = outputs.last_hidden_state
cls_embeddings = hidden_states[:, 0, :]
cls_embeddings = cls_embeddings.squeeze(0)
print(cls_embeddings)
print(cls_embeddings.shape)

