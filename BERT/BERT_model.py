import torch
from BERT.embedding_layer import EmbeddingLayer
import torch.nn as nn
from transformers import DistilBertForSequenceClassification,DistilBertForMaskedLM

from BERT.encoder_layer import EncoderLayer

class BERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, vocab_size, seq_len = 512, d_model=768, n_layers=12, heads=12, dropout=0.1):
        """
        :param vocab_size: vocab_size of total words
        :param seq_len: maximum sequence length
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.heads = heads

        # paper noted they used 4 * hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = d_model * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = EmbeddingLayer(vocab_size=vocab_size, embed_size=d_model, seq_len=seq_len, dropout=dropout)

        # multi-layers transformer blocks, deep network
        self.encoder_blocks = nn.ModuleList(
            [EncoderLayer(d_model, heads, d_model * 4, dropout) for _ in range(n_layers)])

    def forward(self, x, segment_info):
        # attention masking for padded token
        # (batch_size, 1, seq_len, seq_len)
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x, segment_info)

        # running over multiple transformer blocks
        for encoder in self.encoder_blocks:
            x = encoder.forward(x, mask)
        return x

class NextSentencePrediction(nn.Module):
    """
    2-class classification model : is_next, is_not_next
    """

    def __init__(self, hidden):
        """
        :param hidden: BERT model output size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input):
        # use only the first token which is the [CLS]
        x = input[:, 0]
        x = self.linear(x)
        x = self.softmax(x)
        return x

class MaskedLanguageModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input):
        x = self.linear(input)
        x = self.softmax(x)
        return x

class BERTLM(nn.Module):
    """
    BERT Language Model
    Next Sentence Prediction Model + Masked Language Model
    Separated to be able to do inference to the main model
    """

    def __init__(self, bert: BERT, vocab_size):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()
        self.bert = bert
        self.next_sentence = NextSentencePrediction(self.bert.d_model)
        self.mask_lm = MaskedLanguageModel(self.bert.d_model, vocab_size)

    def forward(self, x, segment_label):
        x = self.bert(x, segment_label)
        nsp_output = self.next_sentence(x)
        mlm_output = self.mask_lm(x)
        return nsp_output, mlm_output
    

class BERT_TL(nn.Module):
    """
    BERT Language Model - Fine-tuning DistilBERT
    Next Sentence Prediction Model + Masked Language Model
    Separated to be able to do inference to the main model
    """

    def __init__(self, is_inference=False):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """
        super().__init__()
        self.is_inference = is_inference
        model_MLM = DistilBertForMaskedLM.from_pretrained("distilbert-base-uncased")
        model_NSP = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

        self.bert = model_MLM.distilbert
        for param in self.bert.parameters(): # we just keep unfrozen the last encoder (we can change that)
            param.requires_grad = False
        for param in self.bert.transformer.layer[-1].parameters():
            param.requires_Grad = True

        self.next_sentence = nn.Sequential(model_NSP.pre_classifier,model_NSP.classifier) # We can add here dropout regularization
        self.mask_lm = nn.Sequential(model_MLM.vocab_transform,model_MLM.vocab_layer_norm, model_MLM.vocab_projector,nn.LogSoftmax(dim=-1))

        self.d_model = 768
    def forward(self, x, segment_label):
        x = self.bert(x, segment_label)[0]
        pooled_output = torch.mean(x, dim=1) # Mean pooling

        if self.is_inference:
            return pooled_output
        
        nsp_output = self.next_sentence(pooled_output) 

        mlm_output = self.mask_lm(x)

        return nsp_output, mlm_output