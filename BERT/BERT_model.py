from BERT.MLM_head import MLMHead
from BERT.NSP_head import NSPHead
from BERT.embedding_layer import EmbeddingLayer
import torch.nn as nn

from BERT.encoder_layer import EncoderLayer


# class BERT(nn.Module):
#     def __init__(self, vocab_size, max_seq_len, hidden_size, segment_vocab_size, num_hidden_layers, num_attention_heads, intermediate_size, dropout_prob=0.1):
#         super(BERT, self).__init__()
#         self.embedding = EmbeddingLayer(vocab_size, max_seq_len, hidden_size, segment_vocab_size)
#         self.encoder = EncoderLayer(hidden_size, num_attention_heads, intermediate_size, num_hidden_layers, dropout_prob)
#         self.mlm_head = MLMHead(hidden_size, vocab_size)
#         self.nsp_head = NSPHead(hidden_size)

#     def forward(self, input_ids, attention_mask, segment_ids, input_ids_mask):
        
#         embedding_output = self.embedding(input_ids, segment_ids)
#         encoder_output = self.encoder(embedding_output, attention_mask, input_ids_mask)
        
#         sequence_output = encoder_output
#         pooled_output = encoder_output[:, 0]
        
#         mlm_logits = self.mlm_head(sequence_output)
#         nsp_logits = self.nsp_head(pooled_output)
        
#         return nsp_logits, mlm_logits


class BERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, vocab_size, d_model=768, n_layers=12, heads=12, dropout=0.1):
        """
        :param vocab_size: vocab_size of total words
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
        self.embedding = EmbeddingLayer(vocab_size=vocab_size, embed_size=d_model, seq_len=512, dropout=dropout)

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

    def forward(self, x):
        # use only the first token which is the [CLS]
        return self.softmax(self.linear(x[:, 0]))

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

    def forward(self, x):
        return self.softmax(self.linear(x))

class BERTLM(nn.Module):
    """
    BERT Language Model
    Next Sentence Prediction Model + Masked Language Model
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
        return self.next_sentence(x), self.mask_lm(x)