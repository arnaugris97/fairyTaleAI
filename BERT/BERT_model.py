from BERT.MLM_head import MLMHead
from BERT.NSP_head import NSPHead
from BERT.embedding_layer import EmbeddingLayer
import torch.nn as nn

from BERT.encoder_layer import EncoderLayer


class BERT(nn.Module):
    def __init__(self, vocab_size, max_seq_len, hidden_size, segment_vocab_size, num_hidden_layers, num_attention_heads, intermediate_size, dropout_prob=0.1):
        super(BERT, self).__init__()
        self.embedding = EmbeddingLayer(vocab_size, max_seq_len, hidden_size, segment_vocab_size)
        self.encoder = EncoderLayer(hidden_size, num_attention_heads, intermediate_size, num_hidden_layers, dropout_prob)
        self.mlm_head = MLMHead(hidden_size, vocab_size)
        self.nsp_head = NSPHead(hidden_size)

    def forward(self, input_ids, attention_mask, segment_ids, masked_lm_labels=None, next_sentence_labels=None):
        
        embedding_output = self.embedding(input_ids, segment_ids)
        encoder_output = self.encoder(embedding_output, attention_mask)
        
        # sequence_output = encoder_output
        # pooled_output = encoder_output[:, 0]
        
        # mlm_logits = self.mlm_head(sequence_output)
        # nsp_logits = self.nsp_head(pooled_output)
        
        # total_loss = None
        # if masked_lm_labels is not None and next_sentence_labels is not None:
        #     loss_fct = nn.CrossEntropyLoss()
        #     mlm_loss = loss_fct(mlm_logits.view(-1, mlm_logits.size(-1)), masked_lm_labels.view(-1))
        #     nsp_loss = loss_fct(nsp_logits.view(-1, 2), next_sentence_labels.view(-1))
        #     total_loss = mlm_loss + nsp_loss

        # return total_loss, mlm_logits, nsp_logits
        return encoder_output