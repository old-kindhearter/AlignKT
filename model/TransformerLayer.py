import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, d_q, d_k, d_v, n_heads, d_ff, dropout=0.1, bias=True):
        """
        :param d_model: int, the dimension of multi-head attention vector, equal to d_q, d_k, d_v, i.e. emb_size.
        :param d_q: int, the dimension of query vector
        :param d_k: int, the dimension of key vector
        :param d_v: int, the dimension of value vector
        :param n_heads: int, the number of attention heads
        :param d_ff: int, the number of fully connected layers between multi-head attention and feed-forward network.
        :param dropout: float, the dropout probability
        :param bias: bool, whether to use bias in linear layers
        """
        super().__init__()
        self.d_model, self.d_q, self.d_k, self.d_v = d_model, d_q, d_k, d_v
        self.n_heads, self.d_ff = n_heads, d_ff
        self.dropout_rate = dropout

        self.masked_sa = MultiHeadAttention(self.d_model, self.d_q, self.d_k, self.d_v, self.n_heads, bias)
        self.masked_mha = MultiHeadAttention(self.d_model, self.d_q, self.d_k, self.d_v, self.n_heads, bias)

        self.dropout_sa = nn.Dropout(dropout)
        self.layer_norm_sa = nn.LayerNorm(self.d_model, eps=1e-12)
        self.dropout_mha = nn.Dropout(dropout)
        self.layer_norm_mha = nn.LayerNorm(self.d_model, eps=1e-12)

        self.full_connect_in = nn.Linear(self.d_model, self.d_ff, bias=False)
        self.relu = nn.ReLU()
        self.full_connect_out = nn.Linear(self.d_ff, self.d_model, bias=False)

        self.dropout_fc_1 = nn.Dropout(dropout)
        self.dropout_fc_2 = nn.Dropout(dropout)
        self.layer_norm_fc = nn.LayerNorm(self.d_model, eps=1e-12)


    def forward(self, q, mem_k, mem_v, mask_sa=None, mask_mha=None, is_effect=True):
        # 1 self-attention layer
        q_sa, attn_sa = self.masked_sa(q, q, q, mask_sa, is_effect)
        q = self.layer_norm_sa(q + self.dropout_sa(q_sa))

        # 2 cross-attention layer
        q_mha, _ = self.masked_mha(q, mem_k, mem_v, mask_mha, is_effect)
        q = self.layer_norm_mha(q + self.dropout_mha(q_mha))

        # 3 ffn
        q_ffn_in = self.full_connect_in(q)
        q_ffn_out = self.full_connect_out(self.dropout_fc_1(self.relu(q_ffn_in)))
        q = self.layer_norm_fc(q + self.dropout_fc_2(q_ffn_out))

        return q, attn_sa

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, d_q, d_k, d_v, n_heads, d_ff, dropout=0.2, bias=True):
        """
        :param d_model: int, the dimension of multi-head attention vector, equal to d_q, d_k, d_v, i.e. emb_size.
        :param d_q: int, the dimension of query vector
        :param d_k: int, the dimension of key vector
        :param d_v: int, the dimension of value vector
        :param n_heads: int, the number of attention heads
        :param d_ff: int, the number of fully connected layers between multi-head attention and feed-forward network.
        :param dropout: float, the dropout probability
        :param bias: bool, whether to use bias in linear layers
        """
        super().__init__()
        self.d_model, self.d_q, self.d_k, self.d_v = d_model, d_q, d_k, d_v
        self.n_heads, self.d_ff = n_heads, d_ff
        self.dropout_rate = dropout
        self.masked_sa = MultiHeadAttention(self.d_model, self.d_q, self.d_k, self.d_v, self.n_heads, bias)

        self.dropout_sa = nn.Dropout(dropout)
        self.layer_norm_sa = nn.LayerNorm(self.d_model, eps=1e-12)

        self.full_connect_in = nn.Linear(self.d_model, self.d_ff, bias=bias)
        self.relu = nn.ReLU()
        self.full_connect_out = nn.Linear(self.d_ff, self.d_model, bias=bias)

        self.dropout_fc_1 = nn.Dropout(dropout)
        self.dropout_fc_2 = nn.Dropout(dropout)
        self.layer_norm_fc = nn.LayerNorm(self.d_model, eps=1e-12)


    def forward(self, q, k, v, mask=None, is_effect=True):
        # 1 self-attention layer
        q_sa, attn = self.masked_sa(q, k, v, mask, is_effect)
        q = self.layer_norm_sa(q + self.dropout_sa(q_sa))

        # 2 ffn
        q_ffn_in = self.full_connect_in(q)
        q_ffn_out = self.full_connect_out(self.dropout_fc_1(self.relu(q_ffn_in)))
        q = self.layer_norm_fc(q + self.dropout_fc_2(q_ffn_out))

        return q, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_q, d_k, d_v, n_heads, bias=True):
        """
        :param d_model: int, the dimension of multi-head attention vector, equal to d_q, d_k, d_v, i.e. emb_size.
        :param d_q: int, the dimension of query vector
        :param d_k: int, the dimension of key vector
        :param d_v: int, the dimension of value vector
        :param n_heads: int, the number of attention heads
        :param bias: bool, whether to use bias in linear layers
        """
        super().__init__()
        self.d_model, self.d_q, self.d_k, self.d_v = d_model, d_q, d_k, d_v
        self.n_heads = n_heads
        if d_model % n_heads != 0:
            raise ValueError(f"d_model{self.d_model} must be divisible by n_heads{self.n_heads}")
        else:
            self.d_mha = d_model // n_heads

        self.w_q = nn.Linear(self.d_q, self.d_model, bias=bias)
        self.w_k = nn.Linear(self.d_k, self.d_model, bias=bias)
        self.w_v = nn.Linear(self.d_v, self.d_model, bias=bias)
        self.w_o = nn.Linear(self.d_model, self.d_model, bias=bias)

        self.gamma = nn.Parameter(torch.ones(1, self.n_heads, 1, 1), requires_grad=True)  # shape=(1, n_heads, 1, 1)
        # self.dropout = nn.Dropout(dropout)
        self._init_weights()


    def _init_weights(self):
        xavier_uniform_(self.w_q.weight)
        xavier_uniform_(self.w_k.weight)
        xavier_uniform_(self.w_v.weight)
        xavier_uniform_(self.w_o.weight)
        constant_(self.w_q.bias, 0.)
        constant_(self.w_k.bias, 0.)
        constant_(self.w_v.bias, 0.)
        constant_(self.w_o.bias, 0.)


    def forward(self, query, key, value, mask=None, is_effect=True):
        batch_size = query.shape[0]

        # 1
        query, key, value = (self.w_q(query).reshape(batch_size, -1, self.n_heads, self.d_mha),
                             self.w_k(key).reshape(batch_size, -1, self.n_heads, self.d_mha), self.w_v(value).reshape(batch_size, -1, self.n_heads, self.d_mha))
        query, key, value = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)

        # 2
        if is_effect:
            output, attn = attention(query, key, value, mask, self.gamma, is_effect)  # attn: (batch_size, n_heads, seq_len, seq_len)
        else:
            output, attn = attention(query, key, value, mask, None, is_effect)

        # 3
        output = self.w_o(output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model))

        return output, attn



def attention(query, key, value, mask, gamma=None, is_effect=True):  #
    bs, n_heads, seq_len, emb_size  = query.size()

    if is_effect:  # add position effect and forget effect in Transformer sub-model.
        x1 = torch.arange(seq_len).float().expand(seq_len, -1).to(device)
        x2 = x1.transpose(0, 1).contiguous()

        if torch.equal(query, key):  # the encoder in Transformer implement a self-attention.
            relation_position = torch.clamp(torch.abs(x1 - x2), max=40)[None, None, :, :]  # MAX=40: default setting on the AS09
            #  [[0, 1, 2...],
            #   [1, 0, 1...],
            #   [2, 1, 0...]]

        else:  # the decoder in Transformer implement a cross-attention.
            relation_position = torch.clamp(torch.abs(x1 - x2), max=39)[None, None, :, :] + 1
            #  [[1, 2, 3...],
            #   [2, 1, 2...],
            #   [3, 2, 1...]]

        score = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(emb_size)

        # forget_effect: R(t) = exp(-sin(t/50) / (gamma*d+1))
        with torch.no_grad():  # get the d
            score_ = score.detach()
            score_ = score_.masked_fill(mask == False, -1e32)
            score_ = F.softmax(score_, dim=-1)
            score_ = score_ * mask.float()

        gamma = gamma.to(device)
        forget_effect = torch.exp(-torch.sin(relation_position/40) / (gamma*score_+1))
        score = score * forget_effect

        score = score.masked_fill(mask == False, -1e32)
        score = F.softmax(score, dim=-1)
        score = score * mask.float()

    else:  # no position effect and forget effect in personalState retriever, i.e. method _cal_personal_states().
        score = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(emb_size)

        score = score.masked_fill(mask == False, -1e32)
        score = F.softmax(score, dim=-1)
        score = score * mask.float()

    # calculate output
    output = torch.matmul(score, value)
    return output, score

