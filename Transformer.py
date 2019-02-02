import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class ScaledDotAttention(nn.Module):
    def __init__(self):
        super(ScaledDotAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, q, k, v, mask=None):
        """
        perform scaled-dot-attention, given query,key,value
        :params q: query, size=[batch, 1, d_q]
        :params k: key, size=[batch, seqlen, d_q]
        :params v: value, size=[batch, seqlen, d_v]
        :return attn_vec:
        :return attn_weight:
        """
        scale = 1 / np.sqrt(q.shape[-1])
        # [batch, srclen, d_q] * [batch, d_q, outlen] *  ==> [batch, srclen, outlen]
        attn_weight = torch.bmm(q, k.transpose(1,2)) * scale
        if mask is not None:
            attn_weight = attn_weight.masked_fill(mask, -np.inf)
            
        attn_weight = self.softmax(attn_weight)
        # [batch, 1, seqlen] * [batch, seqlen, d_v] ==> [batch, 1, d_v]
        attn_vec = torch.bmm(attn_weight, v)
        return attn_vec, attn_weight


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.scale = nn.Parameter(torch.ones(hidden_size))
        self.shift = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps
    
    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = (x - mu).pow(2).mean(-1, keepdim=True)
        x = (x - mu) / torch.sqrt(sigma + self.eps) # avoid sqrt(0)
        return self.scale * x + self.shift


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    exponents = np.array([2 * (i // 2) / d_hid for i in range(d_hid)])
    pow_table = np.power(10000, exponents)
    sinusoid_table = np.array([pos / pow_table for pos in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)


class Embedding(nn.Module):
    def __init__(self, n_vocab, max_seq_len, emb_size, dropout=0.1, embeddings=None):
        super(Embedding, self).__init__()

        if embeddings is not None:
            self.word_embeddings = nn.Embedding.from_pretrained(embeddings)
        else:
            self.word_embeddings = nn.Embedding(n_vocab, emb_size)

        self.position_embeddings = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(max_seq_len, emb_size), freeze=True)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = LayerNorm(emb_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class PositionWiseFeedForward(nn.Module):
    def __init__(self, in_size, hid_size, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.w1 = nn.Conv1d(in_size, hid_size, 1)
        self.w2 = nn.Conv1d(hid_size, in_size, 1)
        self.layer_norm = nn.LayerNorm(in_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        # why transpose?
        # w1=[32, 32, 1] = [C_in, C_out, kernel_size]
        # which requires input.shape=[N, C_in, L], C_in == in_channels
        # output.shape=[N, C_out, L_out], N is batch_size
        output = self.w2(F.relu(self.w1(x.transpose(1,2))))
        output = output.transpose(1,2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head=8, d_model=512, d_q=64, d_k=64, d_v=64):
        super(MultiHeadAttention, self).__init__()
        assert d_q == d_k, "dim(key) must be equal to dim(query)"

        self.n_head = n_head

        self.W_Q = nn.ModuleList([nn.Linear(d_model, d_q) for _ in range(n_head)])
        self.W_K = nn.ModuleList([nn.Linear(d_model, d_k) for _ in range(n_head)])
        self.W_V = nn.ModuleList([nn.Linear(d_model, d_v) for _ in range(n_head)])
        self.W_O = nn.Linear(n_head * d_v, d_model)

        self.attn_layer = ScaledDotAttention()
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, q, k, v, mask=None):
        Qs = [Wq(q) for Wq in self.W_Q]
        Ks = [Wk(k) for Wk in self.W_K]
        Vs = [Wv(v) for Wv in self.W_V]

        heads = [self.attn_layer(Qs[i], Ks[i], Vs[i], mask) for i in range(self.n_head)]
        output = torch.cat([x[0] for x in heads], dim=-1)
        attns = torch.cat([x[1] for x in heads], dim=-1)

        output = self.W_O(output)
        output = self.layer_norm(output + q)
        return output, attns


class EncoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_inner, d_q, d_k, d_v):
        super(EncoderLayer, self).__init__()
        self.multi_attn = MultiHeadAttention(n_head, d_model, d_q, d_k, d_v)
        self.pos_ffn = PositionWiseFeedForward(d_model, d_inner)

    def forward(self, enc_input, non_pad_mask = None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.multi_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask
        )
        if non_pad_mask is not None:
            enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        if non_pad_mask is not None:
            enc_output *= non_pad_mask

        return enc_output, enc_slf_attn


class DecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_inner, d_q, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.multi_attn_masked = MultiHeadAttention(n_head, d_model, d_q, d_k, d_v)
        self.multi_attn = MultiHeadAttention(n_head, d_model, d_q, d_k, d_v)
        self.pos_ffn = PositionWiseFeedForward(d_model, d_inner)

    def forward(self, dec_input, enc_output, non_pad_mask=None,
            slf_attn_mask=None, dec_enc_attn_mask=None):

        dec_output, dec_slf_attn = self.multi_attn_masked(
            dec_input, dec_input, dec_input, mask=slf_attn_mask
        )
        if non_pad_mask is not None:
            dec_output *= non_pad_mask

        dec_output, dec_enc_attn = self.multi_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask
        )
        if non_pad_mask is not None:
            dec_output *= non_pad_mask

        dec_output = self.pos_ffn(dec_output)
        if non_pad_mask is not None:
            dec_output *= non_pad_mask

        return dec_output, dec_slf_attn, dec_enc_attn


class Encoder(nn.Module):
    def __init__(self, n_src_vocab, max_src_len, d_src_emb,
                N, n_head, d_q, d_k, d_v, d_model, d_inner, dropout=0.1):
        super(Encoder, self).__init__()
        
        self.embedding = Embedding(n_src_vocab, max_src_len, d_src_emb)
        self.layers = nn.ModuleList(
            [EncoderLayer(n_head, d_model, d_inner, d_q, d_k, d_v) for _ in range(N)]
        )
    
    def forward(self, src_seq, return_attns=False):
        enc_slf_attn_list = []
        enc_output_list = []

        # batch*seqlen ==> batch*seqlen*emb
        enc_output = self.embedding(src_seq)
        for layer in self.layers:
            enc_output, enc_slf_attn = layer(enc_output)
            enc_output_list.append(enc_output)
            if return_attns:
                enc_slf_attn_list.append(enc_slf_attn)
        
        if return_attns:
            return enc_output_list, enc_slf_attn_list
        return enc_output_list,


def get_non_pad_mask(seq):
    assert seq.dim() == 2
    # a = torch.tensor([1,3,2,2]), a.ne(2) = [1,1,0,0]
    # batch*seq ==> batch*seq*1
    return seq.ne(2).type(torch.float).unsqueeze(-1) # 2 == vocab["<pad>"]


def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(2) # 2 == vocab["<pad>"]
    # b*lk ==> b*1*lk ==> b*lq*lk, lk means len_key
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
    # padding_mask, the elements equal to <pad> are set to 1, other set to 0
    return padding_mask


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size()
    # torch.triu: return the upper triangular part of matrix,
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask


class Decoder(nn.Module):
    def __init__(self, n_tgt_vocab, max_tgt_len, d_tgt_emb,
                N, n_head, d_q, d_k, d_v, d_model, d_inner, dropout=0.1):
        super(Decoder, self).__init__()
        self.embedding = Embedding(n_tgt_vocab, max_tgt_len, d_tgt_emb, dropout)
        self.layers = nn.ModuleList(
            [DecoderLayer(n_head, d_model, d_inner, d_q, d_k, d_v) for _ in range(N)]
        )
    
    def forward(self, tgt_seq, src_seq, enc_output_list, return_attns=True):
        dec_slf_attn_list = []
        dec_enc_attn_list = []

        # -- Prepare masks
        non_pad_mask = get_non_pad_mask(tgt_seq)

        slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=tgt_seq)


        dec_output = self.embedding(tgt_seq)
        idx = 0
        for layer in self.layers:
            enc_output = enc_output_list[idx]
            idx += 1
            dec_output, dec_slf_attn, dec_enc_attn = layer(
                        dec_output, enc_output,
                        non_pad_mask=non_pad_mask,
                        slf_attn_mask=slf_attn_mask,
                        dec_enc_attn_mask=dec_enc_attn_mask)
            if return_attns:
                dec_slf_attn_list.append(dec_slf_attn)
                dec_enc_attn_list.append(dec_enc_attn)
        
        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,


class Transformer(nn.Module):
    def __init__(self, n_src_vocab, n_tgt_vocab, max_src_len, max_tgt_len,
                d_word_vec, N, n_head, d_q, d_k, d_v, d_model, d_inner,
                dropout=0.1, tgt_emb_prj_weight_share=False):

        super(Transformer, self).__init__()
        
        self.encoder = Encoder(n_src_vocab, max_src_len, d_word_vec,
                N, n_head, d_q, d_k, d_v, d_model, d_inner, dropout=0.1
        )

        self.decoder = Decoder(n_tgt_vocab, max_tgt_len, d_word_vec,
                N, n_head, d_q, d_k, d_v, d_model, d_inner, dropout=0.1)

        self.tgt_word_proj = nn.Linear(d_model, n_tgt_vocab)

        if tgt_emb_prj_weight_share:
            self.tgt_word_proj.weight = self.decoder.embedding.word_embeddings.weight
            self.x_logit_scale = (d_model ** -0.5)
        else:
            self.x_logit_scale = 1.0


        self.loss_layer = nn.CrossEntropyLoss(ignore_index=3) # 3==vocab['<pad>']

    def forward(self, src_seq, tgt_seq):
        tgt_seq = tgt_seq[:, :-1]
        enc_output_list, *_ = self.encoder(src_seq)
        dec_output, *_ = self.decoder(tgt_seq, src_seq, enc_output_list)
        seq_logit = self.tgt_word_proj(dec_output) * self.x_logit_scale
        return seq_logit

