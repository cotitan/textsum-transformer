import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

# 842MB my model, 567mb others model, compared on the same parameters

pad_value = 3


class ScaledDotAttention(nn.Module):
    def __init__(self):
        super(ScaledDotAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, q, k, v, mask=None):
        """
        perform scaled-dot-attention, given query,key,value
        :params q: query, size=[batch, n_head, q_len, d_q]
        :params k: key, size=[batch, n_head, k_len, d_k], d_q == d_k
        :params v: value, size=[batch, n_head, v_len, d_v], v_len==k_len
        :params mask: size=[batch, n_head, q_len, k_len]
        :return attn_vec:
        :return attn_weight:
        """
        scale = 1 / np.sqrt(q.shape[-1])
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale  # [batch, n_head, q_len, k_len]
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
            
        attn_weight = self.softmax(scores)
        # [batch, n_head, q_len, k_len] * [batch, n_head, k_len, d_v] = [batch, n_head, q_len, d_v]
        context = torch.matmul(attn_weight, v)
        return context, attn_weight


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
            get_sinusoid_encoding_table(max_seq_len, emb_size), freeze=False)

    def forward(self, input_ids, pos_ids):
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(pos_ids)

        embeddings = words_embeddings + position_embeddings
        return embeddings


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.conv1 = nn.Conv1d(d_model, d_ff, 1)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1)
        self.layer_norm = nn.LayerNorm(d_model)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        # why transpose?
        # w1 = [C_in, C_out, kernel_size]
        # which requires input.shape=[N, C_in, L], C_in == in_channels
        # output.shape=[N, C_out, L], N is batch_size
        output = F.relu(self.conv1(x.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        # output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head=8, d_model=512, d_q=64, d_k=64, d_v=64):
        super(MultiHeadAttention, self).__init__()
        assert d_q == d_k, "dim(key) must be equal to dim(query)"

        self.n_head = n_head

        self.W_Q = nn.Linear(d_model, d_q * n_head)
        self.W_K = nn.Linear(d_model, d_k * n_head)
        self.W_V = nn.Linear(d_model, d_v * n_head)
        self.W_O = nn.Linear(n_head * d_v, d_model)

        self.attn_layer = ScaledDotAttention()
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, q, k, v, mask=None):
        """

        :param q: size(batch, seqlen, d_model)
        :param k:
        :param v:
        :param mask:
        :return:
        """
        # (batch, n_head, seq_len, d)
        Qs = self.W_Q(q).view(q.shape[0], q.shape[1], self.n_head, -1).transpose(1, 2)
        Ks = self.W_K(k).view(k.shape[0], k.shape[1], self.n_head, -1).transpose(1, 2)
        Vs = self.W_V(v).view(v.shape[0], v.shape[1], self.n_head, -1).transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.n_head, 1, 1)
        context, attn = self.attn_layer(Qs, Ks, Vs, mask)  # batch, n_head, q_len, d_v

        context = torch.cat([context[:, i, :, :] for i in range(self.n_head)], dim=-1)
        output = self.W_O(context)
        output = self.layer_norm(output + q)
        return output, attn


class EncoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_ff, d_q, d_k, d_v):
        super(EncoderLayer, self).__init__()
        self.multi_attn = MultiHeadAttention(n_head, d_model, d_q, d_k, d_v)
        self.pos_ffn = PositionWiseFeedForward(d_model, d_ff)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.multi_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask
        )
        # if non_pad_mask is not None:
        #     enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        # if non_pad_mask is not None:
        #     enc_output *= non_pad_mask

        return enc_output, enc_slf_attn


class DecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_ff, d_q, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.multi_attn_masked = MultiHeadAttention(n_head, d_model, d_q, d_k, d_v)
        self.multi_attn = MultiHeadAttention(n_head, d_model, d_q, d_k, d_v)
        self.pos_ffn = PositionWiseFeedForward(d_model, d_ff)

    def forward(self, dec_input, enc_output, non_pad_mask=None,
                slf_attn_mask=None, dec_enc_attn_mask=None):

        dec_output, dec_slf_attn = self.multi_attn_masked(
            dec_input, dec_input, dec_input, mask=slf_attn_mask
        )  # diff: mask=None in Transformer_ref.py
        # if non_pad_mask is not None:
        #     dec_output *= non_pad_mask

        dec_output, dec_enc_attn = self.multi_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask
        )
        # if non_pad_mask is not None:
        #     dec_output *= non_pad_mask

        dec_output = self.pos_ffn(dec_output)
        # if non_pad_mask is not None:
        #     dec_output *= non_pad_mask

        return dec_output, dec_slf_attn, dec_enc_attn


class Encoder(nn.Module):
    def __init__(self, n_src_vocab, max_src_len, d_src_emb,
                 n_layer, n_head, d_q, d_k, d_v, d_model, d_ff, dropout=0.1):
        super(Encoder, self).__init__()
        
        self.embedding = Embedding(n_src_vocab, max_src_len, d_src_emb)
        self.layers = nn.ModuleList(
            [EncoderLayer(n_head, d_model, d_ff, d_q, d_k, d_v) for _ in range(n_layer)]
        )
    
    def forward(self, src_seq, return_attns=False):
        enc_slf_attn_list = []
        enc_output_list = []

        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        non_pad_mask = get_non_pad_mask(src_seq)

        # batch*seqlen ==> batch*seqlen*emb
        enc_output = self.embedding(src_seq)
        for layer in self.layers:
            enc_output, enc_slf_attn = layer(enc_output,
                                            non_pad_mask,
                                            slf_attn_mask)
            enc_output_list.append(enc_output)
            enc_slf_attn_list.append(enc_slf_attn)
        
        if return_attns:
            return enc_output_list, enc_slf_attn_list
        return enc_output_list,


def get_non_pad_mask(seq):
    assert seq.dim() == 2
    # a = torch.tensor([1,3,2,2]), a.ne(2) = [1,1,0,0]
    # batch*seq ==> batch*seq*1
    return seq.ne(pad_value).type(torch.float).unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(pad_value)
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
                n_layer, n_head, d_q, d_k, d_v, d_model, d_ff, dropout=0.1):
        super(Decoder, self).__init__()
        self.embedding = Embedding(n_tgt_vocab, max_tgt_len, d_tgt_emb, dropout)
        self.layers = nn.ModuleList(
            [DecoderLayer(n_head, d_model, d_ff, d_q, d_k, d_v) for _ in range(n_layer)]
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
                        dec_output, enc_output_list[-1],
                        non_pad_mask=non_pad_mask,
                        slf_attn_mask=slf_attn_mask,
                        dec_enc_attn_mask=dec_enc_attn_mask)

            dec_slf_attn_list.append(dec_slf_attn)
            dec_enc_attn_list.append(dec_enc_attn)
        
        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,


class Transformer(nn.Module):
    def __init__(self, n_src_vocab, n_tgt_vocab, max_src_len, max_tgt_len,
                d_word_vec, n_layer, n_head, d_q, d_k, d_v, d_model, d_ff,
                dropout=0.1, tgt_emb_prj_weight_share=False):

        super(Transformer, self).__init__()
        
        self.encoder = Encoder(n_src_vocab, max_src_len, d_word_vec, n_layer,
                               n_head, d_q, d_k, d_v, d_model, d_ff, dropout=dropout)

        self.decoder = Decoder(n_tgt_vocab, max_tgt_len, d_word_vec, n_layer,
                               n_head, d_q, d_k, d_v, d_model, d_ff, dropout=dropout)

        self.tgt_word_proj = nn.Linear(d_model, n_tgt_vocab)

        if tgt_emb_prj_weight_share:
            self.tgt_word_proj.weight = self.decoder.embedding.word_embeddings.weight
            self.x_logit_scale = (d_model ** -0.5)
        else:
            self.x_logit_scale = 1.0

        self.loss_layer = nn.CrossEntropyLoss(ignore_index=pad_value)

    def forward(self, src_seq, tgt_seq):
        tgt_seq = tgt_seq[:, :-1]
        enc_output_list, *_ = self.encoder(src_seq)
        dec_output, *_ = self.decoder(tgt_seq, src_seq, enc_output_list)
        seq_logit = self.tgt_word_proj(dec_output) * self.x_logit_scale
        return seq_logit


class EncoderShareEmbedding(nn.Module):
    def __init__(self, n_src_vocab, max_src_len, d_src_emb,
                 n_layer, n_head, d_q, d_k, d_v, d_model, d_ff, dropout=0.1):
        super(EncoderShareEmbedding, self).__init__()

        self.layers = nn.ModuleList(
            [EncoderLayer(n_head, d_model, d_ff, d_q, d_k, d_v) for _ in range(n_layer)]
        )

    def forward(self, src_seq, src_emb, return_attns=False):
        enc_slf_attn_list = []
        enc_output_list = []

        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        non_pad_mask = get_non_pad_mask(src_seq)

        # batch*seqlen ==> batch*seqlen*emb
        enc_output = src_emb
        for layer in self.layers:
            enc_output, enc_slf_attn = layer(enc_output,
                                             non_pad_mask,
                                             slf_attn_mask)
            enc_output_list.append(enc_output)
            enc_slf_attn_list.append(enc_slf_attn)

        if return_attns:
            return enc_output_list, enc_slf_attn_list
        return enc_output_list,


class DecoderShareEmbedding(nn.Module):
    def __init__(self, n_tgt_vocab, max_tgt_len, d_tgt_emb,
                 n_layer, n_head, d_q, d_k, d_v, d_model, d_ff, dropout=0.1):
        super(DecoderShareEmbedding, self).__init__()
        self.embedding = Embedding(n_tgt_vocab, max_tgt_len, d_tgt_emb, dropout)
        self.layers = nn.ModuleList(
            [DecoderLayer(n_head, d_model, d_ff, d_q, d_k, d_v) for _ in range(n_layer)]
        )

    def forward(self, tgt_seq, tgt_emb, src_seq, src_emb, enc_output_list, return_attns=True):
        dec_slf_attn_list = []
        dec_enc_attn_list = []

        # -- Prepare masks
        non_pad_mask = get_non_pad_mask(tgt_seq)

        slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=tgt_seq)

        dec_output = tgt_emb
        idx = 0
        for layer in self.layers:
            enc_output = enc_output_list[idx]
            idx += 1
            dec_output, dec_slf_attn, dec_enc_attn = layer(
                dec_output, enc_output_list[-1],
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask)

            dec_slf_attn_list.append(dec_slf_attn)
            dec_enc_attn_list.append(dec_enc_attn)

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,


class TransformerShareEmbedding(nn.Module):
    def __init__(self, n_src_vocab, n_tgt_vocab, max_src_len, max_tgt_len,
                 d_word_vec, n_layer, n_head, d_q, d_k, d_v, d_model, d_ff,
                 dropout=0.1, tgt_emb_prj_weight_share=False):

        super(TransformerShareEmbedding, self).__init__()

        self.embedding = Embedding(n_src_vocab, max_src_len, d_word_vec)

        self.encoder = EncoderShareEmbedding(n_src_vocab, max_src_len, d_word_vec, n_layer,
                               n_head, d_q, d_k, d_v, d_model, d_ff, dropout=dropout)

        self.decoder = DecoderShareEmbedding(n_tgt_vocab, max_tgt_len, d_word_vec, n_layer,
                               n_head, d_q, d_k, d_v, d_model, d_ff, dropout=dropout)

        self.tgt_word_proj = nn.Linear(d_model, n_tgt_vocab)

        if tgt_emb_prj_weight_share:
            self.tgt_word_proj.weight = self.decoder.embedding.word_embeddings.weight
            self.x_logit_scale = (d_model ** -0.5)
        else:
            self.x_logit_scale = 1.0

        self.loss_layer = nn.CrossEntropyLoss(ignore_index=pad_value)

    def forward(self, src_seq, tgt_seq, src_pos, tgt_pos):
        tgt_seq, tgt_pos = tgt_seq[:, :-1], tgt_pos[:, :-1]
        src_emb = self.embedding(src_seq, src_pos)
        tgt_emb = self.embedding(tgt_seq, tgt_pos)
        enc_output_list, *_ = self.encoder(src_seq, src_emb)
        dec_output, *_ = self.decoder(tgt_seq, tgt_emb, src_seq, src_emb, enc_output_list)
        seq_logit = self.tgt_word_proj(dec_output) * self.x_logit_scale
        return seq_logit


