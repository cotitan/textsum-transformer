import torch
from torch import nn
import numpy as np
import config
# use elmo embeddings, comment the following line if you don't need it
# from allennlp.modules.elmo import Elmo, batch_to_ids

pad_index = config.pad_index


def get_attn_mask(seq_q, seq_k):
    """
    :param seq_q: [batch, l_q]
    :param seq_k: [batch, l_k]
    """
    l_q = seq_q.size(-1)
    mask = seq_k.eq(pad_index).unsqueeze(1).expand(-1, l_q, -1)
    return mask


def get_subsequent_mask(seq_q):
    bs, l_q = seq_q.size()
    subsequent_mask = torch.triu(
        torch.ones((l_q, l_q), device=seq_q.device, dtype=torch.uint8), diagonal=1
    )
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(bs, -1, -1)
    return subsequent_mask


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
    def __init__(self, n_vocab, d_model, max_seq_len, embeddings=None):
        super(Embedding, self).__init__()
        self.word_embedding = nn.Embedding(n_vocab, d_model)
        if embeddings is not None:
            self.word_embedding = nn.Embedding.from_pretrained(embeddings, freeze=True)
        self.pos_embedding = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(max_seq_len, d_model))

    def forward(self, input_ids):
        pos_ids = torch.arange(input_ids.shape[-1], dtype=torch.long, device=input_ids.device)
        pos_ids.unsqueeze(0).expand_as(input_ids)
        embeddings = self.word_embedding(input_ids) + self.pos_embedding(pos_ids)
        return embeddings


class ScaledDotAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super(ScaledDotAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, q, k, v, mask):
        """
        perform scaled-dot-attention, given query,key,value
        :params q: query, size=[batch, q_len, d_q]
        :params k: key, size=[batch, k_len, d_k], d_q == d_k
        :params v: value, size=[batch, v_len, d_v], v_len==k_len
        :params mask: size=[batch, q_len, k_len]
        :return attn_vec:
        :return attn_weight:
        """
        scale = 1 / np.sqrt(q.size(-1))
        scores = torch.bmm(q, k.transpose(-1, -2)) * scale
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        attn_weight = self.softmax(scores)
        attn_weight = self.dropout(attn_weight)  # dropout
        context = torch.bmm(attn_weight, v)
        return context, attn_weight


class MultiheadAttention(nn.Module):
	def __init__(self, n_head, d_model, dropout=0.1):
		super(MultiheadAttention, self).__init__()
		self.n_head = n_head
		d_k = d_model // n_head
		d_v = d_model // n_head
		self.W_Q = nn.Linear(d_model, d_k * n_head)
		self.W_K = nn.Linear(d_model, d_k * n_head)
		self.W_V = nn.Linear(d_model, d_v * n_head)
		self.W_O = nn.Linear(d_v * n_head, d_model)

		self.attn_layer = ScaledDotAttention()
		self.dropout = nn.Dropout(p=dropout)
		self.layer_norm = nn.LayerNorm(d_model)

	def forward(self, Q, K, V, mask):
		"""

		:param Q: size=[bsz, l_q, d]
		:param K: size=[bsz, l_k, d]
		:param V: size=[bsz, l_k, d]
		:param mask: size=[bsz, l_q, l_k]
		:return:
		"""
		(bsz, l_q, d), l_k = Q.shape, K.shape[1]
		n = self.n_head

		residual = Q
		Qs = self.W_Q(Q).view(bsz, l_q, n, -1).transpose(1,2).contiguous().view(bsz * n, l_q, -1)
		Ks = self.W_K(K).view(bsz, l_k, n, -1).transpose(1,2).contiguous().view(bsz * n, l_k, -1)
		Vs = self.W_V(V).view(bsz, l_k, n, -1).transpose(1,2).contiguous().view(bsz * n, l_k, -1)

		mask = torch.cat([mask for _ in range(n)], dim=0)
		# [bsz * n, l_q, d // n],  [bsz * n, l_q, l_k]
		context, attns = self.attn_layer(Qs, Ks, Vs, mask)

		context = context.view(bsz, n, l_q, -1).transpose(1, 2).contiguous().view(bsz, l_q, d)
		attns = attns.view(bsz, n, l_q, l_k).transpose(0, 1)  # [n, bsz, l_q, l_k]

		# attn_weight = torch.cat(attn_weight, dim=-1)
		output = self.W_O(context)
		# TODO try output = self.layer_norm(self.dropout(output+residual))
		# instead of self.layer_norm(self.dropout(output) + residual)
		output = self.dropout(output)
		output = self.layer_norm(output + residual)
		return output, context, attns.sum(dim=0) / n
		# return output, context, attn_weight

        
class PositionWiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_inner, dropout=0.1):
        super(PositionWiseFeedForwardNet, self).__init__()
        self.w1 = nn.Linear(d_model, d_inner, bias=True)
        self.w2 = nn.Linear(d_inner, d_model, bias=True)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, inputs):
        """
        :param inputs: [batch, src_len, d_model]
        """
        residual = inputs
        output = self.relu(self.w1(inputs))
        output = self.w2(output)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


# class PositionWiseFeedForwardNet(nn.Module):
#     def __init__(self, d_model, d_inner, dropout=0.1):
#         super(PositionWiseFeedForwardNet, self).__init__()
#         self.conv1 = nn.Conv1d(d_model, d_inner, 1, bias=True)
#         self.conv2 = nn.Conv1d(d_inner, d_model, 1, bias=True)
#         self.relu = nn.ReLU()
#         self.layer_norm = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(p=dropout)
    
#     def forward(self, inputs):
#         """
#         :param inputs: [batch, src_len, d_model]
#         """
#         residual = inputs
#         # why transpose?
#         # w1 = [C_in, C_out, kernel_size]
#         # which requires input.shape=[N, C_in, L], C_in == in_channels
#         # output.shape=[N, C_out, L], N is batch_size
#         output = self.relu(self.conv1(inputs.transpose(-1, -2)))
#         output = self.conv2(output).transpose(-1, -2)
#         output = self.dropout(output)
#         output = self.layer_norm(output + residual)
#         return output


class EncoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_inner, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.multi_attn = MultiheadAttention(n_head, d_model, dropout)
        self.poswise_ffn = PositionWiseFeedForwardNet(d_model, d_inner, dropout)

    def forward(self, enc_inputs, mask=None):
        output, _, attn_weights = self.multi_attn(enc_inputs, enc_inputs, enc_inputs, mask)
        output = self.poswise_ffn(output)
        return output, attn_weights


class Encoder(nn.Module):
    def __init__(self, n_vocab, max_seq_len, n_layer, n_head,
                 d_model, d_inner, dropout=0.1, embeddings=None):
        super(Encoder, self).__init__()
        self.embedding = Embedding(n_vocab, d_model, max_seq_len, embeddings)
        self.layers = nn.ModuleList(
            [EncoderLayer(n_head, d_model, d_inner, dropout) for _ in range(n_layer)]
        )
    
    def forward(self, src_seq):

        mask = get_attn_mask(src_seq, src_seq)
        
        enc_output_list = []
        attn_weight_list = []

        enc_output = self.embedding(src_seq)
        for layer in self.layers:
            enc_output, attn_weight = layer(enc_output, mask)
            enc_output_list.append(enc_output)
            attn_weight_list.append(attn_weight)

        return enc_output, attn_weight_list


class DecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_inner, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.multi_attn_masked = MultiheadAttention(n_head, d_model, dropout)
        self.multi_attn = MultiheadAttention(n_head, d_model, dropout)
        self.poswise_ffn = PositionWiseFeedForwardNet(d_model, d_inner, dropout)
    
    def forward(self, enc_outputs, dec_inputs, self_mask=None, dec_enc_mask=None):
        dec_outputs, _, self_attns = self.multi_attn_masked(
            dec_inputs, dec_inputs, dec_inputs, self_mask
        )
        
        dec_outputs, _, dec_enc_attns = self.multi_attn(
            dec_outputs, enc_outputs, enc_outputs, dec_enc_mask
        )

        output = self.poswise_ffn(dec_outputs)
        return output, self_attns, dec_enc_attns


class Decoder(nn.Module):
    def __init__(self, n_vocab, max_seq_len, n_layer, n_head,
                 d_model, d_inner, dropout=0.1, embeddings=None):
        super(Decoder, self).__init__()
        self.embedding = Embedding(n_vocab, d_model, max_seq_len, embeddings)
        self.layers = nn.ModuleList(
            [DecoderLayer(n_head, d_model, d_inner, dropout) for _ in range(n_layer)]
        )

    def forward(self, enc_outputs, src_seq, tgt_seq):
        """  """
        dec_slf_mask = get_attn_mask(tgt_seq, tgt_seq)
        dec_subseq_mask = get_subsequent_mask(tgt_seq)
        dec_slf_mask = (dec_slf_mask + dec_subseq_mask).gt(0)

        dec_enc_mask = get_attn_mask(tgt_seq, src_seq)
        
        dec_slf_attn_list = []
        dec_enc_attn_list = []

        dec_outputs = self.embedding(tgt_seq)  # init
        
        for layer in self.layers:
            dec_outputs, dec_slf_attn, dec_enc_attn = \
                layer(enc_outputs, dec_outputs, dec_slf_mask, dec_enc_mask)
            
            dec_slf_attn_list.append(dec_slf_attn)
            dec_enc_attn_list.append(dec_enc_attn)

        return dec_outputs, dec_slf_attn_list, dec_enc_attn_list


class Transformer(nn.Module):
    def __init__(self, n_src_vocab, n_tgt_vocab, max_src_len, max_tgt_len, n_layer,
                 n_head, d_model, d_inner, dropout=0.1, embeddings=None,
                 src_tgt_emb_share=True, tgt_prj_wt_share=True):
        super(Transformer, self).__init__()

        self.encoder = Encoder(n_src_vocab, max_src_len, n_layer, n_head,
                               d_model, d_inner, dropout, embeddings)
        self.decoder = Decoder(n_tgt_vocab, max_tgt_len, n_layer, n_head,
                               d_model, d_inner, dropout, embeddings)

        self.tgt_word_proj = nn.Linear(d_model, n_tgt_vocab, bias=False)

        # It seems weight sharing leads to GPU out of memory
        if src_tgt_emb_share:
            assert n_src_vocab == n_tgt_vocab, \
                "To share word embedding table, the vocabulary size of src/tgt shall be the same."
            self.encoder.embedding.word_embedding.weight \
                = self.decoder.embedding.word_embedding.weight

        if tgt_prj_wt_share:
            self.tgt_word_proj.weight = self.decoder.embedding.word_embedding.weight
            self.logit_scale = (d_model ** -0.5)
            # self.logit_scale = 1.
        else:
            self.logit_scale = 1.

        self.loss_layer = nn.CrossEntropyLoss(ignore_index=pad_index)

    def forward(self, src_seq, tgt_seq):
        enc_outputs, *_ = self.encoder(src_seq)

        tgt_seq = tgt_seq[:, :-1]
        logits, dec_enc_attn = self.decode(enc_outputs, src_seq, tgt_seq)
        return logits * self.logit_scale, dec_enc_attn

    def encode(self, src_seq):
        enc_outputs, *_ = self.encoder(src_seq)
        return enc_outputs

    def decode(self, enc_outputs, src_seq, tgt_seq):
        dec_outputs, _, dec_enc_attn_list = self.decoder(enc_outputs, src_seq, tgt_seq)
        logits = self.tgt_word_proj(dec_outputs)
        return logits * self.logit_scale, dec_enc_attn_list[-1]


class EncoderShareEmbedding(nn.Module):
    def __init__(self, n_layer, n_head, d_model, d_inner, dropout=0.1):
        super(EncoderShareEmbedding, self).__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(n_head, d_model, d_inner, dropout) for _ in range(n_layer)]
        )

    def forward(self, enc_inputs, src_seq):
        mask = get_attn_mask(src_seq, src_seq)

        enc_output_list = []
        attn_weight_list = []

        enc_output = enc_inputs
        for layer in self.layers:
            enc_output, attn_weight = layer(enc_output, mask)
            enc_output_list.append(enc_output)
            attn_weight_list.append(attn_weight)

        return enc_output_list, attn_weight_list


class DecoderShareEmbedding(nn.Module):
    def __init__(self, n_layer, n_head, d_model, d_inner, dropout=0.1):
        super(DecoderShareEmbedding, self).__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(n_head, d_model, d_inner, dropout) for _ in range(n_layer)]
        )

    def forward(self, enc_output_list, dec_inputs, src_seq, tgt_seq):
        """
        Noticement: enc_outputs is a list
        """
        dec_slf_mask = get_attn_mask(tgt_seq, tgt_seq)
        dec_subseq_mask = get_subsequent_mask(tgt_seq)
        dec_slf_mask = (dec_slf_mask + dec_subseq_mask).gt(0)

        dec_enc_mask = get_attn_mask(tgt_seq, src_seq)

        dec_slf_attn_list = []
        dec_enc_attn_list = []

        dec_outputs = dec_inputs
        enc_outputs = enc_output_list[-1]  # use only the top layer
        for layer in self.layers:
            dec_outputs, dec_slf_attn, dec_enc_attn = \
                layer(enc_outputs, dec_outputs, dec_slf_mask, dec_enc_mask)

            dec_slf_attn_list.append(dec_slf_attn)
            dec_enc_attn_list.append(dec_enc_attn)

        return dec_outputs, dec_slf_attn_list, dec_enc_attn_list


class TransformerShareEmbedding(nn.Module):
    def __init__(self, n_vocab, max_seq_len, n_layer, n_head,
                 d_model, d_inner, tgt_prj_wt_share=False, embeddings=None):
        super(TransformerShareEmbedding, self).__init__()

        self.embedding = Embedding(n_vocab, d_model, max_seq_len, embeddings=embeddings)
        
        self.encoder = EncoderShareEmbedding(n_layer, n_head, d_model, d_inner)
        self.decoder = DecoderShareEmbedding(n_layer, n_head, d_model, d_inner)

        if tgt_prj_wt_share:
            self.tgt_word_proj = nn.Linear(d_model, n_vocab)
            self.tgt_word_proj.weight = self.embedding.word_embedding.weight
            self.logit_scale = (d_model ** -0.5)
        else:
            self.tgt_word_proj = nn.Linear(d_model, n_vocab)
            self.logit_scale = 1.

        self.loss_layer = nn.CrossEntropyLoss(ignore_index=pad_index)

    def forward(self, src_seq, tgt_seq):
        tgt_seq = tgt_seq[:, :-1]
        
        src_embeds = self.embedding(src_seq)
        tgt_embeds = self.embedding(tgt_seq)
        enc_output_list, *_ = self.encoder(src_embeds, src_seq)
        dec_outputs, *_ = self.decoder(enc_output_list, tgt_embeds, src_seq, tgt_seq)

        logits = self.tgt_word_proj(dec_outputs) * self.logit_scale
        return logits


class ElmoEmbedder(nn.Module):
    def __init__(self, requires_grad=False, dropout=0.5):
        super(ElmoEmbedder, self).__init__()
        # TODO,
        # set num_output_representations=3 may result to better performance
        self.elmo = Elmo(config.options_file,
                         config.weight_file,
                         num_output_representations=2,
                         requires_grad=requires_grad,
                         dropout=dropout)

    @ staticmethod
    def batch_to_ids(stncs, tgt_flag=False):
        """
        convert list of text into ids that elmo accepts
        :param stncs: [['I', 'Like', 'you'],['Yes'] ]
        :param tgt_flag: indicates if the inputs is a target sentences, if it is,
                        use only the previous words as context, and neglect last word
        :return ids: indices to feed into elmo
        """
        ids = batch_to_ids(stncs)  # (batch, seqlen, 50)
        if tgt_flag:
            ids = ids[:,:-1,:]  # neglect the last word
            b_size, _len, dim = ids.shape
            expand_ids = torch.zeros(b_size * _len, _len, dim, dtype=torch.long)
            for i in range(1, _len + 1):
                expand_ids[b_size*(i-1):b_size*i, :i, :] = ids[:, :i, :]
            return expand_ids
        return ids

    def forward(self, stncs, tgt_flag=False, idx=-1):
        """
        produce elmo embedding of a batch of variable length text
        :param stnc: list of sentences, a sentence is a list of words
        :param tgt_flag: indicates if the inputs is a target sentences, if it is,
                        use only the previous words as context
        :param idx: the idx-th layer representation
        :return embeddings: list of tensor in shape [batch, max_len, d_model]
        :return mask: 0-1 tensor in shape [batch, max_len], 0 for padding
        """
        elmo_ids = self.batch_to_ids(stncs, tgt_flag).cuda()
        elmo_output = self.elmo(elmo_ids)
        embeddings = elmo_output['elmo_representations'][idx]
        mask = elmo_output['mask']
        if tgt_flag:
            b_size = len(stncs)
            embeddings = [embeddings[i][i//b_size].unsqueeze(0)
                          for i in range(embeddings.shape[0])]
            embeddings = torch.cat(embeddings, dim=0).view(elmo_ids.shape[1], b_size, -1)
            embeddings = embeddings.transpose(0, 1)
            mask = mask[-b_size:]
        return embeddings, mask


class ElmoTransformer(nn.Module):
    def __init__(self, max_len, n_vocab, n_layer, n_head,
                 d_model, d_inner, dropout=0.1, elmo_requires_grad=False):
        super(ElmoTransformer, self).__init__()

        self.n_vocab = n_vocab
        self.d_model = d_model  # 256 is the size of small elmo
        self.elmo_embedder = ElmoEmbedder(requires_grad=elmo_requires_grad,
                                          dropout=dropout)  # False means freeze
        self.word_embedder = nn.Embedding(n_vocab, d_model)
        self.position_embedder = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(max_len, self.d_emb), freeze=True)

        self.encoder = EncoderShareEmbedding(n_layer, n_head, d_model, d_inner, dropout)
        self.decoder = DecoderShareEmbedding(n_layer, n_head, d_model, d_inner, dropout)

        self.dropout = nn.Dropout(dropout)
        self.tgt_word_prj = nn.Linear(self.d_model, n_vocab)

        self.loss_layer = nn.CrossEntropyLoss(ignore_index=config.pad_index)

    @ staticmethod
    def get_position_ids(stncs, tgt_flag=False):
        max_len = max(len(s) for s in stncs)
        if tgt_flag:
            max_len -= 1  # exclude </s> token
        pos_ids = torch.arange(max_len, dtype=torch.long)
        pos_ids = pos_ids.unsqueeze(0).expand(len(stncs), -1)
        return pos_ids.cuda()

    def embed(self, stncs, ids, tgt_flag=False):
        elmo_emb, mask = self.elmo_embedder(stncs, tgt_flag, idx=-1)
        pos_emb = self.position_embedder(self.get_position_ids(stncs, tgt_flag))
        if tgt_flag:
            word_emb = self.word_embedder(ids[:,:-1])
        else:
            word_emb = self.word_embedder(ids)
        embedding = torch.cat([elmo_emb, self.dropout(word_emb + pos_emb)], dim=-1)
        return embedding, mask

    def forward(self, src_stncs, tgt_stncs, src_ids, tgt_ids):
        src_embeds, src_mask = self.embed(src_stncs, src_ids, tgt_flag=False)
        tgt_embeds, tgt_mask = self.embed(tgt_stncs, tgt_ids, tgt_flag=True)

        enc_output_list, *_ = self.encoder(src_embeds, src_ids)
        dec_output, *_ = self.decoder(enc_output_list, tgt_embeds, src_mask, tgt_mask)
        logits = self.tgt_word_prj(dec_output)
        return logits

