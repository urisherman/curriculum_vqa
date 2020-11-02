import torch
import math

from torch import nn
import torch.nn.functional as F

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def parse_dims_dict(args):
    return {
        'a': args['d_a'],
        'w': args['d_w'],
        'c': args['d_c'],
        'o': args['d_o'],
        'k': args['d_k'],
    }


def nn_parameter(*shape):
    W = nn.Parameter(torch.Tensor(*shape))
    nn.init.kaiming_uniform_(W, a=math.sqrt(5))
    return W


class ContextModel(nn.Module):
    def __init__(self, args, ans_vocab):
        super().__init__()
        d = parse_dims_dict(args)
        self.d = d
        self.ans_vocab = ans_vocab

        self.img_layer_norm = nn.LayerNorm([d['o']])
        self.W_viz = nn.Sequential(
            nn.Linear(d['o'], d['o']),
            nn.ReLU(),
            nn.Linear(d['o'], d['c']),
            nn.ReLU(),
        )
        self.img_feat_layer_norm = nn.LayerNorm([d['c']])

    def forward(self, prompt, img):
        img = self.img_layer_norm(img)
        img_feats = self.W_viz(img)
        img_feats = self.img_feat_layer_norm(img_feats)

        B, N_objs, _ = img.shape

        init_w = torch.ones(B, N_objs).to(device)
        no_answer = torch.zeros(B, 1, len(self.ans_vocab)).to(device)
        return {
            'X': img_feats,
            'init_w': init_w,
            'no_answer': no_answer
        }


class CondFFN(nn.Module):

    def __init__(self, d_in, d_h, d_out, d_seed, has_bias=True, n_hidden_layers=1):
        super().__init__()

        self.layers = [CondLinear(d_in, d_h, d_seed, has_bias)]
        for i in range(n_hidden_layers-1):
            self.layers.append(CondLinear(d_h, d_h, d_seed, has_bias))
        self.layers.append(CondLinear(d_h, d_out, d_seed, has_bias))

        self.layer_norm = nn.LayerNorm(d_h)

    def forward(self,  input, seed):
        x = input
        for l in self.layers[:-1]:
            x = l(x, seed)
            x = F.relu(x)
            x = self.layer_norm(x)

        x = self.layers[-1](x, seed)
        return x


class CondLinear(nn.Module):

    def __init__(self, d_in, d_out, d_seed, bias=False):
        super().__init__()
        self.W_op = nn_parameter(d_in, d_out, d_seed)
        self.has_bias = bias
        if bias:
            self.W_bias = nn_parameter(d_out, d_seed)
            # fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W_op)
            bound = 1 / math.sqrt(50)
            nn.init.uniform_(self.W_bias, -bound, bound)
        # self.layer_norm = nn.LayerNorm([d_out])

    def forward(self, input, seed):
        """
        seed: [B, d_seed]
        input: [B, N*, d_in]
        """
        B, d_seed = seed.shape
        d_in, d_out, d_seed = self.W_op.shape
        W = F.linear(seed, self.W_op.reshape(-1, d_seed))  # [B, d_in * d_out]
        W = W.reshape(B, d_in, d_out)  # [B, d_in, d_out]

        if len(input.shape) == 2:
            # [B, 1,    d_in]
            # [B, d_in, d_out]
            output = torch.matmul(input.unsqueeze(1), W)
            output = output.squeeze(1)  # [B, d_out]
        elif len(input.shape) == 3:
            # [B, N,    d_in]
            # [B, d_in, d_out]
            output = torch.matmul(input, W)  # [B, N, d_out]

        if self.has_bias:
            b = F.linear(seed, self.W_bias)  # [B, d_out]
            if len(input.shape) == 3:
                b = b.unsqueeze(1)
            output += b

        # output = F.relu(output)
        # output = self.layer_norm(output)
        return output


class SoftlyMaskedAttention(nn.Module):

    def __init__(self, d_q, d_k, d_v, h):
        super().__init__()
        self.W_q = nn.Sequential(
            nn.Linear(d_q, h),
            # nn.ReLU(),
            # nn.Linear(h, h),
            # nn.ReLU()
        )
        self.W_k = nn.Sequential(
            nn.Linear(d_k, h),
            # nn.ReLU(),
            # nn.Linear(h, h),
            # nn.ReLU()
        )
        self.W_v = nn.Sequential(
            nn.Linear(d_k, h),
            nn.ReLU(),
            nn.Linear(h, d_v),
            nn.ReLU()
        )
        self.layer_norm = nn.LayerNorm(h)

    @staticmethod
    def attention(query, key, value, mask_weights=None, dropout=None):
        """
        Copied and adjusted from the annotated transformer; https://nlp.seas.harvard.edu/2018/04/03/attention.html
        """
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask_weights is not None:
            scores += mask_weights.unsqueeze(1).log()

        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

    def forward(self, Q, K, mask_weights):
        """
        :param Q:  [B, Nq, d_q]
        :param K:  [B, Nk, d_k]
        :param mask_weights:  [B, Nk]
        :return:
        """
        Q_trans = self.layer_norm(self.W_q(Q))  # [B, Nq, h]
        K_trans = self.layer_norm(self.W_k(K))  # [B, Nk, h]
        V_trans = self.W_v(K)  # [B, Nk, v]

        att_output, _ = self.attention(Q_trans, K_trans, V_trans, mask_weights)  # [B, Nq, v]
        return att_output





class TransformerDecoder(nn.Module):

    @staticmethod
    def build(d_model=16,
              decoder_ffn_dim=32,
              decoder_layers=2,
              decoder_attention_heads=2,
              decoder_dropout=0):
        decoder_layer = nn.TransformerDecoderLayer(d_model, decoder_attention_heads, decoder_ffn_dim, decoder_dropout)
        transformer_decoder = nn.TransformerDecoder(decoder_layer, decoder_layers)

        return TransformerDecoder(transformer_decoder)

    def __init__(self, transformer_decoder, tokens_embedding=None):
        super().__init__()
        self.tokens_embedding = tokens_embedding
        self.transformer_decoder = transformer_decoder

    def forward(self, target, encoder_out, encoder_padding_mask=None):
        encoder_out = encoder_out.transpose(0, 1)
        target = target.transpose(0, 1)
        decoder_output = self.transformer_decoder(target, encoder_out, memory_key_padding_mask=encoder_padding_mask)
        decoder_output = decoder_output.transpose(0, 1)

        # logits = F.linear(decoder_output, self.tokens_embedding.weight)
        return decoder_output


class TransformerEncoder(nn.Module):

    @staticmethod
    def build(vocab,
              d_model=16,
              encoder_ffn_dim=32,
              encoder_layers=2,
              encoder_attention_heads=2,
              encoder_dropout=0):
        encoder_layer = nn.TransformerEncoderLayer(d_model, encoder_attention_heads, encoder_ffn_dim, encoder_dropout)
        transformer_encoder = nn.TransformerEncoder(encoder_layer, encoder_layers)

        return TransformerEncoder(transformer_encoder, vocab, d_model)

    def __init__(self, transformer_encoder, vocab, d_model, pos_dropout=.1):
        super().__init__()

        self.vocab = vocab
        V = len(vocab)
        tokens_embedding = Embedding(V, d_model, vocab.pad())
        self.tokens_embedding = tokens_embedding

        self.pos_encoder = PositionalEncoding(d_model, pos_dropout)
        self.transformer_encoder = transformer_encoder

        self.d_model = d_model

    def forward(self, tokens):
        prompt_embed = self.tokens_embedding(tokens) * math.sqrt(self.d_model)
        prompt_embed = self.pos_encoder(prompt_embed)
        x = prompt_embed

        # compute padding mask
        encoder_padding_mask = tokens.eq(self.vocab.pad_index)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        x = x.transpose(0, 1)
        encoder_output = self.transformer_encoder(x, src_key_padding_mask=encoder_padding_mask)  # [N_in, B, d]
        encoder_output = encoder_output.transpose(0, 1)

        return encoder_output, encoder_padding_mask


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1).to(device)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    # nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    return m