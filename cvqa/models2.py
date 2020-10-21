import torch
import math

from torch import nn
import torch.nn.functional as F

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def nn_parameter(*shape):
    W = nn.Parameter(torch.Tensor(*shape))
    nn.init.kaiming_uniform_(W, a=math.sqrt(5))
    return W


def default_args():
    return {
        'd_a': 4,
        'd_w': 16,
        'd_c': 16,
        'd_o': 16,
        'd_k': 8,
    }


def get_op(op_seed, op_mat):
    """
    op_seed: [B, d_op]
    op_mat: [d_in, d_out, d_op]
    """
    B, d_op = op_seed.shape
    d_in, d_out, d_op = op_mat.shape
    W = F.linear(op_seed, op_mat.reshape(-1, d_op))  # [B, d_out * d_in]
    return W.reshape(B, d_in, d_out)


def get_shapes(tensors, dims_dict):
    d_reversed = {v: k for k, v in dims_dict.items()}

    def shape(t):
        return list(map(lambda dim: d_reversed.get(dim, dim), t.shape))

    return [shape(t) for t in tensors]


def parse_dims_dict(args):
    return {
        'a': args['d_a'],
        'w': args['d_w'],
        'c': args['d_c'],
        'o': args['d_o'],
        'k': args['d_k'],
    }


USE_CONCEPT_SPACE = False


class MostBasicModel(nn.Module):

    # @staticmethod
    # def build(vocab,
    #           img_perceptor,

    def __init__(self, prompt_vocab, answer_vocab, img_preceptor, args):
        super().__init__()
        self.args = args

        d = parse_dims_dict(args)
        self.dims_dict = d
        self.prompt_vocab = prompt_vocab
        self.answer_vocab = answer_vocab

        self.img_perceptor = img_preceptor

        self.prompt_encoder = TransformerEncoder.build(
            prompt_vocab,
            d['w'],
            encoder_layers=1,
            encoder_attention_heads=1,
            encoder_dropout=0
        )

        self.op_decoder = TransformerDecoder.build(
            d['w'],
            decoder_ffn_dim=32,
            decoder_layers=1,
            decoder_attention_heads=1,
            decoder_dropout=0)

        if USE_CONCEPT_SPACE:
            self.E_c = Embedding(d['N_c'], d['c'])
            self.W_w_op = nn.Linear(d['w'], d['c'], bias=False)
        else:
            d['N_c'] = len(prompt_vocab)
            self.E_c = self.prompt_encoder.tokens_embedding
            self.W_w_op = None

        self.m_f1 = F1Module(args, self.E_c)
        self.m_ans = AnswerModule(d, answer_vocab)

        d['N_ops'] = 2
        self.E_ops = Embedding(d['N_ops'], d['w'])

        self.dropout_layer = nn.Dropout(p=0)
        self.layer_norm = nn.LayerNorm([d['w']])

    def bos_embeddings(self, B, N):
        bos_tensor = torch.ones(B, N, dtype=torch.int64).to(device) * self.prompt_vocab.bos_index
        bos_tensor = self.tokens_embedding(bos_tensor).transpose(0, 1)
        return bos_tensor

    def forward_train(self, sample):
        return self.forward(sample['prompt'], sample['img'], sample.get('attention_mask', None))

    def forward_test(self, sample, max_len=None):
        logits = self.forward(sample['prompt'], sample['img'], sample.get('attention_mask', None))
        _, y_pred = torch.max(logits, axis=-1)
        return logits, y_pred

    def forward(self, prompt_tokens, img, target_attention_mask=None):
        B, N_prompt = prompt_tokens.shape
        prompt_encoded, prompt_pad_mask = self.prompt_encoder(prompt_tokens)

        X = self.img_perceptor(img)
        # X = self.dropout_layer(X)

        op_inputs = self.E_ops.weight  # [N_ops, w]
        op_inputs = op_inputs.repeat(B, 1, 1)  # [B, N_ops, w]

        w_ops = self.op_decoder(op_inputs, prompt_encoded, prompt_pad_mask)  # [B, N_ops, w]
        w_ops = self.layer_norm(w_ops)

        if USE_CONCEPT_SPACE:
            ops = self.W_w_op(w_ops)  # [B, N_ops, c]
        else:
            ops = w_ops

        # ops = F.relu(ops)
        # ops = self.dropout_layer(ops)

        f1_op = ops[:, 0, :]
        ans_op = ops[:, 1, :]

        B, N_o, d_o = X.shape
        X_w = torch.ones(B, N_o, dtype=torch.float32).to(device)
        X_w = self.m_f1(X, X_w, f1_op)

        logits = self.m_ans(X, X_w, ans_op)
        return logits.unsqueeze(1)


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
        input: [B, d_in]
        """
        B, d_seed = seed.shape
        d_in, d_out, d_seed = self.W_op.shape
        W = F.linear(seed, self.W_op.reshape(-1, d_seed))  # [B, d_in * d_out]
        W = W.reshape(B, d_in, d_out)  # [B, d_in, d_out]

        # [B, 1, d_in] @ [B, d_in, d_out]
        output = torch.matmul(input.unsqueeze(1), W)
        output = output.squeeze(1)

        if self.has_bias:
            b = F.linear(seed, self.W_bias)  # [B, d_out]
            output += b

        # output = F.relu(output)
        # output = self.layer_norm(output)
        return output


class SoftlyMaskedAttention(nn.Module):

    def __init__(self, d_q, d_k, d_v, h):
        super().__init__()
        self.W_q = nn.Linear(d_q, h)
        self.W_k = nn.Linear(d_k, h)
        self.W_v = nn.Linear(d_k, d_v)

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
        Q = self.W_q(Q)  # [B, Nq, h]
        K = self.W_k(K)  # [B, Nk, h]
        V = self.W_v(K)  # [B, Nk, v]

        att_output, _ = self.attention(Q, K, V, mask_weights)  # [B, Nq, v]
        return att_output


class TransformerAnswerModule(nn.Module):

    def __init__(self, dims_dict, vocab):
        super().__init__()
        d = dims_dict

        self.vocab = vocab
        self.dims_dict = dims_dict
        self.m_att = SoftlyMaskedAttention(d['w'], d['o'], d['a'], 16)
        # self.W_ans = nn.Linear()
        self.E_a = Embedding(len(vocab), d['a'])

    def forward(self, X, X_weights_in, question_op):
        """
        X: [B, N_o, d_o]
        X_weights_in: [B, N_o]
        question_op: [B, d_c]
        """
        weighted_X = (X_weights_in.unsqueeze(2) * X).sum(axis=1)  # [B, o]

        ans_vec = self.CW_ans(weighted_X, question_op)  # [B, a]
        ans_vec = F.relu(ans_vec)
        ans_vec = self.W_ans2(ans_vec)

        # W_answer = get_op(question_op, self.Q_ops)  # [B, o, a]
        # ans_vec = torch.matmul(weighted_X.unsqueeze(1), W_answer)  # [B, a]

        logits = F.linear(ans_vec.squeeze(1), self.E_a.weight)  # [B, N_a]
        return logits


class AnswerModule(nn.Module):

    def __init__(self, dims_dict, vocab):
        super().__init__()
        d = dims_dict

        self.vocab = vocab
        self.dims_dict = dims_dict
        # self.Q_ops = nn_parameter(d['o'], d['a'], d['c'])
        self.CW_ans = CondLinear(d['o'], d['a'], d['c'], bias=True)
        # self.W_ans2 = nn.Sequential(
        #     nn.Linear(10, d['a']),
        #     nn.ReLU()
        # )
        self.E_a = Embedding(len(vocab), d['a'])

    def forward(self, X, X_weights_in, question_op):
        """
        X: [B, N_o, d_o]
        X_weights_in: [B, N_o]
        question_op: [B, d_c]
        """
        # weighted_X = (X_weights_in.unsqueeze(2) * X).sum(axis=1)  # [B, o]
        weighted_X = (X_weights_in.unsqueeze(2) * torch.ones_like(X)).sum(axis=1)  # [B, o]

        ans_vec = self.CW_ans(weighted_X, question_op)  # [B, a]
        # ans_vec = F.relu(ans_vec)
        # ans_vec = self.W_ans2(ans_vec)

        # W_answer = get_op(question_op, self.Q_ops)  # [B, o, a]
        # ans_vec = torch.matmul(weighted_X.unsqueeze(1), W_answer)  # [B, a]

        logits = F.linear(ans_vec.squeeze(1), self.E_a.weight)  # [B, N_a]
        return logits


class ExistsAnswerModule(nn.Module):

    def __init__(self, dims_dict, vocab):
        super().__init__()
        d = dims_dict

        self.vocab = vocab
        self.dims_dict = dims_dict
        # self.E_a = Embedding(len(vocab), d['a'])
        self.idx_TRUE = self.vocab.index('TRUE')
        self.idx_FALSE = self.vocab.index('FALSE')

    def forward(self, X, X_weights_in, question_op):
        """
        X: [B, N_o, d_o]
        X_weights_in: [B, N_o]
        question_op: [B, d_c]
        """
        B, N_o, d_o = X.shape
        a = X_weights_in.sum(axis=1)  # [B]
        logits = torch.zeros(B, len(self.vocab)).to(device)  # [B, N_a]
        logits[:, self.idx_TRUE] = a
        logits[:, self.idx_FALSE] = 1-a

        return logits


class F1Module(nn.Module):

    def __init__(self, args, E_c):
        super().__init__()
        self.args = args
        d = parse_dims_dict(args)
        self.dims_dict = d

        self.W_kc = nn_parameter(d['k'], d['c'])
        self.C_ops = nn_parameter(d['o'], d['c'], d['k'])
        self.E_c = E_c
        self.layer_norm = nn.LayerNorm([d['c']])
        self.cosim = nn.CosineSimilarity(2)

    def forward(self, X, X_weights_in, op):
        """
        X: [B, N_o, d_o]
        X_weights_in: [B, N_o]
        op: [B, d_c]
        """
        B, N_o, _ = X.shape
        d = self.dims_dict

        #### Concept Category Predicate
        # 1) op seed --> batch of concept ops
        P = op  # P for predicate
        P_k = F.linear(P, self.W_kc)  # [B, d_k] P_k for predicate category (eg color)
        P_k = get_op(P_k, self.C_ops)  # [B, d_o, d_c]
        P_k = self.layer_norm(P_k)

        # 2) Apply derived concept op to each object
        X_c = torch.matmul(X, P_k)  # [B, N_o, d_c]

        # 3) Compute cosine similarity
        XP_res = self.cosim(X_c, P.unsqueeze(1))  # [B, N_o]
        zeros = torch.zeros_like(XP_res).to(device)
        XP_res = torch.max(XP_res, zeros)

        # # 3) Compute cosine similarity in concept distribution space
        # X_c_probs = F.softmax(torch.matmul(X_c, self.E_c.weight.T), dim=-1)  # [B, N_o, N_c]
        # P_c_probs = F.softmax(torch.matmul(P, self.E_c.weight.T), dim=-1)  # [B, N_c]
        # XP_res = torch.sum(X_c_probs * P_c_probs.unsqueeze(1), dim=-1)  # [B, N_o], holds the probability every object passes the predicate

        #### Logical Not? Other Structures?

        return X_weights_in * XP_res


class F1CosineModule(nn.Module):

    def __init__(self, dims_dict, E_c):
        super().__init__()
        self.dims_dict = dims_dict
        d = dims_dict

        self.W_co = nn.Linear(d['c'], d['o'])
        # self.C_ops = C_ops
        # self.E_c = E_c
        # self.layer_norm = nn.LayerNorm([d['c']])
        self.cosim = nn.CosineSimilarity(2)

    def forward(self, X, X_weights_in, op):
        """
        X: [B, N_o, d_o]
        X_weights_in: [B, N_o]
        op: [B, d_c]
        """
        P_o = self.W_co(op)  # [B, d_o] predicate in object space
        XP_res = self.cosim(X, P_o.unsqueeze(1))

        return X_weights_in * XP_res


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
        pe = pe.unsqueeze(0).transpose(0, 1)
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