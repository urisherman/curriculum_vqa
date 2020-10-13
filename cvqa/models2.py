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


USE_CONCEPT_SPACE = False


class MostBasicModel(nn.Module):

    # @staticmethod
    # def build(vocab,
    #           img_perceptor,

    def __init__(self, prompt_vocab, answer_vocab, img_preceptor, dims_dict):
        super().__init__()
        d = dims_dict
        self.dims_dict = dims_dict
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
            d['c'] = d['w']
            self.E_c = self.prompt_encoder.tokens_embedding
            self.W_w_op = None

        self.C_ops = nn_parameter(d['o'], d['c'], d['k'])
        self.m_f1 = F1Module(d, self.E_c, self.C_ops)
        self.m_ans = ExistsAnswerModule(d, answer_vocab)

        d['N_ops'] = 2
        self.E_ops = Embedding(d['N_ops'], d['w'])

        self.dropout_layer = nn.Dropout(p=0)
        self.layer_norm = nn.LayerNorm([d['w']])

    def bos_embeddings(self, B, N):
        bos_tensor = torch.ones(B, N, dtype=torch.int64).to(device) * self.prompt_vocab.bos_index
        bos_tensor = self.tokens_embedding(bos_tensor).transpose(0, 1)
        return bos_tensor

    def forward(self, prompt_tokens, img):
        B, N_prompt = prompt_tokens.shape
        prompt_encoded, prompt_pad_mask = self.prompt_encoder(prompt_tokens)

        X = self.img_perceptor(img)
        X = self.dropout_layer(X)

        op_inputs = self.E_ops.weight  # [N_ops, w]
        op_inputs = op_inputs.repeat(B, 1, 1)  # [B, N_ops, w]

        w_ops = self.op_decoder(op_inputs, prompt_encoded, prompt_pad_mask)
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


class AnswerModule(nn.Module):

    def __init__(self, dims_dict, vocab):
        super().__init__()
        d = dims_dict

        self.vocab = vocab
        self.dims_dict = dims_dict
        self.Q_ops = nn_parameter(d['o'], d['a'], d['c'])
        self.E_a = Embedding(len(vocab), d['a'])

    def forward(self, X, X_weights_in, question_op):
        """
        X: [B, N_o, d_o]
        X_weights_in: [B, N_o]
        question_op: [B, d_c]
        """
        W_answer = get_op(question_op, self.Q_ops)  # [o, a]

        weighted_X = torch.matmul(X_weights_in.unsqueeze(1), X).squeeze()  # [B, o]

        ans_vec = torch.matmul(weighted_X.unsqueeze(1), W_answer)  # [B, a]
        logits = F.linear(ans_vec.squeeze(), self.E_a.weight)
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

    def __init__(self, dims_dict, E_c, C_ops):
        super().__init__()
        self.dims_dict = dims_dict
        d = dims_dict

        self.W_kc = nn_parameter(d['k'], d['c'])
        self.C_ops = C_ops
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
        XP_res = self.cosim(X_c, P.unsqueeze(1))

        # # 3) Compute cosine similarity in concept distribution space (is there a better way to do this?)
        # X_c_probs = F.softmax(torch.matmul(X_c, self.E_c.weight.T), dim=-1)  # [B, N_o, N_c]
        # P_c_probs = F.softmax(torch.matmul(P, self.E_c.weight.T), dim=-1)  # [B, N_c]
        # XP_res = torch.sum(X_c_probs * P_c_probs.unsqueeze(1), dim=-1)  # [B, N_o], holds the probability every object passes the predicate

        #### Logical Not? Other Structures?

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