import torch
import torch.nn as nn
import torch.nn.functional as F

from cvqa.model_dev.misc import Vocabulary
from cvqa.utils import device


class Seq2ConstTreeModel(object):

    def __init__(self, prog_vocab, fixed_prog_str):
        self.prog_vocab = prog_vocab
        self.fixed_tree_tokens = torch.tensor(prog_vocab.encode(tokenize(fixed_prog_str))).to(device)

    def forward(self, prompt):
        B, N = prompt.shape
        return self.fixed_tree_tokens.repeat(B, 1)


class Seq2VecsLSTM(nn.Module):

    @staticmethod
    def args(input_vocab, decoder_vocab):
        return {
            'V_input': len(input_vocab),
            'V_decoder': len(decoder_vocab),
            'd_input': 16,
            'd_decoder': 4,
            'd_target': 16,
            'd_h': 16,
            'num_layers': 2
        }

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.d_h = args['d_h']
        self.num_layers = args['num_layers']

        self.E_input = nn.Embedding(args['V_input'], args['d_input'])
        self.encoder_lstm = nn.LSTM(args['d_input'], args['d_h'], num_layers=args['num_layers'])
        self.decoder_lstm = nn.LSTM(args['d_decoder'], args['d_h'], num_layers=args['num_layers'])
        self.W_o = nn.Linear(args['d_h'], args['d_target'])
        self.E_decoder = nn.Embedding(args['V_decoder'], args['d_decoder'])

    def init_hiddens(self, B):
        h_0 = torch.zeros(self.num_layers, B, self.d_h).to(device)
        c_0 = torch.zeros(self.num_layers, B, self.d_h).to(device)
        return h_0, c_0

    def forward(self, X, decoder_input, decoder_hidden_state=None):
        """
        X: [B, N_seq]
        target: [B, N_seq_t]

        returns logits for target tokens: [B, N_seq_t, V_target]
        """
        B, N_seq = X.shape
        if decoder_hidden_state is None:
            X_embed = self.E_input(X)
            X_embed = X_embed.transpose(0, 1)  # [N_seq, B, d_input]

            h_0, c_0 = self.init_hiddens(B)
            encoder_out, (enc_h_n, enc_c_n) = self.encoder_lstm(X_embed, (h_0, c_0))
            decoder_hidden_state = (enc_h_n, enc_c_n)

        decoder_input_embed = self.E_decoder(decoder_input)
        decoder_input_embed = decoder_input_embed.transpose(0, 1)
        decoder_out, decoder_hidden_state = self.decoder_lstm(decoder_input_embed, decoder_hidden_state)

        # decoder_out: [N_seq_t, B, d_h]
        decoder_out = decoder_out.transpose(0, 1)
        # Add here decoder_out.att( encoder hiddens )  --> Section 3.3 in Dong & Lapata
        pred_target_vecs = self.W_o(decoder_out)
        # logits = F.linear(pred_target_vecs, self.E_target.weight)

        return pred_target_vecs


class ExecTreeNode(object):

    def __init__(self, module, module_id, arg_nodes):
        self.module = module
        self.module_id = module_id
        self.arg_nodes = arg_nodes
        self.seed = None

    def set_seed(self, seed):
        self.seed = seed

    def exec(self, context=None):
        args = []
        for arg_node in self.arg_nodes:
            args.append(arg_node.exec(context=context))

        output = self.module(args, seed=self.seed, context=context)
        return output

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        args_str = ', '.join([str(a) for a in self.arg_nodes])
        return f'{self.module_id}({args_str})'


class ProgramsVocab(Vocabulary):

    def __init__(self, module_symbols):
        super().__init__()
        self.module_symbols = module_symbols
        self.module_idxs = self.encode(module_symbols)
        self.start_subtree = self.encode_symbol('(')
        self.end_subtree = self.encode_symbol(')')
        self.sibling_symbol = self.encode_symbol(',')
        self.build()

    def is_module(self, idx):
        return idx in self.module_idxs


def tokenize(program_str):
    return program_str.strip().split(' ')


def build_tree(program, seeds, idx, vocab, ids2modules):
    if vocab.is_module(program[idx]):
        args = []
        next_idx = idx + 1
        if vocab.start_subtree == program[next_idx]:
            args, next_idx = build_tree(program, seeds, next_idx, vocab, ids2modules)

        module_id = program[idx]
        node = ExecTreeNode(ids2modules[module_id], module_id, args)
        node.set_seed(seeds[idx])
        return node, next_idx
    elif vocab.start_subtree == program[idx]:
        args = []
        next_idx = idx + 1
        while True:
            if vocab.end_subtree == program[next_idx]:
                break
            elif vocab.sibling_symbol == program[next_idx]:
                continue
            else:
                next_arg, next_idx = build_tree(program, seeds, next_idx, vocab, ids2modules)
                args.append(next_arg)
        return args, next_idx
    else:
        raise ValueError('Illegal state')


if __name__ == '__main__':
    prompt = 'How many trees in the image?'
    vocab = ProgramsVocab(['M'])
    program_tokens = vocab.encode(tokenize('M ( M )'))
    print(program_tokens)
    root_node, _ = build_tree(program_tokens, 0, vocab)
    print(root_node)