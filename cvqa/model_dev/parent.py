import torch
import torch.nn as nn
import torch.nn.functional as F

from cvqa.model_dev import programs, answer_model, f1_model
from cvqa.model_dev.misc import Vocabulary
from cvqa.model_dev.programs import ProgramsVocab, Seq2ConstTreeModel, Seq2VecsLSTM
from cvqa.utils import device


def default_args():
    return {
        'd_a': 4,
        'd_w': 16,
        'd_c': 16,
        'd_o': 16,
        'd_k': 8,
    }


def parse_dims_dict(args):
    return {
        'a': args['d_a'],
        'w': args['d_w'],
        'c': args['d_c'],
        'o': args['d_o'],
        'k': args['d_k'],
    }


class ContextModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.d = parse_dims_dict(args)
        self.layer_norm = nn.LayerNorm([self.d['o']])

    def forward(self, prompt, img):
        img = self.layer_norm(img)
        B, N_objs, _ = img.shape

        init_w = torch.ones(B, N_objs).to(device)
        no_answer = torch.zeros(B, self.d['a']).to(device)
        return {
            'X': img,
            'init_w': init_w,
            'no_answer': no_answer
        }


class ProgramSpec(object):

    def __init__(self, modules_dict):
        self.modules_dict = modules_dict
        self.vocab = ProgramsVocab(list(modules_dict.keys()))
        self.ids2modules = {self.vocab.encode_symbol(k): m for k, m in modules_dict.items()}


class MyModel(nn.Module):

    @staticmethod
    def build(args, prompt_vocab, answer_vocab):
        dims_dict = parse_dims_dict(args)

        modules_dict = {
            'A': answer_model.AnswerModule(dims_dict, answer_vocab),
            'F': f1_model.F1ModuleSimple(args)
        }
        program_spec = ProgramSpec(modules_dict)
        seq2tree = Seq2ConstTreeModel(program_spec.vocab, 'A ( F )')

        seeder_args = Seq2VecsLSTM.args(prompt_vocab, program_spec.vocab)
        seeder_model = Seq2VecsLSTM(seeder_args)

        context_model = ContextModel(args)

        return MyModel(seq2tree, seeder_model, context_model, program_spec)

    def __init__(self, seq2tree_model, seeder_model, context_model, program_spec):
        super().__init__()
        self.seq2tree_model = seq2tree_model
        self.seeder_model = seeder_model
        self.context_model = context_model
        self.program_spec = program_spec

    def forward(self, prompt, img):
        """
        :param prompt: B, N_prompt, d_w
        :param img: B, N_objs, d_o
        :return:
        """
        # 1) Encode prompt to tree structure
        program_tokens_batch = self.seq2tree_model.forward(prompt)

        # 2) Build and seed the tree.
        #    The assumption is all programs in the batch have the same structure, so we do tree building from the first sample.
        seeds = self.seeder_model(prompt, program_tokens_batch)
        seeds = seeds.transpose(0, 1)  # [B, N_p, d_z] --> [N_p, B, d_z] so each tree node will get a seed of shape [B, d_z]
        program_tokens = program_tokens_batch[0].detach().cpu().numpy()
        root_node, _ = programs.build_tree(program_tokens, seeds, 0, self.program_spec.vocab, self.program_spec.ids2modules)

        # 3) Run context model for cross module computations
        context = self.context_model(prompt, img)

        # 4) Execute the tree
        output = root_node.exec(context=context)[1]

        return output