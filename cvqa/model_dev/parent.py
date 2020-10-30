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
        d = parse_dims_dict(args)
        self.layer_norm = nn.LayerNorm([d['o']])

    def forward(self, prompt, img):
        img = self.layer_norm(img)
        B, N_objs, _ = img.shape

        init_w = torch.ones(B, N_objs).to(device)
        return {
            'img': img,
            'init_w': init_w
        }


class MyModel(nn.Module):

    @staticmethod
    def build(prompt_vocab, answer_vocab):
        prog_vocab = ProgramsVocab(['A', 'F'])
        seq2tree = Seq2ConstTreeModel(prog_vocab, 'A ( F )')

        seeder_args = Seq2VecsLSTM.args(prompt_vocab, prog_vocab)
        seeder_model = Seq2VecsLSTM(seeder_args)

        args = default_args()
        dims_dict = parse_dims_dict(args)
        context_model = ContextModel(args)
        modules_repo = {
            prog_vocab.encode('A'): answer_model.AnswerModule(dims_dict, answer_vocab),
            prog_vocab.encode('F'): f1_model.F1ModuleSimple(args)
        }
        return MyModel(seq2tree, seeder_model, context_model, modules_repo)

    def __init__(self, seq2tree_model, seeder_model, context_model, modules_repo):
        super().__init__()
        self.seq2tree_model = seq2tree_model
        self.seeder_model = seeder_model
        self.context_model = context_model
        self.modules_repo = modules_repo

    def forward(self, prompt, img):
        """
        :param prompt: B, N_prompt, d_w
        :param img: B, N_objs, d_o
        :return:
        """
        # 1) encode prompt to tree structure
        program_tree_tokens = self.seq2tree_model(prompt)

        # 2) seed the tree
        seeds = self.seeder_model(prompt, program_tree_tokens)
        root_node, _ = programs.parse(program_tree_tokens, seeds, 0, self.seq2tree_model.prog_vocab, self.modules_repo)

        # 3) run context model for cross module computations
        context = self.context_model(prompt, img)

        # 4) execute the tree
        output = root_node.exec(context=context)

        return output