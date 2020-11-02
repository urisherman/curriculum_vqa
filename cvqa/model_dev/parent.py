import torch
import torch.nn as nn
import torch.nn.functional as F

from cvqa.model_dev import programs, answer_model, f1_model
from cvqa.model_dev.blocks import ContextModel
from cvqa.model_dev.misc import Vocabulary
from cvqa.model_dev.programs import ProgramsVocab, Seq2ConstTreeModel, Seq2VecsLSTM, TransformerSeederModel, ProgramSpec
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



class MultiModule(nn.Module):

    def __init__(self, d_seed, sub_modules):
        super().__init__()
        self.sub_modules = sub_modules
        self.N_modules = len(sub_modules)
        self.W = nn.Linear(d_seed, self.N_modules)

    def forward(self, inputs, seed, context):
        weights = F.softmax(self.W(seed), dim=-1)
        outputs = []
        for i, m in enumerate(self.sub_modules):
            m_outputs = m(inputs, seed, context)
            for j, o in enumerate(m_outputs):

                # [B] * [B, *]
                o_weighted = (weights[:, i]*o.T).T
                if j >= len(outputs):
                    outputs.append(o_weighted)
                else:
                    outputs[j] += o_weighted
        return outputs


class MyModel(nn.Module):

    @staticmethod
    def build(args, prompt_vocab, answer_vocab):
        dims_dict = parse_dims_dict(args)

        modules_dict = {
            'A': answer_model.AnswerModule(dims_dict, answer_vocab).to(device),
            'F': f1_model.F1ModuleSimple(args).to(device)
        }
        program_spec = ProgramSpec(modules_dict)
        seq2tree = Seq2ConstTreeModel(program_spec.vocab, 'A ( F )')

        # seeder_args = Seq2VecsLSTM.args(prompt_vocab, program_spec.vocab)
        # seeder_args['d_target'] = args['d_w']
        # seeder_model = Seq2VecsLSTM(seeder_args)

        seeder_model = TransformerSeederModel(prompt_vocab, program_spec.vocab, args)
        context_model = ContextModel(args)

        return MyModel(seq2tree, seeder_model, program_spec, context_model)

    def __init__(self, seq2tree_model, seeder_model, program_spec, context_model=None):
        super().__init__()
        self.seq2tree_model = seq2tree_model
        self.seeder_model = seeder_model
        self.program_spec = program_spec
        self.context_model = context_model

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