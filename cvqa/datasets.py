import os
import json
import urllib
import zipfile
from pathlib import Path
import shutil

import gdown
import torch

import torch.utils as torch_utils
import torch.nn.functional as F

from fairseq.data import Dictionary
from fairseq.tokenizer import tokenize_line

import torchvision as tv


DATASETS = {
    'NLVR': {
        'url': 'https://drive.google.com/uc?id=1cVUPwPYIwvHY_TQxRQ9sSvpDZdxzTRXz'
    },
    'basic_curriculum': {
        'url': 'https://drive.google.com/uc?id=14TuslNdx_cmQL_mH-_pRTWsRaO7Gwa4x'
    }
}


def download_if_needed(dataset_name, root, a_file):
    if not os.path.exists(a_file):
        shutil.rmtree(root, ignore_errors=True)
        os.makedirs(root)
        # print(f"Downloading {dataset_name} dataset...")
        dataset_zip = os.path.join(root, f'{dataset_name}.zip')
        gdown.download(DATASETS[dataset_name]['url'], dataset_zip, quiet=False)
        with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
            zip_ref.extractall(root)
        print(f'Dataset was downloaded successfully and extracted to {root}')


class LabelIndexer:
    def __init__(self):
        self.labels = set()

    def add(self, l):
        self.labels.add(l)

    def get_index(self):
        labels = list(self.labels)
        labels.sort()
        label_to_idx = {l: i for i, l in enumerate(labels)}
        return label_to_idx


class WithIndicesDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        s = self.base_dataset[index]
        s['index'] = index
        return s


class BaseDataset(torch_utils.data.Dataset):

    @staticmethod
    def load_train_dev():
        """
        TODO: implement and handle mutual cls_to_index etc
        :return:
        """
        pass

    def __init__(self, root_dir, samples, img_transform, vocabs_from=None, prompt_mode='natural', target_mode='natural', limit=None):
        """
        samples should be
        :param root_dir:
        :param samples: a list of {
              'prompt':
              'target':
              'image_path':
              'concept': <-- required only if prompt_mode == 'concept'
        }
        :param img_transform:
        :param vocab:
        :param prompt_mode:
        :param target_mode:
        :param limit:
        """
        self.root_dir = root_dir
        self.prompt_mode = prompt_mode
        self.target_mode = target_mode
        self.img_transform = img_transform

        is_build_vocab = vocabs_from is None
        if is_build_vocab:
            vocab = Dictionary()
            ans_vocab = Dictionary()
            struct_viz_vocab = Dictionary()
        else:
            vocab = vocabs_from.vocab
            ans_vocab = vocabs_from.ans_vocab
            struct_viz_vocab = vocabs_from.struct_viz_vocab

        classes = LabelIndexer()
        concepts = LabelIndexer()

        new_samples = []
        for i, sample in enumerate(samples):
            if limit is not None and i >= limit:
                break

            if self.prompt_mode == 'natural':
                sample['encoded_prompt'] = encode_line(sample['prompt'], vocab, add_if_not_exist=is_build_vocab)
            elif self.prompt_mode == 'concept' and 'concept' in sample:
                concepts.add(sample['concept'])
            else:
                raise ValueError(f'No such prompt_mode "{self.prompt_mode}"')

            if self.target_mode == 'natural':
                sample['encoded_target'] = encode_line(sample['target'], ans_vocab, add_if_not_exist=is_build_vocab)
            elif self.target_mode == 'class':
                classes.add(sample['target'])
            else:
                raise ValueError(f'No such target_mode "{self.target_mode}"')

            new_samples.append(sample)

        # vocab.finalize()

        cls_to_idx = classes.get_index()
        concept_to_idx = concepts.get_index()

        for sample in new_samples:
            if self.prompt_mode == 'concept' and 'concept' in sample:
                sample['encoded_prompt'] = concept_to_idx[sample['concept']]

            if self.target_mode == 'class':
                sample['encoded_target'] = cls_to_idx[sample['target']]

        self.samples = new_samples
        self.cls_to_idx = cls_to_idx
        self.concept_to_idx = concept_to_idx

        self.idx_to_cls = {v: k for k, v in cls_to_idx.items()}
        self.idx_to_concept = {v: k for k, v in concept_to_idx.items()}

        self.vocab = vocab
        self.ans_vocab = ans_vocab
        self.struct_viz_vocab = struct_viz_vocab
        if self.prompt_mode == 'natural':
            self.N_prompt = max(map(lambda s: len(s['encoded_prompt']), new_samples))
        else:
            self.N_prompt = 1

        if self.target_mode == 'natural':
            self.N_target = 1  # max(map(lambda s: len(s['encoded_target']), new_samples))
        else:
            self.N_target = 1

        if 'program_tokens' in new_samples[0]:
            self.N_program = max(map(lambda s: len(s['program_tokens']), new_samples)) + 1  # +1 for eos / bos

        self.teacher_forcing = True
        self.use_viz_rep = False
        self.debug_mode = False

    def __len__(self):
        return len(self.samples)

    def load_img(self, index):
        sample = self.samples[index]
        img = tv.datasets.folder.default_loader(os.path.join(self.root_dir, sample['image_path']))
        return self.img_transform(img)

    def pad_tokens(self, t, target_len):
        pad = target_len- len(t)
        padded = F.pad(t, (0, pad), value=self.vocab.pad_index)
        return padded

    def __getitem__(self, index):
        sample = self.samples[index]

        prompt = sample['encoded_prompt']
        target = sample['encoded_target']

        if self.prompt_mode == 'natural':
            pad = self.N_prompt - len(prompt)
            prompt = F.pad(prompt, (0, pad), value=self.vocab.pad_index)

        if self.target_mode == 'natural':
            pad = self.N_target - len(target)
            target = F.pad(target, (0, pad), value=self.vocab.pad_index)

            if target[-1] == self.vocab.eos():
                target = target[:-1]

        if not self.use_viz_rep:
            img = tv.datasets.folder.default_loader(os.path.join(self.root_dir, sample['image_path']))
            img = self.img_transform(img)
        else:
            img = sample['viz_rep']['encoded']

        ret = {
            'prompt': prompt,
            'img': img,
            'target': target
        }
        if 'program_tokens' in sample:
            program_tokens = sample['program_tokens']
            bos = torch.tensor([self.vocab.bos_index])
            eos = torch.tensor([self.vocab.eos_index])
            program_target_input = self.pad_tokens(torch.cat([bos, program_tokens]), self.N_program)
            program_target_output = self.pad_tokens(torch.cat([program_tokens, eos]), self.N_program)
            ret['target_program_in'] = program_target_input
            ret['target_program_out'] = program_target_output

        if 'attention_mask' in sample:
            ret['target_attention_mask'] = sample['attention_mask']
            ret['objects_mask'] = sample['viz_rep']['objects_mask']

        if self.debug_mode:
            if 'debug_info' in sample:
                ret['debug_info'] = sample['debug_info']
            else:
                debug_info = {
                    'prompt_text': sample['prompt'],
                    'target_text': sample['target'],
                    'struct_rep': sample['viz_rep']
                }
                ret['debug_info'] = debug_info

            if 'program_str' in sample:
                ret['debug_info']['program_str'] = sample['program_str']

        return ret

    def __repr__(self):
        S = len(self.samples)
        return f'Root: {self.root_dir} \n' \
               f'Samples: {S} (N_prompt={self.N_prompt}, N_target={self.N_target})\n' \
               f'Concepts: {len(self.concept_to_idx)} \n' \
               f'Classes: {len(self.cls_to_idx)} \n' \
               f'Prompt Vocab Tokens:{len(self.vocab)} \n' \
               f'Answer Vocab Tokens:{len(self.ans_vocab)} \n'


class Curriculum(BaseDataset):

    @staticmethod
    def from_samples(train_samples, dev_samples):
        train_samples = Curriculum.process(train_samples)
        dev_samples = Curriculum.process(dev_samples)

        train_dataset = BaseDataset('-', train_samples, None)
        dev_dataset = BaseDataset('-', dev_samples, None, vocabs_from=train_dataset)

        train_dataset.use_viz_rep = True
        dev_dataset.use_viz_rep = True

        # train_dataset.debug_mode = True
        # dev_dataset.debug_mode = True
        return train_dataset, dev_dataset

    @staticmethod
    def load_train_dev(root, struct_viz=True):
        train_dataset = Curriculum(root, 'train')
        dev_dataset = Curriculum(root, 'dev', vocabs_from=train_dataset)

        if struct_viz:
            train_dataset.use_viz_rep = True
            dev_dataset.use_viz_rep = True

        return train_dataset, dev_dataset

    def __init__(self, root, split='train', vocabs_from=None, prompt_mode='natural', target_mode='natural', limit=None):
        root_dir = os.path.join(root, split)
        ds_file = os.path.join(root, split, 'dataset.json')

        download_if_needed('basic_curriculum', root, ds_file)

        with open(ds_file) as data:
            samples = json.load(data)

        Curriculum.process(samples)

        img_transform = tv.transforms.Compose([
            tv.transforms.Resize(256),
            tv.transforms.CenterCrop(224),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        super().__init__(root_dir, samples, img_transform, vocabs_from=vocabs_from,
                         prompt_mode=prompt_mode, target_mode=target_mode, limit=limit)

    @staticmethod
    def process(samples):
        for sample in samples:
            if 'question' in sample:
                sample['prompt'] = sample['question']
            if 'answer' in sample:
                sample['target'] = sample['answer']

        return samples


class CLEVR(BaseDataset):

    @staticmethod
    def load_train_dev(root, samples_filter=None, struct_viz=True, d_img=24):
        ds_train = CLEVR(root, 'train', samples_filter=samples_filter)
        ds_dev = CLEVR(root, 'val', vocabs_from=ds_train, samples_filter=samples_filter)

        if struct_viz:
            ds_train.use_viz_rep = True
            ds_dev.use_viz_rep = True
            CLEVR.add_encoded_viz(ds_train.samples + ds_dev.samples, d_img=d_img)

        return ds_train, ds_dev

    @staticmethod
    def add_encoded_viz(samples, d_img=24):
        import itertools
        from tqdm import tqdm
        import torch.nn.functional as F
        from cvqa.curriculum import VizEncoder

        concepts = {
            'color': set(),
            'shape': set(),
            'material': set(),
            'size': set()
        }

        for o in itertools.chain(*(s['scene']['objects'] for s in samples)):
            for k in concepts:
                concepts[k].add(o[k])
        concepts = {k: list(v) for k, v in concepts.items()}

        N_max_objs = max(map(lambda s: len(s['scene']['objects']), samples))

        vizenc = VizEncoder(concepts, numeric_fields=['3d_coords'], d_img=d_img-9)

        for s in tqdm(samples):
            N_objs = len(s['scene']['objects'])
            encoded_viz = vizenc.encode(s['scene'])
            encoded_viz = F.pad(encoded_viz, [0, 0, 0, N_max_objs - N_objs], value=0)
            s['viz_rep']['encoded'] = encoded_viz

    @staticmethod
    def tokenize_program(prog_str):
        import re
        return [x for x in re.compile('([().,\\s])').split(prog_str) if x.strip() != '']

    @staticmethod
    def build_prog_str(prog):
        answer_op = prog[-1]
        answer_op_inputs = []
        for i in answer_op['inputs']:
            # generate obj pipeline by rolling backwards
            pipe = []
            curr_line = prog[i]
            while True:
                func_args = curr_line["value_inputs"]
                if len(func_args) > 0:
                    func_args = f"'{func_args[0]}'"
                else:
                    func_args = ''
                pipe.append(curr_line['function'] + f'({func_args})')
                if len(curr_line['inputs']) == 0:
                    break
                else:
                    curr_line = prog[curr_line['inputs'][0]]
            answer_op_inputs.append('.'.join(pipe[::-1]))

        output = answer_op['function'] + '(' + ', '.join(answer_op_inputs) + ')'
        return output

    def __init__(self, root, split='train', samples_filter=None, vocabs_from=None, parse_programs=True, limit=None):
        questions_path = f'{root}/questions/CLEVR_{split}_questions.json'
        scenes_path = f'{root}/scenes/CLEVR_{split}_scenes.json'

        scenes_data = None
        with open(scenes_path) as data:
            scenes_data = json.load(data)['scenes']

        questions_file = os.path.join(root, questions_path)
        samples = None
        with open(questions_file) as data:
            samples = json.load(data)['questions']

        if vocabs_from is not None:
            programs_vocab = vocabs_from.programs_vocab
        else:
            programs_vocab = Dictionary()

        final_samples = []
        for sample in samples:
            sample['prompt'] = sample['question']
            sample['target'] = sample['answer']
            img_idx = sample['image_index']
            scene = scenes_data[img_idx]
            sample['image_path'] = os.path.join('images', scene['image_filename'])
            sample['viz_rep'] = self.scene_to_canonical_rep(scene)
            sample['scene'] = scene
            if parse_programs:
                prog_str = CLEVR.build_prog_str(sample['program'])
                sample['program_str'] = prog_str
                program_tokens = []
                for token in CLEVR.tokenize_program(prog_str):
                    program_tokens.append(programs_vocab.add_symbol(token))
                sample['program_tokens'] = torch.tensor(program_tokens)

            if samples_filter is None or samples_filter(sample):
                final_samples.append(sample)

        self.programs_vocab = programs_vocab
        img_transform = tv.transforms.Compose([
            tv.transforms.Pad((0, 150), fill=300, padding_mode='constant'),
            tv.transforms.Resize(224),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        super().__init__(os.path.join(root, split), final_samples, img_transform,
                         vocabs_from=vocabs_from, prompt_mode='natural', target_mode='natural', limit=limit)

    def scene_to_canonical_rep(self, scene):
        objs = []
        for obj in scene['objects']:
            objs.append({
                'tokens': obj['size'] + ' ' + obj['color'] + ' ' + obj['material'] + ' ' + obj['shape'],
                'numerics': obj['3d_coords']
            })

        return {
            'objects': objs
        }


class NLVR(BaseDataset):

    def __init__(self, root, split='train', vocabs_from=None, target_mode='natural', limit=None, download=False):
        ds_file = os.path.join(root, split, f'{split}.json')
        if download:
            download_if_needed('NLVR', root, ds_file)

        samples = []
        with open(ds_file) as data:
            for line in data:
                sample = json.loads(line)
                samples.append(sample)
                sample['prompt'] = sample['sentence']
                sample['target'] = sample['label']
                img_id = sample['identifier']
                img_file = f'{split}-{img_id}-0.png'
                sample['image_path'] = os.path.join('images', sample['directory'], img_file)

        img_transform = tv.transforms.Compose([
            tv.transforms.Pad((0, 150), fill=300, padding_mode='constant'),
            tv.transforms.Resize(224),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        super().__init__(os.path.join(root, split), samples, img_transform,
                         vocabs_from=vocabs_from, prompt_mode='natural', target_mode=target_mode, limit=limit)


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, samples):
        vocab = Dictionary()
        self.vocab = vocab
        self.samples = samples
        for s in samples:
            s['encoded_prompt'] = encode_line(s['prompt'], vocab)
            s['encoded_target'] = encode_line(s['target'], vocab)

        self.N_prompt = max(map(lambda s: len(s['encoded_prompt']), samples))
        self.N_target = 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        prompt = sample['encoded_prompt']
        target = sample['encoded_target']

        pad = self.N_prompt - len(prompt)
        prompt = F.pad(prompt, (0, pad), value=self.vocab.pad_index)

        pad = self.N_target - len(target)
        target = F.pad(target, (0, pad), value=self.vocab.pad_index)
        if target[-1] == self.vocab.eos():
            target = target[:-1]

        return {
            'prompt': prompt,
            'target': target
        }


# TODO: Sort this out - what is the proper way to encode and build the dictionary? Note - fairseq transformer expects int64 type token ids.
def encode_line(line, vocab, add_if_not_exist=True, consumer=None, append_eos=True, reverse_order=False):
    """
    Copied from fairseq.data.Dictionary and changed ids tensor type to Long (==int64)
    :param line:
    :param vocab:
    :param add_if_not_exist:
    :param consumer:
    :param append_eos:
    :param reverse_order:
    :return:
    """
    words = tokenize_line(line)
    if reverse_order:
        words = list(reversed(words))
    nwords = len(words)
    ids = torch.LongTensor(nwords + 1 if append_eos else nwords)

    for i, word in enumerate(words):
        if add_if_not_exist:
            idx = vocab.add_symbol(word)
        else:
            idx = vocab.index(word)
        if consumer is not None:
            consumer(word, idx)
        ids[i] = idx
    if append_eos:
        ids[nwords] = vocab.eos_index
    return ids
