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

        self.teacher_forcing = True
        self.use_viz_rep = False
        self.debug_mode = False

    def __len__(self):
        return len(self.samples)

    def load_img(self, index):
        sample = self.samples[index]
        img = tv.datasets.folder.default_loader(os.path.join(self.root_dir, sample['image_path']))
        return self.img_transform(img)

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
            'target': target,
            'target_attention_mask': sample['attention_mask']
        }

        if self.debug_mode:
            ret['prompt_text'] = sample['prompt']
            ret['target_text'] = sample['target']
            ret['struct_rep'] = sample['viz_rep']

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

    def __init__(self, root, split='train', vocabs_from=None, target_mode='natural', limit=None):
        questions_path = f'{root}/questions/CLEVR_{split}_questions.json'
        scenes_path = f'{root}/scenes/CLEVR_{split}_scenes.json'

        scenes_data = None
        with open(scenes_path) as data:
            scenes_data = json.load(data)['scenes']

        questions_file = os.path.join(root, questions_path)
        samples = None
        with open(questions_file) as data:
            samples = json.load(data)['questions']

        for sample in samples:
            sample['prompt'] = sample['question']
            sample['target'] = sample['answer']
            img_idx = sample['image_index']
            scene = scenes_data[img_idx]
            sample['image_path'] = os.path.join('images', scene['image_filename'])
            sample['viz_rep'] = self.scene_to_canonical_rep(scene)

        img_transform = tv.transforms.Compose([
            tv.transforms.Pad((0, 150), fill=300, padding_mode='constant'),
            tv.transforms.Resize(224),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        super().__init__(os.path.join(root, split), samples, img_transform,
                         vocabs_from=vocabs_from, prompt_mode='natural', target_mode=target_mode, limit=limit)

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
