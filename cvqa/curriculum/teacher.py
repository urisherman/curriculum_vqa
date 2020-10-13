import os
import pathlib
import shutil

from cvqa.curriculum import VQAInstanceDistribution

ws_root = pathlib.Path(__file__).parent.parent.parent.absolute()
data_bin_root = os.path.join(ws_root, f'data-bin')


def lesson_root(lesson):
    return os.path.join(data_bin_root, f'curriculum_{lesson}')


def make_dataset_split(root, split, vqa_dist, images=10, prompts_per_image=3):
    split_root = os.path.join(root, split)
    shutil.rmtree(split_root, ignore_errors=True)
    vqa_dist.generate_dataset(split_root, images=images, prompts_per_image=prompts_per_image)


def make_dataset(root, vqa_dist, images=100, prompts_per_image=4):
    make_dataset_split(root, 'train', vqa_dist, images=images, prompts_per_image=prompts_per_image)
    make_dataset_split(root, 'dev', vqa_dist, images=int(images*.25), prompts_per_image=prompts_per_image)


if __name__ == '__main__':
    lesson = '1'

    root = lesson_root(lesson)

    vqa_dist = VQAInstanceDistribution()
    make_dataset_split(root, 'train', images=500, prompts_per_image=10)
    make_dataset_split(root, 'dev', images=50, prompts_per_image=10)

    shutil.make_archive(f'curriculum_{lesson}', 'zip', root_dir=root)

    f = f'curriculum_{lesson}.zip'
    shutil.move(f, os.path.join(data_bin_root, f))