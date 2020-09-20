import os
import pathlib
import shutil

from cvqa.curriculum import VQAInstanceDistribution

ws_root = pathlib.Path(__file__).parent.parent.parent.absolute()
data_bin_root = os.path.join(ws_root, f'data-bin')


def lesson_root(lesson):
    return os.path.join(data_bin_root, f'curriculum_{lesson}')


def make_dataset(root, split, images=10, prompts_per_image=3):
    split_root = os.path.join(root, split)
    shutil.rmtree(split_root, ignore_errors=True)
    vqa_dist = VQAInstanceDistribution()
    vqa_dist.generate_dataset(split_root, images=images, prompts_per_image=prompts_per_image)


if __name__ == '__main__':
    lesson = '1'

    root = lesson_root(lesson)
    make_dataset(root, 'train', images=100, prompts_per_image=3)
    make_dataset(root, 'dev', images=10, prompts_per_image=3)

    shutil.make_archive(f'curriculum_{lesson}', 'zip', root_dir=root)

    f = f'curriculum_{lesson}.zip'
    shutil.move(f, os.path.join(data_bin_root, f))