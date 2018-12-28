import glob
import os
import pathlib

import numpy as np
from skimage.io import imread
from skimage.transform import resize as imresize

from .conf import FEW_SHOT_LEARN_PATH

CUB_DATA_PATH = os.path.join(FEW_SHOT_LEARN_PATH, 'data', 'cub')


def _get_image_basepaths(split_text_filename):
    image_paths = open(os.path.join(CUB_DATA_PATH, 'lists', split_text_filename)).readlines()
    return [os.path.basename(path[:-1]) for path in image_paths]


def _load_images_with_labels(image_paths, image_size):
    images = np.stack([
        imresize(imread(path), image_size)
        for path in image_paths
    ])
    labels = np.array([
        path.split('/')[-2]
        for path in image_paths
    ])

    return images, labels


def load_split(images_dir, split_image_filenames):
    image_paths = glob.glob(os.path.join(images_dir, '*/**'), recursive=True)
    split_image_paths = set(_get_image_basepaths(split_image_filenames))
    return [image_path for image_path in image_paths if os.path.basename(image_path) in split_image_paths]


def load_cub(image_size=(128, 128)):
    pathlib.Path(CUB_DATA_PATH).mkdir(parents=True, exist_ok=True)
    images_dir = os.path.join(CUB_DATA_PATH, 'images')
    train_image_paths = load_split(images_dir, 'train.txt')
    test_image_paths = load_split(images_dir, 'test.txt')

    return {
        'train': _load_images_with_labels(train_image_paths, image_size),
        'test': _load_images_with_labels(test_image_paths, image_size)
    }
