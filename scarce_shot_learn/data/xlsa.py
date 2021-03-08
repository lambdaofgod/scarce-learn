"""
data.xlsa

Utilities for loading data from "Zero-Shot Learning - A Comprehensive Evaluation of the Good, the Bad and the Ugly"
See https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/zero-shot-learning/zero-shot-learning-the-good-the-bad-and-the-ugly/
"""
import os
import zipfile

import numpy as np
import scipy.io

from .conf import FEW_SHOT_LEARN_PATH
from .util import _download_file
import pathlib


def _index_with_squeeze(splits, loc):
    return np.squeeze(splits[loc] - 1)


def _relabel(labels_seen, labels_split):
    i = 0
    for labels in labels_seen:
        labels_split[labels_split == labels] = i
        i += 1
    return labels_split


def validate_data_existence(base_path, path):
    full_path = os.path.join(base_path, path)
    data_path = os.path.join(base_path, 'data')
    if not os.path.exists(full_path):
        pathlib.Path(data_path).mkdir(parents=True, exist_ok=True)
        zip_path = os.path.join(data_path, 'xlsa17.zip')
        if not os.path.exists(zip_path):
            print('Dataset not found, downloading xlsa17.zip...')
            _download_file('http://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip', zip_path)
        print('Unzipping xlsa17 datasets...')
        file = zipfile.ZipFile(zip_path)
        file.extractall(data_path)


def select_split(split_key, feature_data, attribute_data):
    labels = feature_data['labels']
    split_labels = labels[_index_with_squeeze(attribute_data, split_key)]
    split_labels_deduped = np.unique(split_labels)
    split_labels = np.ravel(_relabel(split_labels_deduped, split_labels))
    X = feature_data['features']
    split_features = X[:,_index_with_squeeze(attribute_data, split_key)]
    attribute_features = attribute_data['att']
    split_attribute_features = attribute_features[:,(split_labels_deduped)-1]
    return split_features.T, split_attribute_features.T, split_labels


def load_dataset(path):
    validate_data_existence(FEW_SHOT_LEARN_PATH, path)

    data_path = os.path.join(FEW_SHOT_LEARN_PATH, path)
    resnet_features_path = os.path.join(data_path, 'res101.mat')
    att_split_path = os.path.join(data_path, 'att_splits.mat')

    res101 = scipy.io.loadmat(resnet_features_path)
    att_splits = scipy.io.loadmat(att_split_path)
    return {
        'train': select_split('train_loc', res101, att_splits),
        'val': select_split('val_loc', res101, att_splits),
        'trainval': select_split('trainval_loc', res101, att_splits),
        'test': select_split('test_unseen_loc', res101, att_splits)
    }
