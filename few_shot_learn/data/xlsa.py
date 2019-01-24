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


def validate_data_existence(path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    if not os.path.exists(path):
        temporary_path = os.path.join(path, 'xlsa17.zip')
        if not os.path.exists(temporary_path):
            _download_file('http://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip', temporary_path)
        file = zipfile.ZipFile(temporary_path)
        file.extractall(path)


def select_split(split_key, feature_data, attribute_data):
    labels = feature_data['labels']
    split_labels = labels[_index_with_squeeze(attribute_data, split_key)]
    split_labels_deduped = np.unique(split_labels)
    split_labels = _relabel(split_labels_deduped, split_labels)
    X = feature_data['features']
    split_features = X[:,_index_with_squeeze(attribute_data, split_key)]
    attribute_features = attribute_data['att']
    split_attribute_features = attribute_features[:,(split_labels_deduped)-1]
    return split_features.T, split_attribute_features.T, split_labels


def load_dataset(path):
    data_path = os.path.join(FEW_SHOT_LEARN_PATH, path)
    validate_data_existence(data_path)

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
