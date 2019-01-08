import zipfile
import os

import shutil

import scipy.io
import pandas as pd

from .conf import FEW_SHOT_LEARN_PATH
from .util import _download_file


def load_awa2_train_test():
    data_path = os.path.join(FEW_SHOT_LEARN_PATH, 'data')
    awa_path = os.path.join(data_path, 'AWA2')
    _setup_awa2(data_path)
    attribute_splits = scipy.io.loadmat(os.path.join(awa_path, 'att_splits.mat'))
    class_names = [cname[0] for cname in attribute_splits['allclasses_names'][:,0]]
    attributes_df = pd.DataFrame(attribute_splits['att'])
    attributes_df.columns = class_names

    resnet_features = scipy.io.loadmat(os.path.join(awa_path, 'res101.mat'))
    example_attributes = attributes_df.iloc[:, resnet_features['labels'][:,0]-1].T

    X_resnet = resnet_features['features'].T
    labels = resnet_features['labels'][:,0] - 1
    attributes = example_attributes

    train_indices = pd.read_csv(os.path.join(awa_path, 'trainvalclasses.txt')).values[:,0]

    is_train = attributes.index.isin(train_indices)
    X_train = X_resnet[is_train]
    X_test = X_resnet[~is_train]
    attributes_train = attributes[is_train]
    attributes_test = attributes[~is_train]
    labels_train = labels[is_train]
    labels_test = labels[~is_train]
    return (X_train, attributes_train, labels_train), (X_test, attributes_test, labels_test)


def _setup_awa2(data_path):
    awa_path = os.path.join(data_path, 'AWA2')

    if not os.path.exists(awa_path):
        temporary_path = os.path.join(data_path, 'xlsa17.zip')
        if not os.path.exists(temporary_path):
            _download_file(temporary_path, 'http://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip')
        file = zipfile.ZipFile(temporary_path)
        file.extractall(data_path)

        shutil.move(os.path.join(data_path, 'xlsa17', 'data', 'AWA2'), data_path)
