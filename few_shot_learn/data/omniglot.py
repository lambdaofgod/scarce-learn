import os
import pathlib

from .conf import FEW_SHOT_LEARN_PATH
from .util import _download_files_if_nonexistent, _load_gzipped_array


def load_omniglot(few_shot_learn_path=FEW_SHOT_LEARN_PATH):
    omniglot_path = os.path.join(few_shot_learn_path, 'data', 'omniglot')
    pathlib.Path(omniglot_path).mkdir(parents=True, exist_ok=True)
    file_link_template = 'https://github.com/lambdaofgod/few-shot-learn-data/blob/master/omniglot/omniglot_{}.pkl.gz?raw=true'
    file_links = [
        file_link_template.format(split)
        for split in ['train', 'test']
    ]
    _download_files_if_nonexistent(omniglot_path, file_links)
    return _load_omniglot_from_path(omniglot_path)


def _load_omniglot_from_path(omniglot_path):
    train_filename = os.path.join(omniglot_path, 'omniglot_train.pkl.gz')
    test_filename = os.path.join(omniglot_path, 'omniglot_test.pkl.gz')
    return _load_gzipped_array(train_filename), _load_gzipped_array(test_filename)

