import zipfile
import os

import shutil
from .conf import FEW_SHOT_LEARN_PATH
from .util import _download_file


def _setup_awa2():
    data_path = os.path.join(FEW_SHOT_LEARN_PATH, 'data')
    awa_path = os.path.join(data_path, 'AWA2')

    if not os.path.exists(awa_path):
        temporary_path = os.path.join(data_path, 'xlsa17.zip')
        if not os.path.exists(temporary_path):
            _download_file(temporary_path, 'http://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip')
        file = zipfile.ZipFile(temporary_path)
        file.extractall(data_path)

        shutil.move(os.path.join(data_path, 'xlsa17', 'data', 'AWA2'), data_path)
