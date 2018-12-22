import gzip
import os
import pickle

import requests


def _download_files_if_nonexistent(path, file_links):
    for file_link in file_links:
        file_name = file_link.split('/')[-1].split('?')[0]
        file_path = os.path.join(path, file_name)
        if not os._exists(file_path):
            _download_file(file_link, file_path)


def _download_file(file_link, file_path):
    response = requests.get(file_link)
    open(file_path, 'wb').write(response.content)


def _load_gzipped_array(file_name):
    return pickle.loads(gzip.decompress(open(file_name, 'rb').read()))
