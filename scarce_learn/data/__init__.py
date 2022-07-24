import os
from .cub import load_cub
from .xlsa import load_dataset


def load_awa2():
    """
    Animals with Attributes 2 Dataset
    returns dict split: (X_split, attribute_features_split, labels_split)
    """
    awa_path = os.path.join("data", "xlsa17", "data", "AWA2")
    return load_dataset(awa_path)
