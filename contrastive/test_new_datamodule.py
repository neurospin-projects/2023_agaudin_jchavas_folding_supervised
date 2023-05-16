import hydra

import numpy as np
import pandas as pd

from contrastive.data.utils import *
from contrastive.data.create_datasets import *


@hydra.main(config_name='config', config_path="configs")
def test_things(config):
    """subject_labels = read_labels(config.subject_labels_file,
                                 config.subject_column_name,
                                 config.label_names)

    print(subject_labels.head())

    output = extract_data_with_labels(
        config.numpy_all, subject_labels, config.crop_dir, config)

    print(output.keys())
    for key in output.keys():
        print(key)
        liste = output[key]
        for truc in liste:
            print(truc.shape)

    print(output['train'][0].head())
    print(output['test'][0].head())"""

    datasets = create_sets_with_labels(config)

    for key in datasets.keys():
        print(key)
        truc = datasets[key]
        print(truc)

    return 0


if __name__ == "__main__":
    test_things()
