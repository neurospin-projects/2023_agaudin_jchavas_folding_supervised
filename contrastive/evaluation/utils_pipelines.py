# File that contains auxiliary functions needed for either (or both) embeddings_pipeline
# and supervised_pipeline

import os
import yaml

def get_save_folder_name(datasets, short_name):
    """Creates a file name from the names of the target datasets or an
    explicit option.
    
    Arguments:
        - datasets: list of str. Contains the names of the target datasets.
        - short_name: str or None. Default answer by the algorithm if not None."""
    if short_name is not None:
        folder_name = short_name
    else:
        folder_name = ''
        for dataset in datasets:
            folder_name = folder_name + dataset + '_'
        folder_name = folder_name[:-1]  # remove the last _
    
    return folder_name


def change_config_datasets(config, new_datasets):
    """Replace the 'dataset' entry of a config 
    with the new target datasets. Works in place.
    
    Arguments:
        - config: a config object (omegaconf).
        - new_datasets: list of str, each corresponding to the name 
        of a target yaml file."""
    
    # replace the datasets
    # first, remove the keys of the older datasets
    config['dataset'] = {}

    # add the ones of the target datasets
    for dataset in new_datasets:
        with open(f'./configs/dataset/{dataset}.yaml', 'r') as file:
            dataset_yaml = yaml.load(file, yaml.FullLoader)
        config.dataset[dataset] = {}
        for key in dataset_yaml:
            config.dataset[dataset][key] = dataset_yaml[key]


def change_config_label(config, new_label):
    """Replace the 'label' entry of a config 
    with the new target label. Works in place.
    
    Arguments:
        - config: a config object (omegaconf).
        - new_label: str corresponding to the name 
        of a target yaml file."""
    
    # remove the keywords of the old label
    if 'label_names' in config.keys():
        current_label = config.label_names[0]
        with open(f'./configs/label/{current_label}.yaml', 'r') as file:
            old_label_yaml = yaml.load(file, yaml.FullLoader)
        for key in old_label_yaml:
            config.pop(key)

    # add the ones of the target label
    with open(f'./configs/label/{new_label}.yaml', 'r') as file:
        label_yaml = yaml.load(file, yaml.FullLoader)
    for key in label_yaml:
        config[key] = label_yaml[key]


def save_used_datasets(save_path, datasets):
    """Save the datasets given in order in a .txt file. Used in embeddings and supervised
    pipelines to know which datasets have been used for the results generation.
    
    Arguments:
        - save_path: str. Where the txt file is saved. Either the name 
        of the directory or directly the full path with the file name.
        - datasets: list of str. Name of the used datasets."""
    # if save path is only a directory
    if os.path.isdir(save_path):
        # add the actual file name at the end
        save_path = os.path.join(save_path, 'datasets_used.txt')
    
    # preprocess datasets
    datasets = list(datasets)

    with open(save_path, 'w') as file:
        for dataset in datasets:
            file.write(dataset)
            file.write('\n')


def save_used_label(save_path, config):
    """Save the label used for classification in a .txt file. Used both in 
    supervised and embeddings pipelines.

    Arguments:
        - save_path: str. Where the txt file is saved. Either the name 
        of the directory or directly the full path with the file name.
        - config: omegaconf object. Contains the label used for test 
        classification."""
    
    # if save path is only a directory
    if os.path.isdir(save_path):
        # add the actual file name at the end
        save_path = os.path.join(save_path, 'label_used.txt')
    
    # get label from config
    label = config.label_names[0]

    with open(save_path, 'w') as file:
        file.write(label)