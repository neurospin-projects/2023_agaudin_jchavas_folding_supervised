# File that contains auxiliary functions needed for either (or both) embeddings_pipeline
# and supervised_pipeline

import os
import yaml
import re
import pandas as pd

from contrastive.utils.logs import set_root_logger_level, set_file_logger

log = set_file_logger(__file__)



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


def detect_collision(run_path):
    """Detects if two models have been saved in the same folder during
    a wandb grid search. Returns True if it is the case.
    
    Arguments:
        - run_path: folder associated to a model to be inspected."""
    log_path = os.path.join(run_path, 'wandb')
    try:
        files = os.listdir(log_path)
    except:
        # not a model
        return False
    count = 0
    for file in files:
        if re.match(r'run*', file) is not None:
            count+=1
    if count > 1:
        return True
    return False


def detect_collisions(sweep_path):
    """Loops detect_collision over a folder at sweep_path.
    Prints all the folder names where there is a collision (two models
    saved in the same folder)."""
    runs = os.listdir(sweep_path)
    for run in runs:
        run_path = os.path.join(sweep_path, run)
        if os.path.isdir(run_path):
            if detect_collision(run_path):
                print(run)


def save_outputs_as_csv(outputs, filenames, labels, csv_path=None, verbose=False):
    """Save and returns outputs of a model to its canonical form from a tensor. If
    the given save_path doesn't exist, creates it.
    
    Arguments:
        - outputs: the output tensor to save.
        - filenames: the ordered subjects names associated to the outputs.
        - labels: the ordered true labels associated to the outputs.
        - csv_path: the path where to save the csv. If None, only returns
        the pandas dataframe.
        - verbose: verbose."""
    columns_names = ['dim'+str(i+1) for i in range(outputs.shape[1])]
    values = pd.DataFrame(outputs.numpy(), columns=columns_names)
    labels = pd.DataFrame(labels, columns=['labels']).astype(int)
    filenames = pd.DataFrame(filenames, columns=['ID'])
    df_outputs = pd.concat([labels, values, filenames], axis=1)
    
    # remove one copy each ID
    df_outputs = df_outputs.groupby('ID').mean()
    df_outputs.labels = df_outputs.labels.astype(int)

    if verbose:
        print("outputs:", df_outputs.iloc[:10, :])
        print("nb of elements:", df_outputs.shape[0])

    # Solves the case in which index type is tensor
    if len(df_outputs.index) > 0:  # avoid cases where empty df
        if type(df_outputs.index[0]) != str:
            index = [idx.item() for idx in df_outputs.index]
            index_name = df_outputs.index.name
            df_outputs.index = index
            df_outputs.index.names = [index_name]

    if csv_path:
        df_outputs.to_csv(csv_path)

    return df_outputs