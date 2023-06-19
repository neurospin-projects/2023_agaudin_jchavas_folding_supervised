import os
import numpy as np
import pandas as pd
import pickle
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

# Functions

def discretize_continous_label(labels, bins: str or int = "sturges"):
    """Get an estimation of the best bin edges. 'Sturges' is conservative for pretty large datasets (N>1000)."""
    bin_edges = np.histogram_bin_edges(labels, bins=bins)
    # Discretizes the values according to these bins
    discretization = np.digitize(labels, bin_edges[1:], right=True)
    return discretization
    
def get_mask_from_df(source_df, target_df, keys):
    source_keys = source_df[keys].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
    assert len(set(source_keys)) == len(source_keys), f"Multiple identique identifiers found"
    target_keys = target_df[keys].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
    mask = source_keys.isin(target_keys).values.astype(bool)
    return mask


if __name__ == "__main__":
    # Parameters of stratification
    test_size = 0.1
    val_size = 0.1
    nb_folds = 1
    # features on which stratifying
    stratify = ["age", "sex", "site", "diagnosis"]
    random_state = 0
    # new pickle to save
    path_to_save = "/neurospin/dico/agaudin/Runs/09_new_repo/2023_agaudin_jchavas_folding_supervised"
    pickle_name = "test_pickle.pkl"
    # Clinical pathology
    target = "scz" # "asd", "scz", "bipolar"

    # Folders  
    root = os.path.join('/neurospin', 'psy_sbox/analyses/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/data')
    _root = os.path.join(root, "cat12vbm")

    # Get subject metadata for all the studies
    pd_files = "%s_t1mri_mwp1_participants.csv"
    studies = {"scz": ["schizconnect-vip", "bsnip", "cnp", "candi"],
              "bipolar": ["biobd", "bsnip", "cnp", "candi"],
              "asd": ["abide1", "abide2"]}[target]
    unique_keys = {"scz": ["participant_id", "session", "study"],
                   "bipolar": ["participant_id", "session", "study"],
                   "asd": ["participant_id", "session", "run", "study"]
                   }[target]

    #all_labels['study'] = study
    all_labels = pd.concat([pd.read_csv(os.path.join(_root, pd_files % db)) 
                    for db in studies], ignore_index=True, sort=False)
    all_labels.loc[all_labels['session'].isna(), 'session'] = 1
    all_labels.loc[all_labels['session'].isin(['v1', 'V1']), 'session'] = 1
    all_labels["session"] = all_labels["session"].astype(int)
    if "run" in all_labels.columns:
        all_labels.loc[all_labels['run'].isna(), 'run'] = 1	
        all_labels['run'] = all_labels['run'].astype(int)
    print("Number of subjects with metadata", len(all_labels))

    # Select subjects from previous stratification
	# Take only train, val and test_intra
	# The test keeps the same
    dict_pickle = pickle.load(open(os.path.join(root, 'train_val_test_test-intra_scz_stratified.pkl'), "rb"))
    selection = pd.concat([dict_pickle[split] for split in ("train", "validation", "test_intra", "test")], ignore_index=True, sort=False)
    selection.loc[selection['session'].isna(), 'session'] = 1
    selection.loc[selection['session'].isin(['v1', 'V1']), 'session'] = 1
    selection["session"] = selection["session"].astype(int)

    mask = get_mask_from_df(source_df=all_labels, target_df=selection, keys=unique_keys)
    all_labels = all_labels[mask]

    # only keep specified sites
    site_split = ['PRAGUE', 'MRN', 'WUSTL', 'vip', 'NU', 'Baltimore', 'Boston',
                  'Hartford', 'Dallas', 'Detroit', 'CNP', 'CANDI']
    site_split.remove('MRN')
    # keep the others in the external test
    external_test_labels = all_labels[~all_labels.site.isin(site_split)]
    df_test_externe = selection[selection.participant_id.isin(external_test_labels.participant_id)]

    all_labels = all_labels[all_labels.site.isin(site_split)]
    print("Number of subject in selection", len(all_labels))
	
    # Create  arrays for splitting
    dummy_x = np.zeros((len(all_labels), 1, 128, 128, 128))

    if isinstance(stratify, list):
        y = all_labels[stratify].copy(deep=True).values
        if "age" in stratify:
            i_age = stratify.index("age")
            y[:, i_age] = discretize_continous_label(y[:, i_age].astype(np.float32))
    else:
        raise ValueError("Unknown stratifier: {}".format(stratify))

    # Stratification
    print("Stratification on", stratify)
    splitter = MultilabelStratifiedShuffleSplit(n_splits=nb_folds, test_size=test_size, train_size=1-test_size, random_state=random_state)
    gen = splitter.split(dummy_x, y)
    for i in range(nb_folds):
        train_index, test_index = next(gen)
        
        dummy_x_train = dummy_x[train_index]
        y_train = y[train_index]
        df_training = all_labels.iloc[train_index]
        df_training.reset_index(drop=True, inplace=True)
        df_test = all_labels.iloc[test_index]
        df_test.reset_index(drop=True, inplace=True)

        if val_size is not None:
            splitter_val = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=val_size, train_size=1-val_size,
                                                            random_state=random_state)
            for train_i, val_i in splitter_val.split(dummy_x_train, y_train):
                
                df_train = df_training.iloc[train_i]
                df_val = df_training.iloc[val_i]
                df_val.reset_index(drop=True, inplace=True)
                df_train.reset_index(drop=True, inplace=True)
        else:
            df_train = df_training
            df_val = None

    # Saving
    dict_to_save = {}
    dict_to_save['train'] = df_train[unique_keys]
    dict_to_save['validation'] = df_val[unique_keys]
    dict_to_save['test_intra'] = df_test[unique_keys]
    #df_test_externe = dict_pickle["test"]
    df_test_externe.loc[df_test_externe['session'].isna(), 'session'] = 1
    df_test_externe.loc[df_test_externe['session'].isin(['v1', 'V1']), 'session'] = 1
    df_test_externe["session"] = df_test_externe["session"].astype(int)
    df_test_externe.reset_index(drop=True, inplace=True)
    dict_to_save['test'] = df_test_externe

    for k, df in dict_to_save.items():
        print("Split", k)
        print("Number of subjects", len(df))
        print(df.head())

    path_to_scheme = os.path.join(path_to_save, pickle_name)
    with open(path_to_scheme, 'wb') as file:
        pickle.dump(dict_to_save, file)
    print("Pickle Saved : {}".format(path_to_scheme))    