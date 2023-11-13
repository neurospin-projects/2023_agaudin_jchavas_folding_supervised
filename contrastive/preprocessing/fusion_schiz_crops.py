"""Same as the end of fusion_schiz_bases.ipynb (i.e. fusion of bsnip1, candi, cnp and schizconnect),
but only deal with the crops."""

import os
import numpy as np
import pandas as pd



def remove_sub(liste):
    cured_list = []
    for name in liste:
        if 'sub-' in name:
            cured_list.append(name[4:])
        else:
            cured_list.append(name)
    return cured_list


def remove_suffixes(liste):
    cured_list = []
    for name in liste:
        if '_ses-1' in name:
            cured_list.append(name[:-6])
        elif '_ses-v1' in name:
            cured_list.append(name[:-7])
        else:
            cured_list.append(name)
    return cured_list


def cure_df(csv_path, keep_sub=False, save_path=None):
    """Load, remove sub and co, and save a copy of a targeted df"""
    df = pd.read_csv(csv_path)
    participants = df.Subject
    if not keep_sub:
        participants = remove_sub(participants)
    participants = remove_suffixes(participants)

    cured_df = pd.DataFrame(participants, columns=['Subject'])

    if save_path != None:
        cured_df.to_csv(save_path, index=False)

    return cured_df




save_path = "/neurospin/dico/data/deep_folding/current/datasets/schiz/"

schiz_subjects = pd.read_csv(save_path + 'used_schiz_subjects.csv')

regions = ['frontal.']
sides = ['R', 'L']
data_types = ['skeleton', 'label']


for region_name in regions:
    save_path_numpy = save_path + f'crops/2mm/{region_name}/mask/'
    for side in sides:
        for data_type in data_types:

            print("Working on", region_name, side, data_type)

            # load path to csv
            bsnip_path = f"/neurospin/dico/data/deep_folding/current/datasets/bsnip1/crops/2mm/{region_name}/mask/{side}{data_type}_subject.csv"
            candi_path = f"/neurospin/dico/data/deep_folding/current/datasets/candi/crops/2mm/{region_name}/mask/{side}{data_type}_subject.csv"
            cnp_path = f"/neurospin/dico/data/deep_folding/current/datasets/cnp/crops/2mm/{region_name}/mask/{side}{data_type}_subject.csv"
            schizconnect_path = f"/neurospin/dico/data/deep_folding/current/datasets/schizconnect-vip-prague/crops/2mm/{region_name}/mask/{side}{data_type}_subject.csv"

            # remove suffixes etc
            cured_bsnip = cure_df(bsnip_path, save_path=bsnip_path[:-4]+"_cured.csv")
            cured_candi = cure_df(candi_path, save_path=candi_path[:-4]+"_cured.csv")
            cured_cnp = cure_df(cnp_path, save_path=cnp_path[:-4]+"_cured.csv")
            cured_schizconnect = cure_df(schizconnect_path, save_path=schizconnect_path[:-4]+"_cured.csv")

            # load deep_folding numpy files
            bsnip_npy = np.load(f"/neurospin/dico/data/deep_folding/current/datasets/bsnip1/crops/2mm/{region_name}/mask/{side}{data_type}.npy")
            candi_npy = np.load(f"/neurospin/dico/data/deep_folding/current/datasets/candi/crops/2mm/{region_name}/mask/{side}{data_type}.npy")
            cnp_npy = np.load(f"/neurospin/dico/data/deep_folding/current/datasets/cnp/crops/2mm/{region_name}/mask/{side}{data_type}.npy")
            schizconnect_npy = np.load(f"/neurospin/dico/data/deep_folding/current/datasets/schizconnect-vip-prague/crops/2mm/{region_name}/mask/{side}{data_type}.npy")

            # load associated csv
            bsnip_csv = pd.read_csv(f"/neurospin/dico/data/deep_folding/current/datasets/bsnip1/crops/2mm/{region_name}/mask/{side}{data_type}_subject_cured.csv")
            candi_csv = pd.read_csv(f"/neurospin/dico/data/deep_folding/current/datasets/candi/crops/2mm/{region_name}/mask/{side}{data_type}_subject_cured.csv")
            # /!\ cnp Subject ids are ints
            cnp_csv = pd.read_csv(f"/neurospin/dico/data/deep_folding/current/datasets/cnp/crops/2mm/{region_name}/mask/{side}{data_type}_subject_cured.csv", dtype=str)
            schizconnect_csv = pd.read_csv(f"/neurospin/dico/data/deep_folding/current/datasets/schizconnect-vip-prague/crops/2mm/{region_name}/mask/{side}{data_type}_subject_cured.csv")


            # checks
            print([cured_bsnip.shape[0], cured_candi.shape[0], cured_cnp.shape[0], cured_schizconnect.shape[0]]) # sum should be equal to about 2100
            print(bsnip_npy.shape, candi_npy.shape, cnp_npy.shape, schizconnect_npy.shape)
            print(bsnip_csv.shape, candi_csv.shape, cnp_csv.shape, schizconnect_csv.shape)


            # Keep only the used subjects (else bug)
            subs = np.copy(schiz_subjects.participant_id.values)

            bsnip_kept_subjects = bsnip_csv[bsnip_csv.Subject.isin(subs)].index
            candi_kept_subjects = candi_csv[candi_csv.Subject.isin(subs)].index
            cnp_kept_subjects = cnp_csv[cnp_csv.Subject.isin(subs)].index
            schizconnect_kept_subjects = schizconnect_csv[schizconnect_csv.Subject.isin(subs)].index

            print("Should be equal to 1292 :",
                  np.sum([bsnip_kept_subjects.shape[0], candi_kept_subjects.shape[0], cnp_kept_subjects.shape[0], schizconnect_kept_subjects.shape[0]]))
            
            # Finally restraint the numpys and the csv
            bsnip_npy = bsnip_npy[bsnip_kept_subjects]
            candi_npy = candi_npy[candi_kept_subjects]
            cnp_npy = cnp_npy[cnp_kept_subjects]
            schizconnect_npy = schizconnect_npy[schizconnect_kept_subjects]

            bsnip_csv = bsnip_csv.loc[bsnip_kept_subjects, :]
            candi_csv = candi_csv.loc[candi_kept_subjects, :]
            cnp_csv = cnp_csv.loc[cnp_kept_subjects, :]
            schizconnect_csv = schizconnect_csv.loc[schizconnect_kept_subjects, :]

            # concat
            schiz_npy = np.concatenate([bsnip_npy, candi_npy, cnp_npy, schizconnect_npy], axis=0)
            print(schiz_npy.shape)

            schiz_csv = pd.concat([bsnip_csv, candi_csv, cnp_csv, schizconnect_csv], axis=0, ignore_index=True)
            print(schiz_csv.shape)
            
            # sort the subjects by name
            print("Before sorting")
            print(schiz_csv[:5])
            schiz_csv = schiz_csv.sort_values(by='Subject')
            print("After sorting")
            print(schiz_csv[:5])
            # sort the numpy for it to correspond to the csv
            schiz_npy = schiz_npy[list(schiz_csv.index)]

            ## save
            # create the folders if they don't exist already
            if not os.path.exists(save_path_numpy):
                os.makedirs(save_path_numpy)

            # save the numpy and csv
            np.save(save_path_numpy + f'{side}{data_type}.npy', schiz_npy)
            schiz_csv = schiz_csv['Subject']
            schiz_csv.to_csv(save_path_numpy + f'{side}{data_type}_subject.csv', index=False)

            # create a folder (empty)
            if data_type == 'skeleton':
                truc = 'crops'
            else:
                truc = 'labels'
            if not os.path.exists(save_path_numpy+f'{side}{truc}'):
                os.mkdir(save_path_numpy+f'{side}{truc}')

            # checks
            print(f"Check by loading what have been saved at {save_path_numpy}")
            loaded_npy = np.load(save_path_numpy + f'{side}{data_type}.npy')
            loaded_csv = pd.read_csv(save_path_numpy + f'{side}{data_type}_subject.csv')

            print(loaded_npy.shape)
            print(loaded_csv.shape)

            print(loaded_csv.applymap(type).Subject.unique())

