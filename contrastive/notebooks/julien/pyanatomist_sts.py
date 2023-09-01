import anatomist.api as anatomist
from soma import aims
import numpy as np
import os
import json
import pandas as pd

ana = anatomist.Anatomist()
#palette = ana.getPalette("Red")

dir_dHCP = '/neurospin/grip/external_databases/dHCP_CR_JD_2018/Projects/denis/release3_morpho_bids/'

## load info_dHCP, the right subjects (copy paste the list) + load session from denis csv
## for TP first, then for TN: find dir_mesh and dir_folds, then same as other pyanatomist
"""
TP = ['CC00529AN18', 'CC00997BN25', 'CC00629XX19', 'CC00955XX15',
       'CC01038XX16', 'CC00805XX13', 'CC00867XX18', 'CC00770XX12',
       'CC00718XX17', 'CC00666XX15', 'CC00661XX10', 'CC00946XX23',
       'CC01025XX11', 'CC00618XX16', 'CC00529BN18', 'CC00530XX11',
       'CC00518XX15', 'CC00526XX15']
"""

TN = ['CC00455XX10', 'CC00139XX16', 'CC00948XX25', 'CC00153XX05',
       'CC01086XX15', 'CC00458XX13', 'CC00402XX06', 'CC00130XX07',
       'CC00203XX05', 'CC00145XX14', 'CC00508XX13', 'CC00858XX17',
       'CC00861XX12', 'CC00914XX15', 'CC00270XX07', 'CC00549XX22',
       'CC00088XX15', 'CC00356XX10']

ft_rd = ['CC01014XX08',
 'CC00516XX13',
 'CC00149XX18',
 'CC00757XX15',
 'CC00457XX12',
 'CC00163XX07',
 'CC00486XX17',
 'CC00427XX15',
 'CC00798XX24',
 'CC00378XX16',
 'CC00165XX09',
 'CC00939XX24',
 'CC00260XX05',
 'CC00622XX12',
 'CC00111XX04',
 'CC00254XX07',
 'CC00948XX25',
 'CC00325XX12']

# get the session ids of R3
dir_sessions = '/neurospin/grip/external_databases/dHCP_CR_JD_2018/Projects/denis/release3_scripts/subjects_file_v4.json'
with open(dir_sessions) as f:
    dict_sessions = json.load(f)
sessions = [value['session_id'] for _, value in dict_sessions.items()]

# get dHCP info
dir_info_dHCP = '/home/jl274628/Documents/info_dHCP.tsv'
info_dHCP = pd.read_csv(dir_info_dHCP, usecols=['participant_id', 'birth_age', 'scan_age', 'session_id'], sep='\t')
# filter over subjects
info_dHCP.drop(info_dHCP[~(info_dHCP['participant_id'].isin(ft_rd))].index, inplace = True)
#info_dHCP.drop(info_dHCP[(info_dHCP['scan_number']!=1)].index, inplace = True) # not always scan 1
# filter over sessions
info_dHCP.drop(info_dHCP[~(info_dHCP['session_id'].isin(sessions))].index, inplace = True)
info_dHCP.reset_index(drop=True, inplace=True)
print(info_dHCP)

# get preterm names
bck_list_1 = []
bck_list_2 = []
window_list = []

ana.loadObject('/casa/host/build/share/brainvisa-share-5.2/nomenclature/hierarchy/sulcal_root_colors.hie')

for idx, (id, session, _, _) in info_dHCP.iterrows():
    dir_folds = dir_dHCP + f'sub-{id}/ses-{session}/anat/t1mri/default_acquisition/default_analysis/folds/3.1/default_session_auto/R{id}_default_session_auto.arg'
    dir_mesh = dir_dHCP + f'sub-{id}/ses-{session}/anat/t1mri/default_acquisition/default_analysis/segmentation/mesh/{id}_Rwhite.gii'
    print(dir_folds)

    bck_list_1.append(ana.loadObject(dir_folds))
    bck_list_2.append(ana.loadObject(dir_mesh))
    window_list.append(ana.createWindow('3D'))

print(len(bck_list_1), len(bck_list_2), len(window_list))
for window, bck1, bck2 in zip(window_list, bck_list_1, bck_list_2):
    #bck1.setPalette(palette)
    window.addObjects(bck1)
    window.addObjects(bck2)

input('Press a key to continue')