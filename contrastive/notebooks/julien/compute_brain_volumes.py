import os.path as op
import os
import numpy as np
import pandas as pd
from soma import aims

def get_gw_volumes(db_path, subname, acquisition, compressed):
    hemi_volumes = []
    for hemi in ['L', 'R']:
        gw_f = op.join(db_path, subname, 't1mri', acquisition, 'default_analysis', 'segmentation', f'{hemi}grey_white_{subname}.nii')
        if compressed:
            gw_f = gw_f+'.gz'
        if op.isfile(gw_f):
            gw = aims.read(gw_f)
            vs = np.array(gw.header()['voxel_size'])[:3]
            print(vs)
            vx_vol = vs[0] * vs[1] * vs[2]
            px_vol = np.sum(np.array(gw) > 0)
            # print(subname, vs, px_vol)
            hemi_volumes.append(vx_vol * px_vol)
        else:
            print(f"file doesn't exist for subject {subname}")
            hemi_volumes.append(0)
    return hemi_volumes

save_dir='/neurospin/dico/jlaval/data/'

#Utrecht
dir = '/neurospin/dico/data/bv_databases/human/manually_labeled/utrecht/utrecht_sulci/'
acquisition='40wks_acquisition'
save_file='full_volumes_Utrecht.csv'
compressed=True

#dHCP old
"""
dir = '/neurospin/dico/data/bv_databases/dHCP/neurospin/grip/external_databases/dHCP_CR_JD_2018/release1/subjects/dHCP_release1/'
acquisition='default_acquisition'
save_file='full_volumes_dHCP.csv'
compressed=False
"""

#dHCP 165 subjects
"""
dir = '/neurospin/grip/external_databases/dHCP_CR_JD_2018/Projects/denis/release3_morpho/dHCP/'
acquisition='default_acquisition'
save_file='full_volumes_dHCP_165_subjects.csv'
compressed=True
"""

### pipeline

list_volumes=[]

files = [f for f in os.listdir(dir) if f[-1]!='f']
files_filtered = files.copy()
print(files)
for file in files:
    print(file)
    vols = get_gw_volumes(dir, file, acquisition, compressed)
    vol=vols[0]+vols[1]
    if vol!=0: #vol==0 means the file didn't exist
        list_volumes.append(vol)
    else:
        files_filtered.remove(file)
list_volumes = np.array(list_volumes)

df = pd.DataFrame({'Subject': files_filtered,
                   'Volume': list_volumes})

save_dir = save_dir + save_file
df.to_csv(save_dir, index=False, sep=',')
print('volumes saved as csv')
