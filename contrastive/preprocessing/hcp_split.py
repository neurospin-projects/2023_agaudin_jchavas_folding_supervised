# /usr/bin/env python3
# coding: utf-8

import os
import sys
import pandas as pd
import numpy as np

half1_dir = '/neurospin/dico/data/deep_folding/current/HCP_half_1.csv'
half_1 = pd.read_csv(half1_dir, header=None)
print(half_1.head())
print(type(half_1[0][0]))

print(len(half_1[0].astype(str)))
print('129634' in list(half_1[0].astype(str)))

subjects = []
cpt = 0
for file in os.listdir('/neurospin/dico/data/deep_folding/current/crops/CINGULATE/mask/sulcus_based/2mm/centered_combined/hcp/Rcrops/'):
    if '.minf' not in file :
        sub = file[:6]
        if str(sub) not in list(half_1[0].astype(str)):
            subjects.append(sub)

print('cpt', cpt)
print(len(subjects))

np.savetxt(f"/neurospin/dico/data/deep_folding/current/HCP_half_2_2.csv", np.array(subjects), delimiter =", ", fmt ='% s')
print('saved')
