import anatomist.api as anatomist
from soma import aims
import numpy as np
import os

ana = anatomist.Anatomist()
#palette = ana.getPalette("Red")

dir_grads = '/neurospin/dico/jlaval/Runs/01_deep_supervised/Program/Output/gradCAM/buckets/'
dir_crops = '/neurospin/dico/data/deep_folding/current/datasets/dHCP_374_subjects/crops/2mm/S.T.s.baby/mask/Rbuckets/'
# get preterm names
filenames = os.listdir(dir_grads)
filenames = [f for f in filenames if f[-1]!='f']
bck_list_1 = []
bck_list_2 = []
window_list = []

for idx, filename in enumerate(filenames):
    subject = filename[:11]
    bck_list_1.append(ana.loadObject(dir_grads + filename))
    bck_list_2.append(ana.loadObject(dir_crops + f'sub-{subject}_cropped_skeleton.bck'))
    window_list.append(ana.createWindow('3D'))

for window, bck1, bck2 in zip(window_list, bck_list_1, bck_list_2):
    #bck1.setPalette(palette)
    window.addObjects(bck1)
    window.addObjects(bck2)

# NB: select all grad in GUI and change color + transparency manually

"""
filename = filenames[0]
subject = filename[:11]
bck1 = ana.loadObject(dir_grads + filename)
bck2 = ana.loadObject(dir_crops + f'sub-{subject}_cropped_skeleton.bck')
window = ana.createWindow('3D')
window.addObjects(bck1)
#bck1.setPalette(palette)
window.addObjects(bck2)
"""

input('Press a key to continue')