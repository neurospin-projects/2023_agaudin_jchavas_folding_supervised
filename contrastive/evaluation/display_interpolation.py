""" The aim of this script is to display saved model outputs thanks to Anatomist.
Model outputs are stored as numpy arrays.
"""

import anatomist.api as anatomist
from soma import aims
import numpy as np


def array_to_ana(ana_a, img, sub_id, phase, status):
    """
    Transforms output tensor into volume readable by Anatomist and defines
    name of the volume displayed in Anatomist window.
    Returns volume displayable by Anatomist
    """
    vol_img = aims.Volume(img)
    a_vol_img = ana_a.toAObject(vol_img)
    vol_img.header()['voxel_size'] = [2, 2, 2]
    a_vol_img.setName(status+'_'+ str(sub_id)+'_'+str(phase)) # display name
    a_vol_img.setChanged()
    a_vol_img.notifyObservers()

    return vol_img, a_vol_img


def main():
    """
    In the Anatomist window, for each model output, corresponding input will
    also be displayed at its left side.
    Number of columns and view (Sagittal, coronal, frontal) can be specified.
    (It's better to choose an even number for number of columns to display)
    """
    root_dir = "./"

    a = anatomist.Anatomist()
    block = a.AWindowsBlock(a, 7)  # Parameter 6 corresponds to the number of columns displayed. Can be changed.

    input_arr = np.load(root_dir+'interpolation_0_1.npy').astype('float32') # Input
    id_arr = np.load(root_dir+'label_interpolation_0_1.npy') # Subject id
    #input_arr = np.load(root_dir+'arr_est2.npy').astype('float32') # Input
    #id_arr = np.load(root_dir+'id_est2.npy') # Subject id
    for k in range(len(id_arr)):
        #for k in range(0, 50):
        print(id_arr.shape)
        img = input_arr[k]
        sub_id = id_arr[k]
        #for img, sub_id in zip(input, id_arr):
        globals()['block%s' % (sub_id)] = a.createWindow('Sagittal', block=block)

        globals()['img%s' % (sub_id)], globals()['a_img%s' % (sub_id)] = array_to_ana(a, img, sub_id, phase='', status='')

        globals()['block%s' % (sub_id)].addObjects(globals()['a_img%s' % (sub_id)])



if __name__ == '__main__':
    main()
    from soma.qt_gui.qt_backend import Qt
    Qt.qApp.exec_()
