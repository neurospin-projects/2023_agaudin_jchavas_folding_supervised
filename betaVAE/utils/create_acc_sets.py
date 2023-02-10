# -*- coding: utf-8 -*-
# /usr/bin/env python3
#
#  This software and supporting documentation are distributed by
#      Institut Federatif de Recherche 49
#      CEA/NeuroSpin, Batiment 145,
#      91191 Gif-sur-Yvette cedex
#      France
#
# This software is governed by the CeCILL license version 2 under
# French law and abiding by the rules of distribution of free software.
# You can  use, modify and/or redistribute the software under the
# terms of the CeCILL license version 2 as circulated by CEA, CNRS
# and INRIA at the following URL "http://www.cecill.info".
#
# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using,  modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and,  more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license version 2 and that you accept its terms.

"""
The aim of this script is to generate csv files of subjects for each set of
data of ACC.
Based on the files in '/neurospin/dico/agaudin/Runs/04_pointnet/Output/' \
                        'pointnet/no_4/cingulate_ACCpatterns_embeddings/',
generation of the lists of associated subjects.
"""

import os
import pandas as pd


def create_acc_sets():
    """
    Reads different embeddings of contrastive part and generates equivalent
    subjects lists. 
    """
    root_dir = '/neurospin/dico/agaudin/Runs/04_pointnet/Output/' \
                            'pointnet/no_4/cingulate_ACCpatterns_embeddings'

    save_dir = '/neurospin/dico/lguillon/collab_joel_cingulate/data'

    df_full_sub = pd.read_csv(os.path.join(root_dir, 'full_embeddings.csv'))
    full_sub = df_full_sub['ID']
    full_sub.to_csv(os.path.join(save_dir, 'full_subjects.csv'))

    df_test_sub = pd.read_csv(os.path.join(root_dir, 'test_embeddings.csv'))
    test_sub = df_test_sub['ID']
    test_sub.to_csv(os.path.join(save_dir, 'test_subjects.csv'))

    df_train_sub = pd.read_csv(os.path.join(root_dir, 'train_embeddings.csv'))
    train_sub = df_train_sub['ID']
    train_sub.to_csv(os.path.join(save_dir, 'train_subjects.csv'))

    df_val_sub = pd.read_csv(os.path.join(root_dir, 'val_embeddings.csv'))
    val_sub = df_val_sub['ID']
    val_sub.to_csv(os.path.join(save_dir, 'val_subjects.csv'))



if __name__ == '__main__':
    create_acc_sets()
