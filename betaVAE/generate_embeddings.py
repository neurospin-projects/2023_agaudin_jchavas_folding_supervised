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
#
# https://github.com/neurospin-projects/2021_jchavas_lguillon_deepcingulate/

import os
import pandas as pd
import torch
import torch.nn as nn

from beta_vae import VAE, ModelTester
from load_data import create_test_subset
from configs.config import Config


def generate_embedding_sets(embedding, config, dataset='cingulate_ACCpatterns'):
    """
    From a dataframe of encoded subjects, generate csv files:
    full_embeddings.csv, train_embeddings.csv etc.
    """
    subjects_dir = '/neurospin/dico/lguillon/collab_joel_aymeric_cingulate/data'
    save_dir = config.test_model_dir + f'/{dataset}_embeddings'
    os.mkdir(save_dir)

    # Loading of data subsets
    full = pd.read_csv(os.path.join(subjects_dir, 'full_subjects.csv'),
                header=None, usecols=[1], names=['ID'])
    full['ID'] = full['ID'].astype('str')

    train = pd.read_csv(os.path.join(subjects_dir, 'train_subjects.csv'),
                header=None, usecols=[1], names=['ID'])
    train['ID'] = train['ID'].astype('str')

    val = pd.read_csv(os.path.join(subjects_dir, 'val_subjects.csv'),
                header=None, usecols=[1], names=['ID'])
    val['ID'] = val['ID'].astype('str')

    test = pd.read_csv(os.path.join(subjects_dir, 'test_subjects.csv'),
                header=None, usecols=[1], names=['ID'])
    test['ID'] = test['ID'].astype('str')

    embedding['ID'] = embedding["ID"].astype('str')

    # Split of full ACC dataset embeddings as different subsets
    full_embeddings = embedding.merge(full, left_on='ID', right_on='ID')
    train_embeddings = embedding.merge(train, left_on='ID', right_on='ID')
    val_embeddings = embedding.merge(val, left_on='ID', right_on='ID')
    test_embeddings = embedding.merge(test, left_on='ID', right_on='ID')

    # Saving of ACC subsets as csv files
    full_embeddings.to_csv(os.path.join(save_dir, 'full_embeddings.csv'), index=False)
    train_embeddings.to_csv(os.path.join(save_dir, 'train_embeddings.csv'), index=False)
    val_embeddings.to_csv(os.path.join(save_dir, 'val_embeddings.csv'), index=False)
    test_embeddings.to_csv(os.path.join(save_dir, 'test_embeddings.csv'), index=False)


def main(dataset='cingulate_ACCpatterns'):
    """
    Infer a trained model on test data and saves the embeddings as csv
    """
    config = Config()

    torch.manual_seed(0)
    device = 'cpu'
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            vae = nn.DataParallel(vae)

    model_dir = os.path.join(config.test_model_dir, 'checkpoint.pt')
    model = VAE(config.in_shape, config.n, depth=3)
    model.load_state_dict(torch.load(model_dir))
    model = model.to(device)

    weights = [1, 2]
    class_weights = torch.FloatTensor(weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='sum')

    subset_test = create_test_subset(config)
    testloader = torch.utils.data.DataLoader(
              subset_test,
              batch_size=config.batch_size,
              num_workers=8,
              shuffle=True)
    dico_set_loaders = {'test': testloader}

    tester = ModelTester(model=model, dico_set_loaders=dico_set_loaders,
                         kl_weight=config.kl, loss_func=criterion,
                         n_latent=config.n, depth=3)

    results = tester.test()
    embedding = pd.DataFrame(results['test']).T.reset_index()
    embedding = embedding.rename(columns={"index":"ID"})
    embedding = embedding.rename(columns={k:f"dim{k+1}" for k in range(config.n)})
    print(embedding.head())

    generate_embedding_sets(embedding, config, dataset=dataset)

if __name__ == '__main__':
    main(dataset='cingulate_ACCpatterns')
