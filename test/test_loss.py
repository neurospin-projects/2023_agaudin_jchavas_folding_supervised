#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
""" Test of loss functions and other relevant functions

"""

import torch
import numpy as np
from contrastive.losses import GeneralizedSupervisedNTXenLoss
from contrastive.utils import logs

log = logs.set_file_logger(__file__)

def test_mock():
    loss = GeneralizedSupervisedNTXenLoss()

def test_weights_two_labels():
    """Verify weights in a simple setting"""
    loss = GeneralizedSupervisedNTXenLoss(temperature=1.0, sigma=0.1)
    z_i = torch.Tensor([[0], 
                        [1]]) # not used in test
    z_j = torch.Tensor([[1], 
                        [0]])  # not used in test
    labels = torch.Tensor([[0], 
                           [1]])
    weights_ref = torch.tensor([[0, 0, 1, 0],
                                [0, 0, 0, 1],
                                [1, 0, 0, 0],
                                [0, 1, 0, 0]])
    _, weights = loss.forward_supervised(z_i, z_j, labels)
    assert torch.allclose(weights.double(), weights_ref.double())


def test_weights_one_label():
    """Verify weights in a simple setting"""
    loss = GeneralizedSupervisedNTXenLoss(temperature=1.0, sigma=0.1)
    z_i = torch.Tensor([[0], 
                        [1]]) # not used in test
    z_j = torch.Tensor([[1], 
                        [0]])  # not used in test
    labels = torch.Tensor([[0], 
                           [0]])
    weights_ref = 1/3.* torch.tensor([[0, 1, 1, 1],
                                      [1, 0, 1, 1],
                                      [1, 1, 0, 1],
                                      [1, 1, 1, 0]])
    _, weights = loss.forward_supervised(z_i, z_j, labels)
    assert torch.allclose(weights.double(), weights_ref.double())


def test_supervised_all_equal():
    """Verify loss labels in a simple setting.
    
    All vectors are equal"""
    loss = GeneralizedSupervisedNTXenLoss(temperature=1.0, sigma=0.1)
    z_i = torch.Tensor([[1, 0], 
                        [1, 0]]) # [N,D]]
    z_j = torch.Tensor([[1, 0], 
                        [1, 0]]) # [N,D]]
    labels = torch.Tensor([[0], 
                           [1]]) # [N]
    
    loss_label, _ = loss.forward_supervised(z_i, z_j, labels)
    loss_label_ref = 2*np.log(3.)
    loss_label_ref = torch.from_numpy(np.array((loss_label_ref)))
    assert torch.allclose(loss_label.double(),
                          loss_label_ref.double())


def test_supervised_different_for_different_labels():
    """Verify weights in a simple setting
    
    All vectors beloning to same label are equal"""
    loss = GeneralizedSupervisedNTXenLoss(temperature=1.0, sigma=0.1)
    z_i = torch.Tensor([[1, 0], 
                        [0, 1]]) # [N,D]]
    z_j = torch.Tensor([[1, 0], 
                        [0, 1]]) # [N,D]]
    labels = torch.Tensor([[0], 
                           [1]]) # [N]
    
    loss_label, _ = loss.forward_supervised(z_i, z_j, labels)
    loss_label_ref = 2*np.log(np.exp(1)+2) - 2
    loss_label_ref = torch.from_numpy(np.array((loss_label_ref)))
    assert torch.allclose(loss_label.double(),
                          loss_label_ref.double())


def test_supervised_3_labels_all_equal():
    """Verify loss labels in a simple setting.
    
    3 vectors, 2 labels, all vectors are equal"""
    loss = GeneralizedSupervisedNTXenLoss(temperature=1.0, sigma=0.1)
    z_i = torch.Tensor([[1, 0], 
                        [1, 0],
                        [1, 0]]) # [N,D]]
    z_j = torch.Tensor([[1, 0], 
                        [1, 0],
                        [1, 0]]) # [N,D]]
    labels = torch.Tensor([[0], 
                           [1],
                           [1]]) # [N]
    
    loss_label, _ = loss.forward_supervised(z_i, z_j, labels)
    loss_label_ref = 2*np.log(5.)
    loss_label_ref = torch.from_numpy(np.array((loss_label_ref)))
    assert torch.allclose(loss_label.double(),
                          loss_label_ref.double())


def test_supervised_3_labels_all_equal_different_for_different_labels():
    """Verify loss labels in a simple setting.
    
    3 vectors, 2 labels, all vectors for same are equal"""
    loss = GeneralizedSupervisedNTXenLoss(temperature=1.0, sigma=0.1)
    z_i = torch.Tensor([[0, 1], 
                        [1, 0],
                        [1, 0]]) # [N,D]]
    z_j = torch.Tensor([[0, 1], 
                        [1, 0],
                        [1, 0]]) # [N,D]]
    labels = torch.Tensor([[0], 
                           [1],
                           [1]]) # [N]
    
    loss_label, _ = loss.forward_supervised(z_i, z_j, labels)
    loss_label_ref = -2.0 + \
                     2./3*np.log(np.exp(1)+4) + \
                     4./3*np.log(3*np.exp(1)+2)
    loss_label_ref = torch.from_numpy(np.array((loss_label_ref)))
    assert torch.allclose(loss_label.double(),
                          loss_label_ref.double())


def test_pure_contrastive():
    """Verifies pure contrastive results in a simple setting"""
    loss = GeneralizedSupervisedNTXenLoss(temperature=1.0, sigma=0.1)
    z_i = torch.Tensor([[0], 
                        [1]]) # N=2; D=1
    z_j = torch.Tensor([[1], 
                        [0]])  # N=2, D=1
    loss_contrastive = loss.forward_pure_contrastive(z_i, z_j)
    loss_contrastive_ref = np.log(3.) + np.log(2+np.exp(1))
    loss_contrastive_ref = torch.from_numpy(np.array(loss_contrastive_ref))
    assert torch.allclose(loss_contrastive.double(),
                          loss_contrastive_ref.double())


def test_pure_contrastive_D_2_different_positive():
    """Verifies pure contrastive results in a simple setting
    
    When the two positive pairs are orthogonal"""
    loss = GeneralizedSupervisedNTXenLoss(temperature=1.0, sigma=0.1)
    z_i = torch.Tensor([[0, 1], 
                        [1, 0]]) # N=2; D=2
    z_j = torch.Tensor([[1, 0], 
                        [0, 1]])  # N=2, D=1
    loss_contrastive = loss.forward_pure_contrastive(z_i, z_j)
    loss_contrastive_ref = 2*np.log(2+np.exp(1))
    loss_contrastive_ref = torch.from_numpy(np.array(loss_contrastive_ref))
    assert torch.allclose(loss_contrastive.double(),
                          loss_contrastive_ref.double())


def test_pure_contrastive_D_2_equal_positive():
    """Verifies pure contrastive results in a simple setting
    
    When the two positive pairs are equal"""
    loss = GeneralizedSupervisedNTXenLoss(temperature=1.0, sigma=0.1)
    z_i = torch.Tensor([[0, 1], 
                        [1, 0]]) # N=2; D=2
    z_j = torch.Tensor([[0, 1], 
                        [1, 0]])  # N=2, D=1
    loss_contrastive = loss.forward_pure_contrastive(z_i, z_j)
    loss_contrastive_ref = 2*np.log(2+np.exp(1)) - 2.0
    loss_contrastive_ref = torch.from_numpy(np.array(loss_contrastive_ref))
    assert torch.allclose(loss_contrastive.double(),
                          loss_contrastive_ref.double())


def test_pure_contrastive_all_ones():
    """Verifies pure contrastive results in a simple setting
    
    All vectors are equal"""
    loss = GeneralizedSupervisedNTXenLoss(temperature=1.0, sigma=0.1)
    z_i = torch.Tensor([[1], 
                        [1]]) # N=2; D=1
    z_j = torch.Tensor([[1], 
                        [1]])  # N=2, D=1
    loss_contrastive = loss.forward_pure_contrastive(z_i, z_j)
    loss_contrastive_ref = 2*np.log(3.)
    loss_contrastive_ref = torch.from_numpy(np.array(loss_contrastive_ref))
    assert torch.allclose(loss_contrastive.double(),
                          loss_contrastive_ref.double())


def test_compare_supervised_unsupervised():
    """Compare supervised and pure contrastive losses
    
    When all labels are different, 
    both supervised and contrastive losses should be equal"""
    loss = GeneralizedSupervisedNTXenLoss(temperature=1.0, sigma=0.1)
    z_i = torch.randint(0, 20, (10,3)).float()
    z_j = torch.randint(0, 20, (10,3)).float()
    labels = torch.arange(0,10).T
    loss_labels, _ = loss.forward_supervised(z_i, z_j, labels)
    loss_pure_contrastive = loss.forward_pure_contrastive(z_i, z_j)
    assert torch.allclose(loss_labels.double(), loss_pure_contrastive.double())


def test_forward_supervised_D_2():
    loss = GeneralizedSupervisedNTXenLoss(temperature=1.0, sigma=0.1)


def test_forward_supervised():
    loss = GeneralizedSupervisedNTXenLoss(temperature=1.0, sigma=0.1)


if __name__ == "__main__":
    test_compare_supervised_unsupervised()