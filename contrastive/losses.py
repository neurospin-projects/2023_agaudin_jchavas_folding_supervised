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
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
from sklearn.metrics.pairwise import rbf_kernel
from contrastive.utils import logs

log = logs.set_file_logger(__file__)


def mean_off_diagonal(a):
    """Computes the mean of off-diagonal elements"""
    n = a.shape[0]
    return ((a.sum() - a.trace()) / (n * n - n))


def quantile_off_diagonal(a):
    """Computes the quantile of off-diagonal elements
    TODO: it is here the quantile of the whole a"""
    return a.quantile(0.75)


def print_info(z_i, z_j, sim_zij, sim_zii, sim_zjj, temperature):
    """prints useful info over correlations"""

    log.info("histogram of z_i after normalization:")
    log.info(np.histogram(z_i.detach().cpu().numpy() * 100, bins='auto'))

    log.info("histogram of z_j after normalization:")
    log.info(np.histogram(z_j.detach().cpu().numpy() * 100, bins='auto'))

    # Gives histogram of sim vectors
    log.info("histogram of sim_zij:")
    log.info(
        np.histogram(
            sim_zij.detach().cpu().numpy() *
            temperature *
            100,
            bins='auto'))

    # Diagonals as 1D tensor
    diag_ij = sim_zij.diagonal()

    # Prints quantiles of positive pairs (views from the same image)
    quantile_positive_pairs = diag_ij.quantile(0.75)
    log.info(
        f"quantile of positives ij = "
        f"{quantile_positive_pairs.cpu()*temperature*100}")

    # Computes quantiles of negative pairs
    quantile_negative_ii = quantile_off_diagonal(sim_zii)
    quantile_negative_jj = quantile_off_diagonal(sim_zjj)
    quantile_negative_ij = quantile_off_diagonal(sim_zij)

    # Prints quantiles of negative pairs
    log.info(
        f"quantile of negatives ii = "
        f"{quantile_negative_ii.cpu()*temperature*100}")
    log.info(
        f"quantile of negatives jj = "
        f"{quantile_negative_jj.cpu()*temperature*100}")
    log.info(
        f"quantile of negatives ij = "
        f"{quantile_negative_ij.cpu()*temperature*100}")


class NTXenLoss(nn.Module):
    """
    Normalized Temperature Cross-Entropy Loss for Constrastive Learning
    Refer for instance to:
    Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton
    A Simple Framework for Contrastive Learning of Visual Representations,
    arXiv 2020
    """

    def __init__(self, temperature=0.1, return_logits=False):
        super().__init__()
        self.temperature = temperature
        self.INF = 1e8
        self.return_logits = return_logits

    def forward(self, z_i, z_j):
        N = len(z_i)
        z_i = func.normalize(z_i, p=2, dim=-1)  # dim [N, D]
        z_j = func.normalize(z_j, p=2, dim=-1)  # dim [N, D]

        # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zii = (z_i @ z_i.T) / self.temperature

        # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zjj = (z_j @ z_j.T) / self.temperature

        # dim [N, N] => the diag contains the correct pairs (i,j)
        # (x transforms via T_i and T_j)
        sim_zij = (z_i @ z_j.T) / self.temperature

        # print_info(z_i, z_j, sim_zij, sim_zii, sim_zjj, self.temperature)

        # 'Remove' the diag terms by penalizing it (exp(-inf) = 0)
        sim_zii = sim_zii - self.INF * torch.eye(N, device=z_i.device)
        sim_zjj = sim_zjj - self.INF * torch.eye(N, device=z_i.device)

        correct_pairs = torch.arange(N, device=z_i.device).long()
        loss_i = func.cross_entropy(torch.cat([sim_zij, sim_zii], dim=1),
                                    correct_pairs)
        loss_j = func.cross_entropy(torch.cat([sim_zij.T, sim_zjj], dim=1),
                                    correct_pairs)

        if self.return_logits:
            return (loss_i + loss_j), sim_zij, sim_zii, sim_zjj

        return (loss_i + loss_j)

    def __str__(self):
        return "{}(temp={})".format(type(self).__name__, self.temperature)


class CrossEntropyLoss_Classification(nn.Module):
    """
    Cross entropy loss between outputs and labels
    """

    def __init__(self, device=None, class_weights=None):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, output_i, output_j, labels):
        output_i = output_i.float()
        output_j = output_j.float()

        loss_i = self.loss(output_i,
                           labels[:, 0])
        loss_j = self.loss(output_j,
                           labels[:, 0])

        return (loss_i + loss_j)

    def __str__(self):
        return f"{type(self).__name__}"


class MSELoss_Regression(nn.Module):
    """
    Regression loss between outputs and regressor
    """

    def __init__(self, device=None):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, output_i, output_j, labels):
        output_i = output_i.float()
        output_j = output_j.float()
        labels = labels.float()

        loss_i = self.loss(output_i,
                           labels)
        loss_j = self.loss(output_j,
                           labels)

        return 100*(loss_i + loss_j)

    def __str__(self):
        return f"{type(self).__name__}"


class GeneralizedSupervisedNTXenLoss(nn.Module):
    def __init__(self, kernel='rbf',
                 temperature=0.1,
                 temperature_supervised=0.5,
                 return_logits=False,
                 sigma=1.0,
                 proportion_pure_contrastive=1.0):
        """
        :param kernel: a callable function f: [K, *] x [K, *] -> [K, K]
                                              y1, y2          -> f(y1, y2)
                        where (*) is the dimension of the labels (yi)
        default: an rbf kernel parametrized by 'sigma'
                 which corresponds to gamma=1/(2*sigma**2)
        :param temperature:
        :param return_logits:
        """

        # sigma = prior over the label's range
        super().__init__()
        self.kernel = kernel
        self.sigma = sigma
        if self.kernel == 'rbf':
            self.kernel = \
                lambda y1, y2: rbf_kernel(y1, y2, gamma=1./(2*self.sigma**2))
        else:
            assert hasattr(self.kernel, '__call__'), \
                'kernel must be a callable'
        self.temperature = temperature
        self.temperature_supervised = temperature_supervised
        self.proportion_pure_contrastive = proportion_pure_contrastive
        self.return_logits = return_logits
        self.INF = 1e8

    def forward_pure_contrastive(self, z_i, z_j):
        N = len(z_i)
        z_i = func.normalize(z_i, p=2, dim=-1)  # dim [N, D]
        z_j = func.normalize(z_j, p=2, dim=-1)  # dim [N, D]
        sim_zii = (z_i @ z_i.T) / self.temperature  # dim [N, N]

        # => Upper triangle contains incorrect pairs
        sim_zjj = (z_j @ z_j.T) / self.temperature  # dim [N, N]

        # => Upper triangle contains incorrect pairs
        sim_zij = (z_i @ z_j.T) / self.temperature  # dim [N, N]
        # => the diag contains the correct pairs (i,j)
        #    (x transforms via T_i and T_j)

        # 'Remove' the diag terms by penalizing it (exp(-inf) = 0)
        sim_zii = sim_zii - self.INF * torch.eye(N, device=z_i.device)
        sim_zjj = sim_zjj - self.INF * torch.eye(N, device=z_i.device)

        correct_pairs = torch.arange(N, device=z_i.device).long()

        loss_i = func.cross_entropy(torch.cat([sim_zij, sim_zii], dim=1),
                                    correct_pairs)
        loss_j = func.cross_entropy(torch.cat([sim_zij.T, sim_zjj], dim=1),
                                    correct_pairs)

        return loss_i+loss_j

    def forward_supervised(self, z_i, z_j, labels):
        N = len(z_i)
        assert N == len(labels), "Unexpected labels length: %i" % len(labels)

        z_i = func.normalize(z_i, p=2, dim=-1)  # dim [N, D]
        z_j = func.normalize(z_j, p=2, dim=-1)  # dim [N, D]
        sim_zii = (z_i @ z_i.T) / self.temperature_supervised  # dim [N, N]

        # => Upper triangle contains incorrect pairs
        sim_zjj = (z_j @ z_j.T) / self.temperature_supervised  # dim [N, N]

        # => Upper triangle contains incorrect pairs
        sim_zij = (z_i @ z_j.T) / self.temperature_supervised  # dim [N, N]
        # => the diag contains the correct pairs (i,j)
        #    (x transforms via T_i and T_j)

        # 'Remove' the diag terms by penalizing it (exp(-inf) = 0)
        sim_zii = sim_zii - self.INF * torch.eye(N, device=z_i.device)
        sim_zjj = sim_zjj - self.INF * torch.eye(N, device=z_i.device)

        all_labels = \
            labels.view(N, -1).repeat(2, 1).detach().cpu().numpy()  # [2N, *]
        if np.sum(np.isnan(all_labels)) > 0:
            raise ValueError("Nan detected in labels")
        weights = self.kernel(all_labels, all_labels)  # [2N, 2N]
        weights = weights * (1 - np.eye(2*N))  # puts 0 on the diagonal

        # We normalize the weights
        norm = weights.sum(axis=1).reshape(2*N, 1)
        weights /= norm
        weights_norm = weights * np.log(norm)

        # if 'rbf' kernel and sigma->0,
        # we retrieve the classical NTXenLoss (without labels)
        sim_Z = torch.cat([torch.cat([sim_zii, sim_zij], dim=1),
                           torch.cat([sim_zij.T, sim_zjj], dim=1)],
                          dim=0)  # [2N, 2N]
        log_sim_Z = func.log_softmax(sim_Z, dim=1)

        weights = torch.from_numpy(weights)
        weights_norm = torch.from_numpy(weights_norm)
        loss_label = -1./N * (weights.to(z_i.device)
                              * log_sim_Z).sum()
        loss_label += -1./N * (weights_norm.to(z_i.device)).sum()

        return loss_label, weights

    def forward_L1(self, z_i, z_j):
        N = len(z_i)
        z_i = func.normalize(z_i, p=2, dim=-1)  # dim [N, D]
        z_j = func.normalize(z_j, p=2, dim=-1)  # dim [N, D]

        loss_i = torch.linalg.norm(z_i, ord=1, dim=-1).sum() / N
        loss_j = torch.linalg.norm(z_j, ord=1, dim=-1).sum() / N

        return loss_i+loss_j

    def compute_parameters_for_display(self, z_i, z_j):
        N = len(z_i)
        z_i = func.normalize(z_i, p=2, dim=-1)  # dim [N, D]
        z_j = func.normalize(z_j, p=2, dim=-1)  # dim [N, D]
        sim_zii = (z_i @ z_i.T) / self.temperature  # dim [N, N]

        # => Upper triangle contains incorrect pairs
        sim_zjj = (z_j @ z_j.T) / self.temperature  # dim [N, N]

        # => Upper triangle contains incorrect pairs
        sim_zij = (z_i @ z_j.T) / self.temperature  # dim [N, N]
        # => the diag contains the correct pairs (i,j)
        #    (x transforms via T_i and T_j)

        # 'Remove' the diag terms by penalizing it (exp(-inf) = 0)
        sim_zii = sim_zii - self.INF * torch.eye(N, device=z_i.device)
        sim_zjj = sim_zjj - self.INF * torch.eye(N, device=z_i.device)

        correct_pairs = torch.arange(N, device=z_i.device).long()

        return sim_zii, sim_zij, sim_zjj, correct_pairs

    def forward(self, z_i, z_j, labels):
        N = len(z_i)
        D = z_i.shape[1]
        assert N == len(labels), "Unexpected labels length: %i" % len(labels)

        # We compute the pure SimCLR loss
        z_i_pure_contrastive = z_i
        z_j_pure_contrastive = z_j
        loss_pure_contrastive = self.forward_pure_contrastive(
            z_i_pure_contrastive,
            z_j_pure_contrastive
        )

        # We compute the generalized supervised loss
        z_i_supervised = z_i
        z_j_supervised = z_j
        loss_supervised, weights = self.forward_supervised(
            z_i_supervised,
            z_j_supervised,
            labels)

        # We compute the L1 norm to enforce sparsity
        # loss_L1 = self.forward_L1(z_i_supervised, z_j_supervised)

        # We compute matrices for tensorboard displays
        sim_zii, sim_zij, sim_zjj, correct_pairs = \
            self.compute_parameters_for_display(z_i, z_j)

        loss_combined = \
            self.proportion_pure_contrastive*loss_pure_contrastive \
            + (1-self.proportion_pure_contrastive) * loss_supervised
        # + loss_L1

        if self.return_logits:
            return loss_combined, loss_supervised.detach(), \
                sim_zij, sim_zii, sim_zjj, correct_pairs, weights

        return loss_combined

    def __str__(self):
        return "{}(temp={}, kernel={}, sigma={})".format(type(self).__name__,
                                                         self.temperature,
                                                         self.kernel.__name__,
                                                         self.sigma)


class CrossEntropyLoss(nn.Module):
    """
    Normalized Temperature Cross-Entropy Loss for Constrastive Learning
    Refer for instance to:
    Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton
    A Simple Framework for Contrastive Learning of Visual Representations,
    arXiv 2020
    """

    def __init__(self, weights=[1, 2], reduction='sum', device=None):
        super().__init__()
        self.class_weights = torch.FloatTensor(weights).to(device)
        self.reduction = reduction
        self.loss = nn.CrossEntropyLoss(weight=self.class_weights,
                                        reduction=self.reduction)

    def forward(self, sample, output_i, output_j):
        sample = (sample >= 1).long()
        output_i = output_i.float()
        output_j = output_j.float()

        loss_i = self.loss(output_i,
                           sample[:, 0, :, :, :])
        loss_j = self.loss(output_j,
                           sample[:, 0, :, :, :])

        return (loss_i + loss_j)

    def __str__(self):
        return "{}(temp={})".format(type(self).__name__, self.temperature)


class NTXenLoss_NearestNeighbours(nn.Module):
    """
    Normalized Nearest Neighbour Temperature Cross-Entropy Loss
    for Constrastive Learning
    Refer for instance to:
    Dwibedi et al, 2021
    With a little help from my friends nearest-neighbours
    """

    def __init__(self, temperature=0.1, return_logits=False):
        super().__init__()
        self.temperature = temperature
        self.INF = 1e8
        self.return_logits = return_logits

    def forward(self, z_i, z_j):
        N = len(z_i)
        diag_inf = self.INF * torch.eye(N, device=z_i.device)

        #####################################################
        # Computes the classical terms for NTXenLoss
        #####################################################

        z_i = func.normalize(z_i, p=2, dim=-1)  # dim [N, D]
        z_j = func.normalize(z_j, p=2, dim=-1)  # dim [N, D]

        # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zii = (z_i @ z_i.T) / self.temperature

        # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zjj = (z_j @ z_j.T) / self.temperature

        # dim [N, N] => the diag contains the correct pairs (i,j)
        # (x transforms via T_i and T_j)
        sim_zij = (z_i @ z_j.T) / self.temperature
        sim_zji = sim_zij.T

        log.info("histogram of zij:")
        log.info(
            np.histogram(
                sim_zij.detach().cpu().numpy() *
                self.temperature,
                bins='auto'))

        #####################################################
        # Computes the terms for NearestNeighbour NTXenLoss
        # loss_i
        #####################################################

        max_ii = torch.max(sim_zii - diag_inf, dim=1)
        max_ij = torch.max(sim_zij - diag_inf, dim=1)

        # Computes nearest-neighbour of z_i
        z_nn_i = torch.zeros(z_i.shape, device=z_i.device)
        for i in range(N):
            if max_ii.values[i] > max_ij.values[i]:
                z_nn_i[i] = z_i[max_ii.indices[i]]
            else:
                z_nn_i[i] = z_j[max_ij.indices[i]]

        # dim [N, N] => Upper triangle contains incorrect pairs (nn(i),i+)
        sim_nn_zii = (z_nn_i @ z_i.T) / self.temperature

        # dim [N, N] => the diag contains the correct pairs (nn(i),j)
        sim_nn_zij = (z_nn_i @ z_j.T) / self.temperature

        # 'Remove' the covariant vectors by penalizing it (exp(-inf) = 0)
        for i in range(N):
            if max_ii.values[i] > max_ij.values[i]:
                sim_nn_zii[i, max_ii.indices[i]] = -self.INF
            else:
                sim_nn_zij[i, max_ij.indices[i]] = -self.INF

        # 'Remove' the diag terms by penalizing it (exp(-inf) = 0)
        sim_nn_zii = sim_nn_zii - diag_inf

        # Computes nearest neighbour contrastive loss for first view i
        correct_pairs = torch.arange(N, device=z_i.device).long()
        loss_i = func.cross_entropy(torch.cat([sim_nn_zij, sim_nn_zii], dim=1),
                                    correct_pairs)

        #####################################################
        # Computes the terms for NearestNeighbour NTXenLoss
        # loss_j
        #####################################################

        max_jj = torch.max(sim_zjj - diag_inf, dim=1)
        max_ji = torch.max(sim_zji - diag_inf, dim=1)

        # Computes nearest-neighbour of z_j
        z_nn_j = torch.zeros(z_j.shape, device=z_j.device)
        for i in range(N):
            if max_jj.values[i] > max_ji.values[i]:
                z_nn_j[i] = z_j[max_jj.indices[i]]
            else:
                z_nn_j[i] = z_i[max_ji.indices[i]]

        # dim [N, N] => Upper triangle contains incorrect pairs (nn(i),i+)
        sim_nn_zjj = (z_nn_j @ z_j.T) / self.temperature

        # dim [N, N] => the diag contains the correct pairs (nn(i),j)
        sim_nn_zji = (z_nn_j @ z_i.T) / self.temperature

        # 'Remove' the covariant vectors by penalizing it (exp(-inf) = 0)
        for i in range(N):
            if max_jj.values[i] > max_ji.values[i]:
                sim_nn_zjj[i, max_jj.indices[i]] = -self.INF
            else:
                sim_nn_zji[i, max_ji.indices[i]] = -self.INF

        # 'Remove' the diag terms by penalizing it (exp(-inf) = 0)
        sim_nn_zjj = sim_nn_zjj - diag_inf

        # Computes nearest neighbour contrastive loss for first view i
        loss_j = func.cross_entropy(torch.cat([sim_nn_zji, sim_nn_zjj], dim=1),
                                    correct_pairs)

        if self.return_logits:
            return (loss_i + loss_j), sim_zij, correct_pairs

        return (loss_i + loss_j)

    def __str__(self):
        return "{}(temp={})".format(type(self).__name__, self.temperature)


class NTXenLoss_WithoutHardNegative(nn.Module):
    """
    Normalized Temperature Cross-Entropy Loss for Constrastive Learning
    Refer for instance to:
    Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton
    A Simple Framework for Contrastive Learning of Visual Representations,
    arXiv 2020
    """

    def __init__(self, temperature=0.1, return_logits=False):
        super().__init__()
        self.temperature = temperature
        self.INF = 1e8
        self.return_logits = return_logits

    def forward(self, z_i, z_j):
        N = len(z_i)
        z_i = func.normalize(z_i, p=2, dim=-1)  # dim [N, D]
        z_j = func.normalize(z_j, p=2, dim=-1)  # dim [N, D]

        # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zii = (z_i @ z_i.T) / self.temperature

        # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zjj = (z_j @ z_j.T) / self.temperature

        # dim [N, N] => the diag contains the correct pairs (i,j)
        # (x transforms via T_i and T_j)
        sim_zij = (z_i @ z_j.T) / self.temperature

        # Diagonals as 1D tensor
        diag_ij = sim_zij.diagonal()

        # Prints quantiles of positive pairs (views from the same image)
        quantile_positive_pairs = diag_ij.quantile(0.75)
        log.info(
            f"quantile of positives ij = "
            f"{quantile_positive_pairs.cpu()*self.temperature*100}")

        # Computes quantiles of negative pairs
        quantile_negative_ii = quantile_off_diagonal(sim_zii)
        quantile_negative_jj = quantile_off_diagonal(sim_zjj)
        quantile_negative_ij = quantile_off_diagonal(sim_zij)

        # Prints quantiles of negative pairs
        log.info(
            f"quantile of negatives ii = "
            f"{quantile_negative_ii.cpu()*self.temperature*100}")
        log.info(
            f"quantile of negatives jj = "
            f"{quantile_negative_jj.cpu()*self.temperature*100}")
        log.info(
            f"quantile of negatives ij = "
            f"{quantile_negative_ij.cpu()*self.temperature*100}")

        # 'Remove' the diag terms by penalizing it (exp(-inf) = 0)
        sim_zii = sim_zii - self.INF * torch.eye(N, device=z_i.device)
        sim_zjj = sim_zjj - self.INF * torch.eye(N, device=z_i.device)

        # 'Remove' the parts that are hard negatives to promote clustering
        sim_zii[sim_zii > quantile_negative_ii] = -self.INF
        sim_zjj[sim_zii > quantile_negative_jj] = -self.INF

        negative_ij = sim_zij - diag_ij.diag()
        negative_ij[negative_ij > quantile_negative_ij] = -self.INF
        negative_ij.fill_diagonal_(0.)
        sim_zij = negative_ij + diag_ij.diag()

        correct_pairs = torch.arange(N, device=z_i.device).long()
        loss_i = func.cross_entropy(torch.cat([sim_zij, sim_zii], dim=1),
                                    correct_pairs)
        loss_j = func.cross_entropy(torch.cat([sim_zij.T, sim_zjj], dim=1),
                                    correct_pairs)

        if self.return_logits:
            return (loss_i + loss_j), sim_zij, correct_pairs

        return (loss_i + loss_j)

    def __str__(self):
        return "{}(temp={})".format(type(self).__name__, self.temperature)


class NTXenLoss_Mixed(nn.Module):
    """
    Normalized Temperature Cross-Entropy Loss for Constrastive Learning
    Refer for instance to:
    Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton
    A Simple Framework for Contrastive Learning of Visual Representations,
    arXiv 2020
    """

    def __init__(self, temperature=0.1, return_logits=False):
        super().__init__()
        self.temperature = temperature
        self.INF = 1e8
        self.return_logits = return_logits

    def forward_NearestNeighbours_OtherView(self, z_i, z_j):
        N = len(z_i)
        diag_inf = self.INF * torch.eye(N, device=z_i.device)

        #####################################################
        # Computes the classical terms for NTXenLoss
        #####################################################

        log.info("histogram of z_i before normalization:")
        log.info(np.histogram(z_i.detach().cpu().numpy() * 100, bins='auto'))

        z_i = func.normalize(z_i, p=2, dim=-1)  # dim [N, D]
        z_j = func.normalize(z_j, p=2, dim=-1)  # dim [N, D]

        log.info("histogram of z_i after normalization:")
        log.info(np.histogram(z_i.detach().cpu().numpy() * 100, bins='auto'))

        # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zii = (z_i @ z_i.T) / self.temperature

        # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zjj = (z_j @ z_j.T) / self.temperature

        # dim [N, N] => the diag contains the correct pairs (i,j)
        # (x transforms via T_i and T_j)
        sim_zij = (z_i @ z_j.T) / self.temperature
        sim_zji = sim_zij.T

        log.info("histogram of sim_zij:")
        log.info(
            np.histogram(
                sim_zij.detach().cpu().numpy() *
                self.temperature *
                100,
                bins='auto'))

        #####################################################
        # Computes the terms for NearestNeighbour NTXenLoss
        # loss_i
        #####################################################

        max_ii = torch.max(sim_zii - diag_inf, dim=1)
        max_ij = torch.max(sim_zij - diag_inf, dim=1)

        # Computes nearest-neighbour of z_i
        z_nn_i = torch.zeros(z_i.shape, device=z_i.device)
        z_nn_i = z_j[max_ij.indices]

        # dim [N, N] => Upper triangle contains incorrect pairs (nn(i),i+)
        sim_nn_zii = (z_nn_i @ z_i.T) / self.temperature

        # dim [N, N] => the diag contains the correct pairs (nn(i),j)
        sim_nn_zij = (z_nn_i @ z_j.T) / self.temperature

        # 'Remove' the covariant vectors by penalizing it (exp(-inf) = 0)
        for i in range(N):
            sim_nn_zij[i, max_ij.indices[i]] = -self.INF

        # 'Remove' the diag terms by penalizing it (exp(-inf) = 0)
        sim_nn_zii = sim_nn_zii - diag_inf

        # Computes nearest neighbour contrastive loss for first view i
        correct_pairs = torch.arange(N, device=z_i.device).long()
        loss_i = func.cross_entropy(torch.cat([sim_nn_zij, sim_nn_zii], dim=1),
                                    correct_pairs)

        #####################################################
        # Computes the terms for NearestNeighbour NTXenLoss
        # loss_j
        #####################################################

        max_jj = torch.max(sim_zjj - diag_inf, dim=1)
        max_ji = torch.max(sim_zji - diag_inf, dim=1)

        # Computes nearest-neighbour of z_j
        z_nn_j = torch.zeros(z_j.shape, device=z_j.device)
        z_nn_j = z_i[max_ji.indices]

        # dim [N, N] => Upper triangle contains incorrect pairs (nn(i),i+)
        sim_nn_zjj = (z_nn_j @ z_j.T) / self.temperature

        # dim [N, N] => the diag contains the correct pairs (nn(i),j)
        sim_nn_zji = (z_nn_j @ z_i.T) / self.temperature

        # 'Remove' the covariant vectors by penalizing it (exp(-inf) = 0)
        for i in range(N):
            sim_nn_zji[i, max_ji.indices[i]] = -self.INF

        # 'Remove' the diag terms by penalizing it (exp(-inf) = 0)
        sim_nn_zjj = sim_nn_zjj - diag_inf

        # Computes nearest neighbour contrastive loss for first view i
        loss_j = func.cross_entropy(torch.cat([sim_nn_zji, sim_nn_zjj], dim=1),
                                    correct_pairs)

        if self.return_logits:
            return (loss_i + loss_j), sim_zij, correct_pairs

        return (loss_i + loss_j)

    def forward_NearestNeighbours(self, z_i, z_j):
        N = len(z_i)
        diag_inf = self.INF * torch.eye(N, device=z_i.device)

        #####################################################
        # Computes the classical terms for NTXenLoss
        #####################################################

        log.info("histogram of z_i before normalization:")
        log.info(np.histogram(z_i.detach().cpu().numpy() * 100, bins='auto'))

        z_i = func.normalize(z_i, p=2, dim=-1)  # dim [N, D]
        z_j = func.normalize(z_j, p=2, dim=-1)  # dim [N, D]

        log.info("histogram of z_i after normalization:")
        log.info(np.histogram(z_i.detach().cpu().numpy() * 100, bins='auto'))

        # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zii = (z_i @ z_i.T) / self.temperature

        # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zjj = (z_j @ z_j.T) / self.temperature

        # dim [N, N] => the diag contains the correct pairs (i,j)
        # (x transforms via T_i and T_j)
        sim_zij = (z_i @ z_j.T) / self.temperature
        sim_zji = sim_zij.T

        log.info("histogram of sim_zij:")
        log.info(
            np.histogram(
                sim_zij.detach().cpu().numpy() *
                self.temperature *
                100,
                bins='auto'))

        #####################################################
        # Computes the terms for NearestNeighbour NTXenLoss
        # loss_i
        #####################################################

        max_ii = torch.max(sim_zii - diag_inf, dim=1)
        max_ij = torch.max(sim_zij - diag_inf, dim=1)

        # Computes nearest-neighbour of z_i
        z_nn_i = torch.zeros(z_i.shape, device=z_i.device)
        for i in range(N):
            if max_ii.values[i] > max_ij.values[i]:
                z_nn_i[i] = z_i[max_ii.indices[i]]
            else:
                z_nn_i[i] = z_j[max_ij.indices[i]]

        # dim [N, N] => Upper triangle contains incorrect pairs (nn(i),i+)
        sim_nn_zii = (z_nn_i @ z_i.T) / self.temperature

        # dim [N, N] => the diag contains the correct pairs (nn(i),j)
        sim_nn_zij = (z_nn_i @ z_j.T) / self.temperature

        # 'Remove' the covariant vectors by penalizing it (exp(-inf) = 0)
        for i in range(N):
            if max_ii.values[i] > max_ij.values[i]:
                sim_nn_zii[i, max_ii.indices[i]] = -self.INF
            else:
                sim_nn_zij[i, max_ij.indices[i]] = -self.INF

        # 'Remove' the diag terms by penalizing it (exp(-inf) = 0)
        sim_nn_zii = sim_nn_zii - diag_inf

        # Computes nearest neighbour contrastive loss for first view i
        correct_pairs = torch.arange(N, device=z_i.device).long()
        loss_i = func.cross_entropy(torch.cat([sim_nn_zij, sim_nn_zii], dim=1),
                                    correct_pairs)

        #####################################################
        # Computes the terms for NearestNeighbour NTXenLoss
        # loss_j
        #####################################################

        max_jj = torch.max(sim_zjj - diag_inf, dim=1)
        max_ji = torch.max(sim_zji - diag_inf, dim=1)

        # Computes nearest-neighbour of z_j
        z_nn_j = torch.zeros(z_j.shape, device=z_j.device)
        for i in range(N):
            if max_jj.values[i] > max_ji.values[i]:
                z_nn_j[i] = z_j[max_jj.indices[i]]
            else:
                z_nn_j[i] = z_i[max_ji.indices[i]]

        # dim [N, N] => Upper triangle contains incorrect pairs (nn(i),i+)
        sim_nn_zjj = (z_nn_j @ z_j.T) / self.temperature

        # dim [N, N] => the diag contains the correct pairs (nn(i),j)
        sim_nn_zji = (z_nn_j @ z_i.T) / self.temperature

        # 'Remove' the covariant vectors by penalizing it (exp(-inf) = 0)
        for i in range(N):
            if max_jj.values[i] > max_ji.values[i]:
                sim_nn_zjj[i, max_jj.indices[i]] = -self.INF
            else:
                sim_nn_zji[i, max_ji.indices[i]] = -self.INF

        # 'Remove' the diag terms by penalizing it (exp(-inf) = 0)
        sim_nn_zjj = sim_nn_zjj - diag_inf

        # Computes nearest neighbour contrastive loss for first view i
        loss_j = func.cross_entropy(torch.cat([sim_nn_zji, sim_nn_zjj], dim=1),
                                    correct_pairs)

        if self.return_logits:
            return (loss_i + loss_j), sim_zij, correct_pairs

        return (loss_i + loss_j)

    def forward_WithoutHardNegative(self, z_i, z_j):
        N = len(z_i)
        z_i = func.normalize(z_i, p=2, dim=-1)  # dim [N, D]
        z_j = func.normalize(z_j, p=2, dim=-1)  # dim [N, D]

        # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zii = (z_i @ z_i.T) / self.temperature

        # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zjj = (z_j @ z_j.T) / self.temperature

        # dim [N, N] => the diag contains the correct pairs (i,j)
        # (x transforms via T_i and T_j)
        sim_zij = (z_i @ z_j.T) / self.temperature

        # Diagonals as 1D tensor
        diag_ij = sim_zij.diagonal()

        # Prints quantiles of positive pairs (views from the same image)
        quantile_positive_pairs = diag_ij.quantile(0.75)
        log.info(
            f"quantile of positives ij = "
            f"{quantile_positive_pairs.cpu()*self.temperature*100}")

        # Computes quantiles of negative pairs
        quantile_negative_ii = quantile_off_diagonal(sim_zii)
        quantile_negative_jj = quantile_off_diagonal(sim_zjj)
        quantile_negative_ij = quantile_off_diagonal(sim_zij)

        # Prints quantiles of negative pairs
        log.info(
            f"quantile of negatives ii = "
            f"{quantile_negative_ii.cpu()*self.temperature*100}")
        log.info(
            f"quantile of negatives jj = "
            f"{quantile_negative_jj.cpu()*self.temperature*100}")
        log.info(
            f"quantile of negatives ij = "
            f"{quantile_negative_ij.cpu()*self.temperature*100}")

        # 'Remove' the diag terms by penalizing it (exp(-inf) = 0)
        sim_zii = sim_zii - self.INF * torch.eye(N, device=z_i.device)
        sim_zjj = sim_zjj - self.INF * torch.eye(N, device=z_i.device)

        # 'Remove' the parts that are hard negatives to promote clustering
        sim_zii[sim_zii > quantile_negative_ii] = -self.INF
        sim_zjj[sim_zjj > quantile_negative_jj] = -self.INF

        # 'Remove' the parts that are hard negatives to promote clustering
        # We keep the positive element j (second view)
        negative_ij = sim_zij - diag_ij.diag()
        negative_ij[negative_ij > quantile_negative_ij] = -self.INF
        negative_ij.fill_diagonal_(0.)
        sim_zij = negative_ij + diag_ij.diag()

        correct_pairs = torch.arange(N, device=z_i.device).long()
        loss_i = func.cross_entropy(torch.cat([sim_zij, sim_zii], dim=1),
                                    correct_pairs)
        loss_j = func.cross_entropy(torch.cat([sim_zij.T, sim_zjj], dim=1),
                                    correct_pairs)

        if self.return_logits:
            return (loss_i + loss_j), sim_zij, correct_pairs

        return (loss_i + loss_j)

    def forward(self, z_i, z_j):
        loss_NN, _, _ = self.forward_NearestNeighbours_OtherView(z_i, z_j)
        loss_WHN, sim_zij, correct_pairs = self.forward_WithoutHardNegative(
            z_i, z_j)

        if self.return_logits:
            return ((loss_NN + loss_WHN) / 2), sim_zij, correct_pairs

        return ((loss_NN + loss_WHN) / 2)

    def __str__(self):
        return "{}(temp={})".format(type(self).__name__, self.temperature)
