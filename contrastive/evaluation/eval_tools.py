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
""" Diverse tools to analyse training results

"""
######################################################################
# Imports and global variables definitions
######################################################################
import itertools
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stat
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchvision.utils import save_image
import pytorch_ssim
# https://github.com/jinh0park/pytorch-ssim-3D


def plot_loss(list_loss_train, list_loss_val, root_dir):
    """
    Plot training loss given two lists of loss
    list_loss_train: list of loss values of training set
    list_loss_val: list of validation set loss values
    """
    plt.clf()
    plt.subplot()
    epoch = [k for k in range(1, len(list_loss_train) + 1)]
    plt.plot(epoch, list_loss_train, label='Train')
    plt.plot(epoch, list_loss_val, label='Validation')
    plt.xlabel('Number of epochs')
    plt.ylabel('Loss value')
    plt.legend()
    plt.savefig(root_dir + "loss.png")


def plot_trajectories(loss_dict, nb_epoch, root_dir):
    """
    Plot error trajectories for all subjects
    loss_dict : dictionary containing loss values for each subject at each
    epoch
    root_dir : folder directory in which saving figure
    """

    plt.clf()

    sc_int = ['111009', '138231', '140319', '159946', '199251', '212419',
              '510225']
    epoch = [k for k in range(1, nb_epoch + 1)]
    dico_int = {
        key: value for key,
        value in loss_dict.items() if key in sc_int}
    dico_cont = {
        key: value for key,
        value in loss_dict.items() if key not in sc_int}

    plt.subplot()
    for subject in dico_int.keys():
        plt.plot(epoch, dico_int[subject], label=subject)
    plt.plot(epoch,
             [np.mean([dico_cont[key][k] for key in dico_cont.keys()])
              for k in range(nb_epoch)],
             label='Average continuous sulci error')
    plt.xlabel('Number of epochs')
    plt.ylabel('Loss value')
    plt.legend()
    plt.savefig(root_dir + "trajectories.png")

    plt.clf()
    plt.subplot()
    epoch = [k for k in range(2, nb_epoch + 1)]
    for subject in dico_int.keys():
        plt.plot(epoch, dico_int[subject][1:], label=subject)
    plt.plot(epoch,
             [np.mean([dico_cont[key][k] for key in dico_cont.keys()])
              for k in range(1, nb_epoch)],
             label='Average continuous sulci error')
    plt.xlabel('Number of epochs')
    plt.ylabel('Loss value')
    plt.legend()
    plt.savefig(root_dir + "trajectories_wo_start.png")

    plt.clf()
    fig, ax = plt.subplots(1)
    epoch = [k for k in range(2, nb_epoch + 1)]
    ave_int = [np.mean([dico_int[key][k] for key in dico_int.keys()])
               for k in range(1, nb_epoch)]
    ave_cont = [np.mean([dico_cont[key][k] for key in dico_cont.keys()])
                for k in range(1, nb_epoch)]
    sigma = [np.std([dico_cont[key][k] for key in dico_cont.keys()])
             for k in range(1, nb_epoch)]
    # sigma = np.std(ave_cont)
    lower_bound = [ave_cont[k] - sigma[k] for k in range(len(ave_cont))]
    upper_bound = [ave_cont[k] + sigma[k] for k in range(len(ave_cont))]
    ax.plot(epoch, ave_int, lw=2, label="Average interrupted sulci error")
    ax.plot(
        epoch,
        ave_cont,
        lw=1,
        label='Average continuous sulci error',
        ls='--')
    ax.fill_between(
        epoch,
        lower_bound,
        upper_bound,
        facecolor='yellow',
        alpha=0.5,
        label='1 sigma range')
    ax.set_xlabel('Number of epochs')
    ax.set_ylabel('Loss value')
    ax.legend(loc='upper right')
    ax.grid()
    fig.savefig(root_dir + "trajectories_ave.png")


def plot_auc(auc_dict, nb_epoch, root_dir):

    sc_int = ['111009', '138231', '140319', '159946', '199251', '212419',
              '510225']
    plt.clf()
    fig, ax = plt.subplots(1)
    epoch = [k for k in range(nb_epoch)]

    auc_ave = [np.mean([auc_dict[key][k] for key in auc_dict.keys()])
               for k in range(nb_epoch)]

    q1_list = [stat.mstats.mquantiles(
        [auc_dict[key][k] for key in auc_dict.keys()],
        prob=[0.25, 0.75])[0] for k in range(nb_epoch)]
    q3_list = [stat.mstats.mquantiles(
        [auc_dict[key][k] for key in auc_dict.keys()],
        prob=[0.25, 0.75])[1] for k in range(nb_epoch)]

    auc_min = [np.min([auc_dict[key][k] for key in auc_dict.keys()])
               for k in range(nb_epoch)]
    auc_max = [np.max([auc_dict[key][k] for key in auc_dict.keys()])
               for k in range(nb_epoch)]

    outlier_list = {key: value for key, value in zip(
        [k for k in range(nb_epoch)], [[] for k in range(nb_epoch)])}
    for k in range(nb_epoch):
        # for loss values > 0
        outlier_list[k] = \
            [value[k] for key, value in auc_dict.items()
                if auc_dict[key][k] > min(auc_max[k],
                                          q3_list[k] +
                                          1.5 * (q3_list[k] - q1_list[k]))]
        # for SSIM (loss values <0)
        """outlier_list[k] = [value[k] for key, value in auc_dict.items() if  \
           auc_dict[key][k] < max(auc_min[k],
            q1_list[k] - 1.5*(q3_list[k]-q1_list[k]))]"""

    outlier_min = [min(outlier_list[k]) if outlier_list[k]
                   != [] else 0 for k in range(nb_epoch)]
    outlier_max = [max(outlier_list[k]) if outlier_list[k]
                   != [] else 0 for k in range(nb_epoch)]

    for subject in sc_int:
        if subject in auc_dict.keys():
            ax.scatter(epoch, auc_dict[subject], label=subject)

    ax.plot(epoch, auc_ave, lw=2, label="Average AUC")
    ax.fill_between(
        epoch,
        outlier_min,
        outlier_max,
        facecolor='yellow',
        alpha=0.5,
        label='outlier range')
    ax.set_xlabel('Number of epochs')
    ax.set_ylabel('Loss value')
    ax.legend(loc='upper left')
    ax.grid()
    fig.savefig(root_dir + "auc_trajectories.png")


def compute_loss(dico_set_loaders, model, loss_type, root_dir):
    """
    Returns list of loss values for each dataset_loader batch
    dataset_loader: dataset on which compute loss
    model: trained model used to compute loss
    loss_type: loss function to use to compute loss (L1, L2, SSIM)
    path_mode: if True (possible to track loss to samples)
    show: if True, display output images in Anatomist
    """
    torch.manual_seed(10)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    list_loss = []
    device = torch.device("cuda", index=0)
    model = model.to(device)
    model.eval()
    encoded_out = True

    # root_dir = "/neurospin/dico/lguillon/data/200320_split_L2/"

    dico_sub_loss = dict()

    if loss_type == 'L2':
        distance = nn.MSELoss()
    elif loss_type == 'L1':
        distance = nn.L1Loss()
    elif loss_type == 'CrossEnt':
        # weights = [1, 1, 8]
        # print(weights)
        weights = [1, 2]
        class_weights = torch.FloatTensor(weights).to(device)
        distance = nn.CrossEntropyLoss(weight=class_weights)

    results = {k: {} for k in dico_set_loaders.keys()}

    print(loss_type)
    for loader_name, loader in dico_set_loaders.items():
        print(loader_name)
        with torch.no_grad():
            for img, path in loader:
                if path[0] not in [
                    '681998875857',
                    '855494225893',
                    '927090337769',
                        '716588902839']:
                    phase = loader_name
                    img = Variable(img).to(device, dtype=torch.float)
                    output, encoded = model(img)
                    encoded = torch.flatten(encoded)
                    if loss_type == 'SSIM':
                        loss = pytorch_ssim.ssim3D(output, img)
                    else:
                        if 'skeleton' in root_dir:
                            target = torch.squeeze(img, dim=0).long()
                            loss = distance(output, target)
                            output = torch.argmax(output, dim=1)
                        else:
                            loss = distance(output, img)
                            error_image = img - output

                            weight = torch.ones(
                                11, 11, 11).unsqueeze(0).unsqueeze(0).to(
                                device, dtype=torch.float)
                            out_conv = F.conv3d(
                                error_image, weight, stride=1, padding=5)
                    results[loader_name][path] = (
                        loss.item(), output, img, list(
                            encoded.squeeze().cpu().detach().numpy()))

    # Saving of outliers
    for loader_name in results.keys():
        input_arr = []
        output_arr = []
        phase_arr = []
        id_arr = []
        error_arr = []
        out_conv_arr = []

        print(loader_name)
        quantile = stat.mstats.mquantiles(
            [res[0] for res in results[loader_name].values()],
            prob=[0.25, 0.75])

        average = np.mean([res[0] for res in results[loader_name].values()])
        var = np.std([res[0] for res in results[loader_name].values()])
        print(
            "For ",
            loader_name,
            "quantile :",
            quantile,
            "average :",
            average,
            "Variance: ",
            var)
        for key, value in results[loader_name].items():
            if value[0] > min(max([res[0]
                                   for res in results[loader_name].values()]),
                              quantile[1] + 1.5 * (quantile[1] - quantile[0])):
                print(key, value[0])
            # for k in range(len(results[loader_name])):
            id_arr.append(key)
            phase_arr.append(loader_name)
            input_arr.append(
                np.array(
                    np.squeeze(
                        value[2]).cpu().detach().numpy()))
            output_arr.append(np.squeeze(value[1]).cpu().detach().numpy())
            # error_arr.append(np.squeeze(value[3]).cpu().detach().numpy())
            # out_conv_arr.append(np.squeeze(value[4]).cpu().detach().numpy())

        print(root_dir)
        for key, array in {'input': input_arr, 'output': output_arr,
                           'id': id_arr, 'phase': phase_arr}.items():
            np.save(root_dir + str(loader_name) + '_' + key, np.array([array]))

    if encoded_out:
        return {loader_name: [res for res in results[loader_name].values()]
                for loader_name in dico_set_loaders}
    else:
        return {loader_name: [res[0] for res in results[loader_name].values()]
                for loader_name in dico_set_loaders}


def plot_distrib(loss_nor, root_dir, *loss_abnor):
    """ Plots distribution of "normal subjects" loss list and "abnormal
    subjects" loss list
    loss_nor, loss_abnor : 2 lists to plot distribution
    """
    plt.clf()
    plt.subplot()
    plt.hist(loss_nor, color='skyblue', bins=20,
             label="Continuous central sulcus")
    if loss_abnor:
        plt.hist(loss_abnor, color='salmon', bins=20,
                 label="Interrupted central sulcus")
    plt.xlabel('Loss values')
    plt.ylabel('Number of subjects')
    plt.title("Loss distributions for continuous and interrupted CS")
    plt.legend()
    plt.savefig(root_dir + "distrib.png")


def get_outliers(skeleton, dico_set_loaders, model, loss_type):
    """
    Print outliers of a given model for each dataset of dico_set_loaders
    IN:
        skeleton: True/False, whether input are skeleton
        dico_set_loaders: dictionary with keys corresponding to name of
            different populations to compare and values, associated dataloader
            model: model trained
        loss_type: 'L2'/'CrossEnt'
    OUT:
    """
    results = test_model(skeleton, dico_set_loaders, model, loss_type)

    # Displaying of outliers
    for loader_name in results.keys():
        print(loader_name)
        quantile = stat.mstats.mquantiles(
            [res[0] for res in results[loader_name].values()],
            prob=[0.25, 0.75])

        average = np.mean([res[0] for res in results[loader_name].values()])
        var = np.std([res[0] for res in results[loader_name].values()])
        print(
            "For ",
            loader_name,
            "quantile :",
            quantile,
            "average :",
            average,
            "Variance: ",
            var)
        for key, value in results[loader_name].items():
            if value[0] > min(max([res[0]
                                   for res in results[loader_name].values()]),
                              quantile[1] + 1.5 * (quantile[1] - quantile[0])):
                print(key, value[0])


def test_model(skeleton, dico_set_loaders, model, loss_type):
    """

    """
    list_loss = []
    device = torch.device("cuda", index=0)
    model = model.to(device)
    model.eval()
    encoded_out = True
    dico_sub_loss = dict()

    if loss_type == 'L2':
        distance = nn.MSELoss()
    elif loss_type == 'L1':
        distance = nn.L1Loss()
    elif loss_type == 'CrossEnt':
        weights = [1, 2]
        class_weights = torch.FloatTensor(weights).to(device)
        distance = nn.CrossEntropyLoss(weight=class_weights)

    results = {k: {} for k in dico_set_loaders.keys()}

    for loader_name, loader in dico_set_loaders.items():
        print(loader_name)
        with torch.no_grad():
            for img, path in loader:
                if path[0] not in [
                    '681998875857',
                    '855494225893',
                    '927090337769',
                        '716588902839']:
                    img = Variable(img).to(device, dtype=torch.float)
                    output, encoded = model(img)
                    encoded = torch.flatten(encoded)
                    if 'skeleton':
                        target = torch.squeeze(img, dim=0).long()
                        loss = distance(output, target)
                        output = torch.argmax(output, dim=1)
                    else:
                        loss = distance(output, img)
                results[loader_name][path] = (
                    loss.item(), output, img, list(
                        encoded.squeeze().cpu().detach().numpy()))

    return results
