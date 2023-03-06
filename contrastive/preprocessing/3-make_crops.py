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
# knowledge of the CeCILL license version 2 and that you ac8

# This script defines the bounding box of the F.C.M.ant (cingular) sulcus
# It is not meant to be generic.
# It is meant for documentation and reproducibility.
# It permits to easily regenerate the bounding boxes and masks

# Imports
from os.path import abspath
from deep_folding.anatomist_tools import dataset_gen_pipe

# Defines source and targets directories
crop_dir = "../../../../Input/Processed_Local/crops/CINGULATE"\
           "/mask/sulcus_based/2mm/centered_combined/hcp"
mask_dir = "../../../../Input/Processed_Local/mask/2mm"

crop_dir = abspath(crop_dir)
mask_dir = abspath(mask_dir)

# Defines other parameters
number_subjects = 10  # 0 is for testing, "all" means all subjects


# Defines the command line parameters for dataset_gen_pipe
args = "-s /tgcc/hcp "\
       f"-t {crop_dir} "\
       f"-a {mask_dir} "\
       "-u F.C.M.ant. paracingular. "\
       f"-i R -p s -c mask -v 2 2 2 -o True -n {number_subjects}"
argv = args.split(' ')

print("")
print("The equivalent command-line is the following:")
print("---------------------------------------------")
print("python3 dataset_gen_pipe.py " + args)
print("---------------------------------------------")
print("")

# Defines the mask and the crops
# and writes the result in bbox_dir and mask_dir
dataset_gen_pipe.main(argv)
