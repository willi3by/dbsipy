# testing specified individual voxels

# utils
import os
import sys
sys.path.append(os.path.abspath('./'))
from tqdm import tqdm

# data
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.optimize as optimize
from scipy.io import loadmat
from scipy.signal import argrelextrema
from sklearn.mixture import GaussianMixture
import ants # handle .nii files

# dbsi
from dbsipy.dbsi.dbsi_utilities import *
from dbsipy.dbsi.dbsi_equations import *
from dbsipy.dbsi.dbsi_run import *


# load image, diffusion gradient directions, and b-values
dbsi_data, dbsi_mask = load_image('./dbsipy/data/dbsi_dataset.nii')
bvecs = np.loadtxt('./dbsi_bvec.txt')[:]
bvals = np.loadtxt('./dbsi_bval.txt')[:]
# basis tensors
dbasis = loadmat('./dbsipy/data/dbset_100.mat')
basis_tensors = dbasis['xx'][:95]

# problems voxel
i = 49
j = 64
k = 32
b = 1
s_k = dbsi_data[i, j, k, :]
# sorting
bvecs = bvecs[np.argsort(bvals)]
dbsi_data = dbsi_data[:, :, :, np.argsort(bvals)]
bvals = np.sort(bvals)
# taking only last 26 values
bvecs = bvecs[1:]
bvals = bvals[1:]
dbsi_data = dbsi_data[:, :, :, 1:]
s_k = dbsi_data[i, j, k, :]

# prediction
fa, primary_aniso_ff, primary_axial_diffusivity, primary_radial_diffusivity, restricted_ff, hindered_ff, free_water_ff, primary_aniso, iso_ff = fit_voxel_data(s_k, basis_tensors, bvecs, bvals)
s_k_predicted = dbsi_predict(bvecs, bvals, primary_aniso, primary_aniso_ff, iso_ff, [primary_radial_diffusivity, primary_axial_diffusivity])
print(s_k_predicted)
