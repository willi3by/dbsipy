from scipy.io import loadmat
from joblib import Parallel, delayed
from dbsi.dbsi_run import fit_slice
from dbsi.dbsi_utilities import load_image
import numpy as np

dbasis = loadmat('./data/dbset_100.mat')
basis_tensors = dbasis['xx'][:95]
bvecs = np.loadtxt('/Users/willi3by/Desktop/dbsi_bvec.txt')[1:]
bvals = np.loadtxt('/Users/willi3by/Desktop/dbsi_bval.txt')[1:]
dbsi_data, dbsi_mask = load_image('./data/dbsi_dataset.nii')
image_slice = dbsi_data[:,:,7,1:]
mask = dbsi_mask[:,:,7]
results = Parallel(n_jobs=80)(delayed(fit_slice)(dbsi_data[:, :, i, 1:], dbsi_mask.numpy()[:, :, i], basis_tensors, bvecs, bvals) for i in range(dbsi_data.shape[2]))


fa_map, aniso_ff_map, axial_diffusivity_map, radial_diffusivity_map, restricted_ff_map, hindered_ff_map, free_water_ff_map = fit_slice(dbsi_slice, dbsi_slice_mask, basis_tensors, bvecs, bvals)
