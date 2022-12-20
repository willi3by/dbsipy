#!/bin/bash/python

import ants
from scipy.io import loadmat
from joblib import Parallel, delayed
from dbsi.dbsi_run import fit_slice
from dbsi.dbsi_utilities import load_image
import numpy as np
import os

slice_number = int(os.environ['SLURM_ARRAY_TASK_ID']) - 1
output_path = '/N/lustre/scratch/willi3by/dbsi_data/'

def do_dbsi_slice(slice_number, output_path):
    dbasis = loadmat('./data/dbset_100.mat')
    basis_tensors = dbasis['xx'][:95]
    bvecs = np.loadtxt('/home/willi3by/dbsi_bvec.txt')[1:]
    bvals = np.loadtxt('/home/willi3by/dbsi_bval.txt')[1:]
    dbsi_data, dbsi_mask = load_image('./data/dbsi_dataset.nii')
    image_slice = dbsi_data[:, :, slice_number, 1:]
    mask = dbsi_mask[:, :, slice_number]
    L = 100

    fa_map, aniso_ff_map, axial_diffusivity_map, radial_diffusivity_map, restricted_ff_map, hindered_ff_map, free_water_ff_map = fit_slice(
        image_slice, mask, basis_tensors, bvecs, bvals, L)

    fa_file_name = output_path + 'fa_slice_' + str(slice_number) + '.nii'
    aniso_ff_file_name = output_path + 'aniso_ff_slice_' + str(slice_number) + '.nii'
    axial_diff_file_name = output_path + 'axial_diff_slice_' + str(slice_number) + '.nii'
    radial_diff_file_name = output_path + 'radial_diff_slice_' + str(slice_number) + '.nii'
    restricted_ff_file_name = output_path + 'restricted_slice_' + str(slice_number) + '.nii'
    hindered_ff_file_name = output_path + 'hindered_slice_' + str(slice_number) + '.nii'
    free_water_ff_file_name = output_path + 'free_water_slice_' + str(slice_number) + '.nii'

    ants.image_write(ants.from_numpy(fa_map), filename=fa_file_name)
    ants.image_write(ants.from_numpy(aniso_ff_map), filename=aniso_ff_file_name)
    ants.image_write(ants.from_numpy(axial_diffusivity_map), filename=axial_diff_file_name)
    ants.image_write(ants.from_numpy(radial_diffusivity_map), filename=radial_diff_file_name)
    ants.image_write(ants.from_numpy(restricted_ff_map), filename=restricted_ff_file_name)
    ants.image_write(ants.from_numpy(hindered_ff_map), filename=hindered_ff_file_name)
    ants.image_write(ants.from_numpy(free_water_ff_map), filename=free_water_ff_file_name)

do_dbsi_slice(slice_number, output_path)
