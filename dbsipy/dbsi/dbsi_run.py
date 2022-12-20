import numpy as np
from dbsi.dbsi_utilities import *
from dbsi.dbsi_equations import *
#%% Perform voxel fit.

def fit_voxel_data(s_k, basis_tensors, bvecs, bvals):
    opt_c_array = optimize_c_array(basis_tensors, bvecs, bvals, s_k)
    opt_init_diffusivities = optimize_diffusivity_array(basis_tensors, bvecs,
                                                       bvals, s_k, opt_c_array)
    aniso_c_array = opt_c_array[:95]
    if np.sum(aniso_c_array) > 0:
        clustered_c = cluster_aniso_c_array(aniso_c_array)
        aniso_tensors = get_aniso_tensors(clustered_c, aniso_c_array, basis_tensors)
        if len(aniso_tensors) > 0:
            lambda_perp = opt_init_diffusivities[0]
            lambda_par = opt_init_diffusivities[1]
            lambda_perp_aniso = np.full((len(aniso_tensors)), opt_init_diffusivities[0])
            lambda_par_aniso = np.full((len(aniso_tensors)), opt_init_diffusivities[1])
            diffusivity_array = np.concatenate([lambda_perp_aniso, lambda_par_aniso])
            opt_f_i = optimize_fiber_fraction(aniso_tensors, bvecs, bvals,
                                              diffusivity_array, s_k)
            opt_fiber_diffusivities = optimize_fiber_diffusivities(diffusivity_array,
                                                                   aniso_tensors, bvecs,
                                                                   bvals, opt_f_i, s_k)
            fa, primary_aniso_ff, primary_axial_diffusivity, primary_radial_diffusivity, restricted_ff, hindered_ff, free_water_ff = extract_metrics(
                opt_f_i, opt_fiber_diffusivities)
        else:
            fa = 0
            primary_aniso_ff = 0
            primary_axial_diffusivity = 0
            primary_radial_diffusivity = 0
            restricted_ff = 0
            hindered_ff = 0
            free_water_ff = 1
    else:
        fa = 0
        primary_aniso_ff = 0
        primary_axial_diffusivity = 0
        primary_radial_diffusivity = 0
        restricted_ff = 0
        hindered_ff = 0
        free_water_ff = 1

    return fa, primary_aniso_ff, primary_axial_diffusivity, primary_radial_diffusivity, restricted_ff, hindered_ff, free_water_ff
#%%
def fit_slice(image_slice, mask, basis_tensors, bvecs, bvals):
    x = image_slice.shape[0]
    y = image_slice.shape[1]
    fa_map = np.empty((x,y))
    aniso_ff_map = np.empty((x,y))
    axial_diffusivity_map = np.empty((x,y))
    radial_diffusivity_map = np.empty((x,y))
    restricted_ff_map = np.empty((x,y))
    hindered_ff_map = np.empty((x,y))
    free_water_ff_map = np.empty((x,y))
    for i in range(x):
        for j in range(y):
            if mask[i,j] == 1:
                s_k = image_slice[i,j,:]
                fa, primary_aniso_ff, primary_axial_diffusivity, primary_radial_diffusivity, restricted_ff, hindered_ff, free_water_ff = fit_voxel_data(s_k, basis_tensors, bvecs, bvals)
                fa_map[i,j] = fa
                if isinstance(primary_aniso_ff, int):
                    aniso_ff_map[i, j] = 0
                else:
                    aniso_ff_map[i,j] = primary_aniso_ff[0]
                if isinstance(primary_axial_diffusivity, int):
                    axial_diffusivity_map[i, j] = 0
                else:
                    axial_diffusivity_map[i,j] = primary_axial_diffusivity[0]
                if isinstance(primary_radial_diffusivity, int):
                    radial_diffusivity_map[i, j] = 0
                else:
                    radial_diffusivity_map[i,j] = primary_radial_diffusivity[0]
                restricted_ff_map[i,j] = restricted_ff
                hindered_ff_map[i,j] = hindered_ff
                free_water_ff_map[i,j] = free_water_ff
            else:
                fa_map[i, j] = 0
                aniso_ff_map[i, j] = 0
                axial_diffusivity_map[i, j] = 0
                radial_diffusivity_map[i, j] = 0
                restricted_ff_map[i, j] = 0
                hindered_ff_map[i, j] = 0
                free_water_ff_map[i, j] = 0

    return fa_map, aniso_ff_map, axial_diffusivity_map, radial_diffusivity_map, restricted_ff_map, hindered_ff_map, free_water_ff_map