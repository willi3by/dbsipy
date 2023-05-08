import numpy as np
from tqdm import tqdm
from dbsipy.dbsi.dbsi_utilities import *
from dbsipy.dbsi.dbsi_equations import *
#%% Perform voxel fit.

def fit_voxel_data(s_k, basis_tensors, bvecs, bvals):
    opt_c_array = optimize_c_array(basis_tensors, bvecs, bvals, s_k)
    opt_init_diffusivities = optimize_diffusivity_array(basis_tensors, bvecs, bvals, s_k, opt_c_array)
    aniso_c_array = opt_c_array[:-1]
    if np.sum(aniso_c_array) > 0:
        clustered_c = cluster_aniso_c_array(aniso_c_array)
        aniso_tensors = get_aniso_tensors(clustered_c, aniso_c_array, basis_tensors)
        if len(aniso_tensors) > 0:
            lambda_perp = opt_init_diffusivities[0]
            lambda_par = opt_init_diffusivities[1]
            lambda_perp_aniso = np.full((len(aniso_tensors)), lambda_perp)
            lambda_par_aniso = np.full((len(aniso_tensors)), lambda_par)
            diffusivity_array = np.concatenate([lambda_perp_aniso, lambda_par_aniso])
            
            opt_f_i = optimize_fiber_fraction(aniso_tensors, bvecs, bvals,
                                              diffusivity_array, s_k)
            opt_fiber_diffusivities = optimize_fiber_diffusivities(diffusivity_array,
                                                                   aniso_tensors, bvecs,
                                                                   bvals, opt_f_i, s_k)
            fa, primary_aniso_ff, primary_axial_diffusivity, primary_radial_diffusivity, restricted_ff, hindered_ff, free_water_ff, iso_ff = extract_metrics(
                opt_f_i, opt_fiber_diffusivities)
            # taking the primary aniso_tensor
            primary_aniso = np.amax(np.array(aniso_tensors), axis=0)


        else:
            fa = 0
            primary_aniso_ff = 0
            primary_axial_diffusivity = 0
            primary_radial_diffusivity = 0
            restricted_ff = 0
            hindered_ff = 0
            free_water_ff = 1
            primary_aniso = np.array([0.0,0.0,0.0])
            
            diffusivity_array = np.array([primary_axial_diffusivity, primary_radial_diffusivity])
            iso_ff = optimize_fiber_fraction(primary_aniso[None], bvecs, bvals, diffusivity_array, s_k)
    else:

        fa = 0
        primary_aniso_ff = 0
        primary_axial_diffusivity = 0
        primary_radial_diffusivity = 0
        restricted_ff = 0
        hindered_ff = 0
        free_water_ff = 1
        primary_aniso = np.array([0.0,0.0,0.0])

        diffusivity_array = np.array([primary_axial_diffusivity, primary_radial_diffusivity])
        iso_ff = optimize_fiber_fraction(primary_aniso[None], bvecs, bvals, diffusivity_array, s_k)

    return fa, primary_aniso_ff, primary_axial_diffusivity, primary_radial_diffusivity, restricted_ff, hindered_ff, free_water_ff, primary_aniso, iso_ff
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
    primary_aniso_map = np.empty((x,y,3))
    iso_ff_map = np.empty((x,y,20)) 
    for i in tqdm(range(x)):
        for j in range(y):
            if mask[i,j] == 1:
                s_k = image_slice[i,j,:]
                fa, primary_aniso_ff, primary_axial_diffusivity, primary_radial_diffusivity, restricted_ff, hindered_ff, free_water_ff, primary_aniso, iso_ff = fit_voxel_data(s_k, basis_tensors, bvecs, bvals)
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
                primary_aniso_map[i,j] = primary_aniso
                iso_ff_map[i,j] = iso_ff
            else:
                fa_map[i, j] = 0
                aniso_ff_map[i, j] = 0
                axial_diffusivity_map[i, j] = 0
                radial_diffusivity_map[i, j] = 0
                restricted_ff_map[i, j] = 0
                hindered_ff_map[i, j] = 0
                free_water_ff_map[i, j] = 0
                primary_aniso_map[i,j] = np.array([0.0, 0.0, 0.0])
                iso_ff_map[i,j] = np.zeros(20)

    return fa_map, aniso_ff_map, axial_diffusivity_map, radial_diffusivity_map, restricted_ff_map, hindered_ff_map, free_water_ff_map, primary_aniso_map, iso_ff_map


def dbsi_predict(bvecs, bvals, primary_aniso, primary_aniso_ff, iso_ff, diffusivities):

    if np.sum(primary_aniso) == 0:
        # no anisotrophy
        ff = iso_ff
    else:
        ff = np.concatenate([[primary_aniso_ff], iso_ff])
        
    primary_radial_diffusivity = diffusivities[0]
    primary_axial_diffusivity = diffusivities[1]
    m_matrix = build_M_matrix(primary_aniso[None], bvecs, bvals, [primary_radial_diffusivity], [primary_axial_diffusivity])
    return np.dot(m_matrix, ff)

