import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.signal import argrelextrema
import ants

def load_image(path_to_file):
    dbsi_img = ants.image_read(path_to_file)
    dbsi_img = dbsi_img.resample_image((2, 2, 2.6, 10))
    dbsi_mask = ants.get_mask(ants.slice_image(dbsi_img, axis=3, idx=(0)))
    dbsi_data = dbsi_img.numpy()
    return dbsi_data, dbsi_mask

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def cluster_aniso_c_array(aniso_c_array):
    n_components = [1, 2, 3, 4, 5]
    all_bic = []
    for i in n_components:
        g = GaussianMixture(n_components=i)
        res = g.fit(aniso_c_array.reshape(-1, 1))
        bic = g.bic(aniso_c_array.reshape(-1, 1))
        all_bic.append(bic)

    local_min = np.array(argrelextrema(np.array(all_bic), np.less))
    if len(local_min[0]) > 0:
        min_idx = local_min[0][len(local_min)-1]
        n_components = n_components[min_idx]
    else:
        n_components = 2
    g = GaussianMixture(n_components=n_components)
    clustered_c = g.fit_predict(aniso_c_array.reshape(-1, 1))
    return clustered_c

def get_aniso_tensors(clustered_c, aniso_c_array, basis_tensors):
    n_aniso = np.max(clustered_c)
    n_aniso_tensors = []
    for i in range(n_aniso+1):
        masked_clustered_c = np.where(clustered_c == i, 1, 0)
        c_weights = aniso_c_array * masked_clustered_c
        if np.sum(c_weights) > 0:
            weighted_basis_tensors = c_weights[:, None] * basis_tensors
            sum_weighted_tensors = np.sum(weighted_basis_tensors, axis=0)
            if np.sum(sum_weighted_tensors) == 0:
                pass
            else:
                weighted_tensor_unit_norm = sum_weighted_tensors / np.sqrt(np.sum(sum_weighted_tensors ** 2))
                n_aniso_tensors.append(weighted_tensor_unit_norm)
        else:
            pass

    return n_aniso_tensors

def calc_fa(eig_1, eig_2, eig_3):
    term_1 = np.sqrt(1 / 2)
    term_2 = np.sqrt((eig_1 - eig_2) ** 2 + (eig_2 - eig_3) ** 2 + (eig_3 - eig_1) ** 2)
    term_3 = np.sqrt(eig_1 ** 2 + eig_2 ** 2 + eig_3 ** 2)
    fa = term_1 * (term_2 / term_3)
    return fa

def extract_metrics(opt_f_i, opt_fiber_diffusivities, L=20):
    opt_f_i_perc = opt_f_i/np.sum(opt_f_i)
    aniso_ff_length = len(opt_f_i) - L
    aniso_ff = opt_f_i_perc[0:aniso_ff_length]
    max_aniso_idx = np.where(aniso_ff == np.max(aniso_ff))
    primary_aniso_ff = aniso_ff[max_aniso_idx]
    iso_ff_idxs = opt_f_i_perc[aniso_ff_length:]
    restricted_ff = np.sum(iso_ff_idxs[0:2])
    hindered_ff = np.sum(iso_ff_idxs[2:12])
    free_water_ff = np.sum(iso_ff_idxs[12:])
    radial_diffusivities = opt_fiber_diffusivities[:aniso_ff_length]
    axial_diffusivities = opt_fiber_diffusivities[aniso_ff_length:]
    primary_radial_diffusivity = radial_diffusivities[max_aniso_idx]
    primary_axial_diffusivity = axial_diffusivities[max_aniso_idx]
    fa = calc_fa(primary_axial_diffusivity, primary_radial_diffusivity, primary_radial_diffusivity)[0]

    return fa, primary_aniso_ff, primary_axial_diffusivity, primary_radial_diffusivity, restricted_ff, hindered_ff, free_water_ff

