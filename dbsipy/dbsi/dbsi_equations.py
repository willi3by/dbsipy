import numpy as np
import scipy.optimize as optimize
from dbsi.dbsi_utilities import *

def diffusion_basis_eq_1(c_array, lambda_perp, lambda_par, d_iso, s_k, b_k, theta_ik):
    ## How do we get term_1 to be of shape (26,)??
    ## c_i is (95,), (np.exp(-np.abs(b_k)*lambda_perp)) is (26,), and
    ## (np.exp(-np.abs(b_k)*(lambda_par-lambda_perp)*(np.cos(theta_ik)**2))) is (95,26)
    ## Params for this part will be [c_i, c_iso]
    c_i = c_array[:95]
    c_iso = c_array[95]
    term_1_part_2 =(np.exp(-np.abs(b_k)*lambda_perp))
    term_1_part_3 = (np.exp(-np.abs(b_k)*(lambda_par-lambda_perp)*(np.cos(theta_ik)**2)))
    term_1 = np.dot(c_i, term_1_part_3)*term_1_part_2
    term_2 = c_iso*np.exp(-np.abs(b_k)*d_iso)
    s_k_predicted = term_1 + term_2

    rmse = np.sqrt(np.sum((s_k_predicted - s_k) ** 2) / len(s_k))
    return rmse

def diffusion_basis_eq_2(diffusivity_array, c_array, s_k, b_k, theta_ik):
    ## How do we get term_1 to be of shape (26,)??
    ## c_i is (95,), (np.exp(-np.abs(b_k)*lambda_perp)) is (26,), and
    ## (np.exp(-np.abs(b_k)*(lambda_par-lambda_perp)*(np.cos(theta_ik)**2))) is (95,26)
    ## Params for this part will be [c_i, c_iso]
    c_i = c_array[:95]
    c_iso = c_array[95]
    lambda_perp = diffusivity_array[0]
    lambda_par = diffusivity_array[1]
    d_iso = diffusivity_array[2]

    term_1_part_2 =(np.exp(-np.abs(b_k)*lambda_perp))
    term_1_part_3 = (np.exp(-np.abs(b_k)*(lambda_par-lambda_perp)*(np.cos(theta_ik)**2)))
    term_1 = np.dot(c_i, term_1_part_3)*term_1_part_2
    term_2 = c_iso*np.exp(-np.abs(b_k)*d_iso)
    s_k_predicted = term_1 + term_2

    rmse = np.sqrt(np.sum((s_k_predicted - s_k)**2)/len(s_k))
    return rmse

def optimize_c_array(basis_tensors, bvecs, bvals, s_k, init_lambda_perp=0.0003,
                     init_lambda_par=0.0005, init_diso=0.0006):

    c_array = np.full((basis_tensors.shape[0]+1,), 1 / (basis_tensors.shape[0]+1))
    theta_ik = np.zeros((len(basis_tensors), len(bvals)))
    for i in range(len(bvecs)):
        theta_i = np.apply_along_axis(angle_between, 1, basis_tensors, bvecs[i])
        theta_ik[:, i] = theta_i

    bounds = [(0, None) for i in range(len(c_array))]
    b_k = bvals
    opt_c_array = optimize.minimize(diffusion_basis_eq_1, c_array,
                                    args=(init_lambda_perp, init_lambda_par, init_diso, s_k, b_k, theta_ik),
                                    bounds=bounds).x
    return opt_c_array


def optimize_diffusivity_array(basis_tensors, bvecs, bvals, s_k, opt_c_array, init_lambda_perp=0.0003,
                               init_lambda_par=0.0005, init_diso=0.0006):

    theta_ik = np.zeros((len(basis_tensors), len(bvals)))
    for i in range(len(bvecs)):
        theta_i = np.apply_along_axis(angle_between, 1, basis_tensors, bvecs[i])
        theta_ik[:, i] = theta_i

    b_k = bvals
    diffusivity_array = np.array([init_lambda_perp, init_lambda_par, init_diso])
    opt_diffusivity_array = optimize.minimize(diffusion_basis_eq_2, diffusivity_array, args=(opt_c_array, s_k, b_k, theta_ik),
                      bounds=[(0, 0.1), (0, 0.1), (0, 0.1)]).x

    return opt_diffusivity_array

def build_M_matrix(aniso_tensors, bvecs, bvals, lambda_perps, lambda_pars, a=0, b=5e-6, L=20):

    v = (b - a) / (L - 1)
    diff_spectrum = np.arange(a, b + v, v)
    aniso_matrix = np.zeros((len(bvals), len(aniso_tensors)))
    b_k = bvals
    for i in range(len(aniso_tensors)):
        aniso_tensor = aniso_tensors[i]
        theta = np.apply_along_axis(angle_between, 1, bvecs, aniso_tensor)
        lambda_perp_i = lambda_perps[i]
        lambda_par_i = lambda_pars[i]
        m_col = np.exp(-np.abs(b_k) * lambda_perp_i) * np.exp(
            -np.abs(b_k) * (lambda_par_i - lambda_perp_i) * (np.cos(theta) ** 2))
        aniso_matrix[:, i] = m_col

    iso_matrix = np.zeros((len(b_k), len(diff_spectrum)))
    for i, val in enumerate(diff_spectrum):
        l_col = np.exp(-np.abs(b_k)*val)
        iso_matrix[:,i] = l_col

    m_matrix = np.concatenate([aniso_matrix, iso_matrix], axis=1)
    return m_matrix

def fiber_fraction_eq_1(f_i, m_matrix, s_k, mu=0.01):
    s_k_predicted = np.dot(m_matrix, f_i)
    error = np.sum(np.abs(s_k_predicted - s_k)**2) + (mu*np.sum(np.abs(f_i)**2))
    return error

def fiber_fraction_eq_2(diffusivity_array, aniso_tensors, bvecs, bvals, f_i, s_k, a=5e-8, b=5e-6,
                        L=20, mu=0.01):

    lambda_perps = diffusivity_array[:len(aniso_tensors)]
    lambda_pars = diffusivity_array[len(aniso_tensors):]
    m_matrix = build_M_matrix(aniso_tensors, bvecs, bvals, lambda_perps, lambda_pars)
    s_k_predicted = np.dot(m_matrix, f_i)
    error = np.sum(np.abs(s_k_predicted - s_k) ** 2) + (mu * np.sum(np.abs(f_i) ** 2))
    return error

def optimize_fiber_fraction(aniso_tensors, bvecs, bvals, diffusivity_array, s_k):
    lambda_perps = diffusivity_array[:len(aniso_tensors)]
    lambda_pars = diffusivity_array[len(aniso_tensors):]
    m_matrix = build_M_matrix(aniso_tensors, bvecs, bvals, lambda_perps, lambda_pars)
    f_i = np.full((m_matrix.shape[1],), 1 / m_matrix.shape[1])
    bounds = [(0, None) for i in range(len(f_i))]
    opt_f_i = optimize.minimize(fiber_fraction_eq_1, f_i, args=(m_matrix, s_k), bounds=bounds).x
    return opt_f_i

def optimize_fiber_diffusivities(diffusivity_array, aniso_tensors, bvecs, bvals, opt_f_i, s_k):
    bounds = [(0, 0.5) for i in range(len(diffusivity_array))]
    opt_diffusivity_array = optimize.minimize(fiber_fraction_eq_2, diffusivity_array,
                                              args=(aniso_tensors, bvecs, bvals, opt_f_i, s_k),
                                              bounds=bounds).x
    return opt_diffusivity_array



