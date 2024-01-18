import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize, differential_evolution, LinearConstraint
import os
import nibabel as nib
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.mixture import GaussianMixture

def calculate_vector_angle(v1, v2):
    return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

def get_theta_matrix(basis_tensors, bvecs):
    theta_matrix = np.zeros((basis_tensors.shape[0], bvecs.shape[0]))
    for i in range(basis_tensors.shape[0]):
        for j in range(bvecs.shape[0]):
            theta_matrix[i,j] = calculate_vector_angle(basis_tensors[i,:], bvecs[j,:])
    return theta_matrix

# Define the objective function witg l2 regularization.
def objective_function_c(params, s, b_values, theta_matrix, alpha=0.01):
    lambda_parallel, lambda_perp, d_iso, c_Nplus1 = params[:4]
    c_values = params[4:]
    # Compute the original data fitting term

    s_k = np.dot(c_values, (np.exp(-np.abs(b_values) * lambda_perp) * np.exp(-np.abs(b_values) * (lambda_parallel - lambda_perp) * np.cos(theta_matrix)**2))) + (c_Nplus1 * np.exp(-np.abs(b_values) * d_iso))

    diff = np.diff(c_values)
    tv_penalty = np.sum(np.abs(diff))

    # regularization_term = l2_penalty * np.sum(c_values**2)
    # Compute the sum of squared differences with regularization

    return np.sum((s_k - s)**2) + alpha * tv_penalty

def fit_voxel_c(dbsi_vox, bvalues_sorted, theta_matrix, alpha=0.01,bounds=None, maxiter=1000):
    # Perform the optimization using differential evolution
    result_np = differential_evolution(objective_function_c, bounds=bounds, 
                                       args=(dbsi_vox, bvalues_sorted, theta_matrix, alpha),  
                                       tol=1e-3, disp=True, maxiter=maxiter)

    # Extract the optimized parameters
    optimized_params_np = result_np.x
    lambda_parallel, lambda_perp, d_iso, c_Nplus1 = optimized_params_np[:4]
    c_values = optimized_params_np[4:]
    error_term = result_np.fun
    return lambda_parallel, lambda_perp, d_iso, c_Nplus1, c_values, error_term

def plot_c_fit(bvals, s, theta_matrix, lambda_parallel, lambda_perp, d_iso, c_Nplus1, c_values):
    s_k = np.dot(c_values, (np.exp(-np.abs(bvals) * lambda_perp) * np.exp(-np.abs(bvals) * (lambda_parallel - lambda_perp) * np.cos(theta_matrix)**2))) + (c_Nplus1 * np.exp(-np.abs(bvals) * d_iso))
    plt.figure(figsize=(10, 6))
    plt.plot(bvals, s, 'o', label='data')
    plt.plot(bvals, s_k, label='fit')
    plt.legend()
    plt.xlabel('b-value')
    plt.ylabel('signal')
    plt.show()

def get_c_clusters(c_values):
    all_aic = []
    for i in range(1,6):
        gmm = GaussianMixture(n_components=i)
        gmm.fit(c_values.reshape(-1,1))
        all_aic.append(gmm.aic(c_values.reshape(-1,1)))

    kn = KneeLocator(range(1,6), all_aic, curve='convex', direction='decreasing')
    n_components = kn.knee
    gmm = GaussianMixture(n_components=n_components)
    gmm.fit(c_values.reshape(-1,1))
    c_clusters = gmm.predict(c_values.reshape(-1,1))
    return gmm, c_clusters, n_components

def objective_function_f(params, s, bvalues_sorted, theta_matrix, fd_range, l2_penalty=0.01):
    lambda_parallel = params[:theta_matrix.shape[0]]
    lambda_perp = params[theta_matrix.shape[0]:2*theta_matrix.shape[0]]
    f_values = params[2*theta_matrix.shape[0]:]
    mat = np.zeros((bvalues_sorted.shape[0], theta_matrix.shape[0] + fd_range.shape[0]))
    for i in range(theta_matrix.shape[0]):
        mat[:,i] = np.exp(-bvalues_sorted * lambda_perp[i])*np.exp(-bvalues_sorted * (lambda_parallel[i] - lambda_perp[i]) * (np.cos(theta_matrix[i,:])**2))
    for j in range(fd_range.shape[0]):
        mat[:,j+theta_matrix.shape[0]] = np.exp(-bvalues_sorted * fd_range[j])

    s_k = mat.dot(f_values)
    return np.sum((s_k - s)**2) + l2_penalty * np.sum(f_values**2)

def fit_voxel_f(dbsi_vox, bvalues_sorted, theta_matrix, fd_range, l2_penalty=0.01, bounds=None, maxiter=1000):
    # Perform the optimization using differential evolution
    result_np = differential_evolution(objective_function_f, bounds=bounds, 
                                       args=(dbsi_vox, bvalues_sorted, theta_matrix, fd_range),  
                                       disp=True, maxiter=500)

    # Extract the optimized parameters
    opt_lambda_parallel = result_np.x[:theta_matrix.shape[0]]
    opt_lambda_perp = result_np.x[theta_matrix.shape[0]:2*theta_matrix.shape[0]]
    opt_f_values = result_np.x[2*theta_matrix.shape[0]:]
    error_term = result_np.fun
    return opt_lambda_parallel, opt_lambda_perp, opt_f_values, error_term

def plot_f_fit(bvals, s, theta_matrix, fd_range, opt_lambda_parallel, opt_lambda_perp, opt_f_values):
    mat = np.zeros((bvals.shape[0], theta_matrix.shape[0]))
    mat = np.zeros((bvals.shape[0], theta_matrix.shape[0] + fd_range.shape[0]))
    for i in range(theta_matrix.shape[0]):
        mat[:,i] = np.exp(-bvals * opt_lambda_perp[i])*np.exp(-bvals * (opt_lambda_parallel[i] - opt_lambda_perp[i]) * (np.cos(theta_matrix[i,:])**2))
    for j in range(fd_range.shape[0]):
        mat[:,j+theta_matrix.shape[0]] = np.exp(-bvals * fd_range[j])
    s_k = mat.dot(opt_f_values)
    plt.figure(figsize=(10, 6))
    plt.plot(bvals, s, 'o', label='data')
    plt.plot(bvals, s_k, label='fit')
    plt.legend()
    plt.xlabel('b-value')
    plt.ylabel('signal')
    plt.show()

