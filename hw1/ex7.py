from statistics import variance
from timeit import repeat
import numpy as np
import pandas as pd
import random as rd
import scipy.stats as ss
import matplotlib.pyplot as plt

rd.seed(0x5eed)
# Setup the sampling dataset
norm_params = np.array([[5, 1],
                        [-1, 1.3]]) # mean, variance
n_components = norm_params.shape[0]
n_samples = 1000

def generate_data(norm_params, n_components, weights, n_samples):
    mixture_idx = np.random.choice(len(weights), size=n_samples, replace=True, p=weights)
    y = np.fromiter((ss.norm.rvs(*(norm_params[i])) for i in mixture_idx),
                   dtype=np.float64)

    return y

def plot_distrubution(y):
    plt.hist(y, bins=25, density=True)
    plt.show()


def em(y, n_components, n_iter=100, tol=1e-6, weights=None, means=None, variances=None):
    # Initialize the parameters if not provided
    if weights is None:
        weights = np.ones(n_components) / n_components
    if means is None:
        means = np.random.choice(y, size=n_components)
    if variances is None:
        variances = np.ones(n_components)
    n_samples = len(y)

    log_likelihoods = []
    for t in range(n_iter):
        # E-step
        responsibilities = np.zeros((n_samples, n_components))
        for i in range(n_samples):
            for j in range(n_components):
                responsibilities[i, j] = weights[j] * ss.norm.pdf(y[i], means[j], np.sqrt(variances[j]))
            responsibilities[i] /= np.sum(responsibilities[i])

        # print("E-step: responsibilities = {}".format(responsibilities))
        #find teh number of point belonging to each cluster
        for i in range(n_samples):
            if responsibilities[i,0] > responsibilities[i,1]:
                responsibilities[i,0] = 1
                responsibilities[i,1] = 0
        # M-step
        for j in range(n_components):
            if np.sum(responsibilities[:, j]) == 0 or np.sum(responsibilities[:, j]) == n_samples:
                continue
            weights[j] = np.sum(responsibilities[:, j]) / n_samples
            means[j] = np.average(y, weights=responsibilities[:, j])
            variances[j] = np.average((y - means[j]) ** 2, weights=responsibilities[:, j])

        # Compute the log-likelihood
        log_likelihood = np.sum(np.log(np.sum(responsibilities, axis=1)))
        log_likelihoods.append(log_likelihood)

        # Check for convergence
        if t > 0:
            if np.abs(log_likelihood - log_likelihoods[-1]) < tol:
                break
    return weights, means, variances, log_likelihoods

def plot_log_likelihood(log_likelihoods):
    plt.plot(log_likelihoods)
    plt.show()

if __name__ == "__main__":
    weights_alpha = np.array([0.1, 0.9])
    weights_beta = np.array([0.8, 0.2])
    
    B = 10
    # run the first experiment
    n_samples = 1000
    D_x = generate_data(norm_params, n_components, weights_alpha, n_samples)
    alphas, mu1, mu2, sigma1, sigma2 = [], [], [], [], []

    # weights, means, variances, ll = em(D_x, n_components)
    for i in range(B):
        weights, means, variances, ll = em(D_x, n_components)
        alphas.append(weights[0])
        mu1.append(means[0])
        mu2.append(means[1])
        sigma1.append(variances[0])
        sigma2.append(variances[1])
    print(f"Experiment 1:\nTrue alpha: {weights_alpha[0]}\tEstimated alpha: {np.mean(alphas)}\tVariance alpha: {np.var(alphas)}")
    print(f"True mu1: {norm_params[0][0]}\tEstimated mu1: {np.mean(mu1)}\tVariance mu1: {np.var(mu1)}")
    print(f"True mu2: {norm_params[1][0]}\tEstimated mu2: {np.mean(mu2)}\tVariance mu2: {np.var(mu2)}")
    print(f"True sigma1: {norm_params[0][1]}\tEstimated sigma1: {np.mean(sigma1)}\tVariance sigma1: {np.var(sigma1)}")
    print(f"True sigma2: {norm_params[1][1]}\tEstimated sigma2: {np.mean(sigma2)}\tVariance sigma2: {np.var(sigma2)}")

    # run the second experiment
    n_samples = 1000
    D_x = generate_data(norm_params, n_components, weights_alpha, n_samples)
    alphas, mu1, mu2, sigma1, sigma2 = [], [], [], [], []
    for i in range(B):
        weights, means, variances, ll = em(D_x, n_components)
        alphas.append(weights[0])
        mu1.append(means[0])
        mu2.append(means[1])
        sigma1.append(variances[0])
        sigma2.append(variances[1])
    print(f"Experiment 2:\nTrue alpha: {weights_alpha[0]}\tEstimated alpha: {np.mean(alphas)}\tVariance alpha: {np.var(alphas)}")
    print(f"True mu1: {norm_params[0][0]}\tEstimated mu1: {np.mean(mu1)}\tVariance mu1: {np.var(mu1)}")
    print(f"True mu2: {norm_params[1][0]}\tEstimated mu2: {np.mean(mu2)}\tVariance mu2: {np.var(mu2)}")
    print(f"True sigma1: {norm_params[0][1]}\tEstimated sigma1: {np.mean(sigma1)}\tVariance sigma1: {np.var(sigma1)}")
    print(f"True sigma2: {norm_params[1][1]}\tEstimated sigma2: {np.mean(sigma2)}\tVariance sigma2: {np.var(sigma2)}")

    # run the third experiment
    n_samples = 100

    D_x = generate_data(norm_params, n_components, weights_alpha, n_samples)
    D_y = generate_data(norm_params, n_components, weights_beta, n_samples)

    alphas, betas, mu1, mu2, sigma1, sigma2 = [], [], [], [], [], []
    for i in range(B):
        weights, means, variances, ll = em(D_x, n_components)
        alphas.append(weights[0])
        mu1.append(means[0])
        mu2.append(means[1])
        sigma1.append(variances[0])
        sigma2.append(variances[1])

        weights, means, variances, ll = em(D_y, n_components, weights=None, means=means, variances=variances)
        betas.append(weights[0])
        mu1.append(means[0])
        mu2.append(means[1])
        sigma1.append(variances[0])
        sigma2.append(variances[1])
    print(f"Experiment 3:\nTrue alpha: {weights_alpha[0]}\tEstimated alpha: {np.mean(alphas)}\tVariance alpha: {np.var(alphas)}")
    print(f"True beta: {weights_beta[0]}\tEstimated beta: {np.mean(betas)}\tVariance beta: {np.var(betas)}")
    print(f"True mu1: {norm_params[0][0]}\tEstimated mu1: {np.mean(mu1)}\tVariance mu1: {np.var(mu1)}")
    print(f"True mu2: {norm_params[1][0]}\tEstimated mu2: {np.mean(mu2)}\tVariance mu2: {np.var(mu2)}")
    print(f"True sigma1: {norm_params[0][1]}\tEstimated sigma1: {np.mean(sigma1)}\tVariance sigma1: {np.var(sigma1)}")
    print(f"True sigma2: {norm_params[1][1]}\tEstimated sigma2: {np.mean(sigma2)}\tVariance sigma2: {np.var(sigma2)}")

    # run the fourth experiment
    n_samples = 1000
    D_x = generate_data(norm_params, n_components, weights_alpha, n_samples)
    D_y = generate_data(norm_params, n_components, weights_beta, n_samples)

    alphas, betas, mu1, mu2, sigma1, sigma2 = [], [], [], [], [], []
    for i in range(B):
        weights, means, variances, ll = em(D_x, n_components)
        alphas.append(weights[0])
        mu1.append(means[0])
        mu2.append(means[1])
        sigma1.append(variances[0])
        sigma2.append(variances[1])

        weights, means, variances, ll = em(D_y, n_components, weights=None, means=means, variances=variances)
        betas.append(weights[0])
        mu1.append(means[0])
        mu2.append(means[1])
        sigma1.append(variances[0])
        sigma2.append(variances[1])
    print(f"Experiment 4:\nTrue alpha: {weights_alpha[0]}\tEstimated alpha: {np.mean(alphas)}\tVariance alpha: {np.var(alphas)}")
    print(f"True beta: {weights_beta[0]}\tEstimated beta: {np.mean(betas)}\tVariance beta: {np.var(betas)}")
    print(f"True mu1: {norm_params[0][0]}\tEstimated mu1: {np.mean(mu1)}\tVariance mu1: {np.var(mu1)}")
    print(f"True mu2: {norm_params[1][0]}\tEstimated mu2: {np.mean(mu2)}\tVariance mu2: {np.var(mu2)}")
    print(f"True sigma1: {norm_params[0][1]}\tEstimated sigma1: {np.mean(sigma1)}\tVariance sigma1: {np.var(sigma1)}")
    print(f"True sigma2: {norm_params[1][1]}\tEstimated sigma2: {np.mean(sigma2)}\tVariance sigma2: {np.var(sigma2)}")
    
