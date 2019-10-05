from sklearn.datasets import make_circles
import pandas as pd
import seaborn as sns
from math import sqrt,exp
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd


# The following program applied k-means algorithm for unsupervised learning 
# with sklearn circles dataset. To create a decision boundary in the case of :
# non-linear dataset, Gaussian kernels are utilized. In order to decrease
# computational heaviness, dimensions reduction using principal component analysis 
# was applied. All functions were written from scratch without using any built-in
# sklearn functions.
 

# A function calculating the distance between two points in any coordinate system
def distance_points(x, y):
    return sqrt(((x - y) ** 2).sum())


# A function calculating similarity between two vectors
def calc_kernel(x, l, sigma):
    return exp(-(distance_points(x, l) / (2 * sigma ** 2)))


# A function creating a new features input based on the kernel function
def new_kernel_features(X, sigma):
    m = X.shape[0]
    new_features = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            new_features[i, j] = calc_kernel(X[i, :], X[j, :], sigma)
    return new_features


# A dimensions reduction function using principal component analysis
# var_limit is a required value of retained variance (0-1)
def pca(X, var_limit):
    m = X.shape[0]
    dim = X.shape[1]
    covariance = np.matmul(np.transpose(X), X) / m
    (U, S, V) = svd(covariance)
    variance_sum = np.sum(np.diag(S))
    red_dim = dim
    for k in range(dim):
        var_retained = np.sum(np.diag(S[:k]))
        if var_retained / variance_sum > var_limit:
            red_dim = k + 1
            break
    Z = np.zeros((m, red_dim))
    U_reduced = U[:, :red_dim]
    for i in range(m):
        x = X[i, :][None, :]
        for j in range(red_dim):
            projection = float(np.matmul(x, U_reduced[:, j][:, None]))
            Z[i, j] = projection
    print(f"PCA reduced dimensions from {dim} to {red_dim}.")
    return Z


# A function randomly initializing centroids    
def init_centroids(K, X):
    centroids = np.zeros((K, X.shape[1]))
    randidx = np.random.randint(0, X.shape[0], size = K)
    centroids = X[randidx[0:K]]
    return centroids


# A function assigning closest centroid to each input from dataset
def assign_centroid(X, c):
    K = c.shape[0]
    idx = np.zeros(X.shape[0], dtype=np.int8)
    for i in range(X.shape[0]):
        for j in range(K):
            if j == 0:
                idx[i] = 0
            sum_new = 0
            for k in range(X.shape[1]):
                sum_new = sum_new + (X[i, k] - c[j, k]) * (X[i, k] - c[j, k])
            sum_new = sqrt(sum_new)
            sum_old = 0
            for k in range(X.shape[1]):
                sum_old = sum_old + (X[i, k] - c[int(idx[i]), k]) \
                    * (X[i, k] - c[int(idx[i]), k])
            sum_old = sqrt(sum_old)
            if sum_old > sum_new:
                idx[i] = j
    return idx


#A function moving centroids to new locations
def move_centroids(X, idx, K):
    (m, n) = X.shape
    centroids = np.zeros((K, n))
    for i in range(K):
        array = X[idx == i, :]
        x_incluster = array.shape[0]
        array = sum(array)
        if x_incluster != 0:
            array = array / x_incluster
        centroids[i, :] = array
    return centroids


# A function calculating the loss of K-means algorithm
def kmeans_loss(X, group, centroids):
    m = X.shape[0]
    loss = 0
    for i in range(m):
        loss += distance_points(X[i, :], centroids[group[i], :])
    loss *= 1 / m
    return loss


# A function executing one run of K-means model training
def run_kmeans(X, K, steps=100):
    centroids = init_centroids(K, X)
    group = assign_centroid(X, centroids)
    centroids = move_centroids(X, group, K)
    for i in range(steps):
        previous_loss = kmeans_loss(X, group, centroids)
        group = assign_centroid(X, centroids)
        centroids = move_centroids(X, group, K)
        current_loss = kmeans_loss(X, group, centroids)
        if current_loss == previous_loss:
            break
    return (group, centroids, current_loss)


# A function loading dataset
def init():
    (X, y) = make_circles(n_samples=500, noise=0.1, factor=0.3)
    data = pd.DataFrame({'X1': X[:, 0], 'X2': X[:, 1], 'Cluster': y})
    X_data = np.array(data[['X1', 'X2']])
    return X_data


# A function repeating training with many initializations and finding the best model
def train(X_data, trials=10,var_limit = 0.999,tested_sigma=[1]):  
    print("Training...")
    sigma_axis = []
    for sigma in tested_sigma:
        print(f"Sigma: {sigma}")
        loss_axis = []
        centroids_axis = []
        assigned_centroids_axis = []
        new_feat = new_kernel_features(X_data, sigma)
        reduced_feat = pca(new_feat, var_limit)
        for i in range(trials):        
            sigma_axis.append(sigma)
            assigned_centroids,centroids, loss = run_kmeans(reduced_feat,K=2)
            loss_axis.append(loss)
            centroids_axis.append(centroids)
            assigned_centroids_axis.append(assigned_centroids)
        best_run = loss_axis.index(min(loss_axis))
        loss = loss_axis[best_run]
        centroids = centroids_axis[best_run]
        assigned_centroids = assigned_centroids_axis[best_run]
        sigma = sigma_axis[best_run]
        print(f"Best resutls for sigma={sigma}")
        visualize(X_data,assigned_centroids)
    return (centroids, assigned_centroids)
   
    
# A function
def visualize (X_data,assigned_centroids):  
    (fig, ax) = plt.subplots(1, 1, figsize=(6, 6))
    results_kernel = pd.DataFrame({'X1': X_data[:, 0], 'X2': X_data[:,
                                  1], 'Cluster': assigned_centroids})
    sns.scatterplot(ax=ax, x='X1', y='X2', hue='Cluster',
                    data=results_kernel)
    plt.show() 