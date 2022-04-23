import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
from sklearn.neighbors import NearestNeighbors


def plot_3d(X, title, ax):
    ax.scatter3D(X[:,0],X[:,1],X[:,2])
    ax.title.set_text(title)
    pass


def kmeans(X, k, threshold, ismean=True):
    print('Starting K-MEANS')
    # K means on euclidean DISTANCE
    if ismean:
        mean_center = np.mean(X, axis = 0)
    else:
        mean_center = np.median(X, axis = 0)

    dist_vec = np.sqrt(np.sum(np.square(X-mean_center), axis = 1))

    # Assign k clusters
    cluster_assignments = KMeans(n_clusters=k, random_state=1).fit_predict(dist_vec[:,None])

    X_clean_list = []
    for i in range(k):

        X_k = X[np.where(cluster_assignments == i)]

        # Keep points that have at least threshold amount of points
        if len(X_k) >= threshold:
            X_clean_list.append(X_k)

    # Rebuild new points after removing
    X_clean = np.vstack(X_clean_list)
    print("Points lost: ", len(X) - len(X_clean))
    print("Points remaining:", len(X_clean))

    return X_clean


def gaussian(X, ismean=True, alpha = 0.05):

    if ismean:
        mean_vec = np.mean(X, axis = 0)
    else:
        mean_vec = np.median(X, axis = 0)

    cov_mat = np.cov(X, rowvar = 0)

    p = multivariate_normal.cdf(X, mean = mean_vec, cov = cov_mat)

    X_clean = X[np.where((p > alpha) & (p < (1-alpha)))]
    print("Points lost:", len(X) - len(X_clean))
    print("Points remaining:", len(X_clean))

    return X_clean



def nearestn(X, n_neighbor, n_int = 1, threshold = 0.001):
    print('Starting NN')
    X_clean = X

    for i in range(n_int):
        print(i)
        n_before = len(X_clean)
        knc = NearestNeighbors(n_neighbors = n_neighbor)
        knc_fit = knc.fit(X_clean)
        neigh_dist, neigh_ind = knc_fit.kneighbors()

        X_clean = X_clean[np.where(neigh_dist < threshold)[0],:]

        n_after = len(X_clean)
        print("Points lost:", n_before - n_after)
    print("Points remaining:", len(X_clean))


    return X_clean



def IQR_test(X):

    mean_center = np.mean(X, axis = 0)
    dist_vec = np.sqrt(np.sum(np.square(X-mean_center), axis = 1))
    dist_vec = np.sort(dist_vec)

    Q1,Q3 = np.percentile(dist_vec , [25,75])
    IQR = Q3 - Q1
    lower = Q1 - (1.5 * IQR)
    upper = Q3 + (1.5 * IQR)

    X_clean = X[np.where((dist_vec < upper)&(dist_vec > lower))]
    print("Points lost:", len(X) - len(X_clean))
    print("Points remaining:", len(X_clean))

    return X_clean



def gmm_cutoff(X, k, threshold):
    GMM = GaussianMixture(n_components=k, random_state=0).fit(X)
    score = GMM.score_samples(X)

    return X[np.where(score > threshold)]



def spatial_cutoff(X, threshold, ifmean = True):

    if ifmean:
        mean_center = np.mean(X, axis = 0)
    else:
        mean_center = np.median(X, axis = 0)

    dist_vec = np.linalg.norm(X - mean_center, axis = 1)

    return X[np.where(dist_vec < threshold)]
