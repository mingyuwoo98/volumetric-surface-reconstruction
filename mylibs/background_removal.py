import matplotlib.image as image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage import feature, img_as_ubyte
from skimage.measure import ransac
from skimage.transform import warp, ProjectiveTransform, PolynomialTransform, rotate
from skimage.color import rgb2gray
from skimage.feature import corner_harris, corner_peaks, plot_matches, BRIEF, match_descriptors
from sklearn.cluster import KMeans
import alphashape
import warnings

import os
import warnings
import math

def background_remover_naive(img, threshold = 0.4):

    h,w,_ = np.shape(img)
    mask = np.where(np.sum(img, axis = 2) > threshold, 1, 0)
    inbound_idx = np.where(mask == 1)
    points = np.array([inbound_idx[1], inbound_idx[0]]).T

    return mask, points


def background_remover_depth(imgL, imgR, TL, TR, threshold=0.0005, min_d=5):

    T_abs = TR - TL
    assert np.all(np.abs(T_abs) < 1), "The absolute translation is too small"

    # Convert image into uint8 with greyscale
    imgL_grey = cv2.cvtColor(imgL, cv2.COLOR_RGB2GRAY)
    imgR_grey = cv2.cvtColor(imgR, cv2.COLOR_RGB2GRAY)

    # Match points using corner harris
    keypointsL = corner_peaks(corner_harris(imgL_grey), threshold_rel=threshold, min_distance=min_d)
    keypointsR = corner_peaks(corner_harris(imgR_grey), threshold_rel=threshold, min_distance=min_d)

    extractor = BRIEF()

    extractor.extract(imgL_grey, keypointsL)
    keypointsL = keypointsL[extractor.mask]
    descriptorsL = extractor.descriptors

    extractor.extract(imgR_grey, keypointsR)
    keypointsR = keypointsR[extractor.mask]
    descriptorsR = extractor.descriptors

    matchesLR = match_descriptors(descriptorsL, descriptorsR, cross_check=True)

    # Compute the imR projection
    data = (np.flip(keypointsR[matchesLR[:,1]], axis=1), np.flip(keypointsL[matchesLR[:,0]], axis=1))
    model_robust, inliers = ransac(data, ProjectiveTransform, min_samples=4, residual_threshold=1, max_trials=5000)

    imR_rpj = warp(imgR_grey, model_robust.inverse)

    # Compute disparity map
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=5)
    disparity = stereo.compute(imgL_grey, img_as_ubyte(imR_rpj))

    return disparity + 16, imR_rpj


def background_remover_k_clustering(img, Kinit = None, epi = 0.3):

    # Convert to integer and gray scale
    img_int = img_as_ubyte(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))

    # Create the lxy list
    x, y = np.ogrid[:img_int.shape[1], :img_int.shape[0]]
    x_mtx, y_mtx = np.meshgrid(x, y)
    img_lxy = np.dstack((img_int, x_mtx * epi, y_mtx * epi))
    img_lxy_reshape = img_lxy.reshape(-1, 3)

    # Create the K init if not provided
    if not Kinit:
        h, w = img_int.shape
        Kinit = np.array([[0, 0, 0], [0, w, h], [255/2, w/2, h/2]])

    kmeans = KMeans(init=Kinit, n_clusters=3).fit(img_lxy_reshape)

    return kmeans.labels_.reshape(img_int.shape[0], img_int.shape[1])
