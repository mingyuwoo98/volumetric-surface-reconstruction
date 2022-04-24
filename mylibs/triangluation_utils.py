'''
Author(s): Min Gyu Woo (mgwoo), Sijie Xu (s362xu)
'''

# General
import numpy as np
import numpy.linalg as la

# Plotting
import matplotlib.pyplot as plt

# Image
import cv2
import sys
import warnings
import pickle as pkl

# ------------------------- Custom Progress Bar -------------------------------

def progressbar(it, prefix="", size=60, file=sys.stdout):
    count = len(it)
    def show(j):
        x = int(size*j/count)
        file.write("%s[%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x), j, count))
        file.flush()
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    file.write("\n")
    file.flush()

# ------------------------ F Matrix Estimation -----------------------------

def get_F_matrix_sourishghosh(K1, R1, T1, K2, R2, T2):

    """
    """

    def get_cross_matrix(Vector):
        """
        Cross_Matrix = [ 0  -a3 a2]
                       [ a3  0 -a1]
                       [-a2 a1  0 ]
        """
        a1, a2, a3 = Vector
        Cross_Matrix = np.zeros((3, 3))
        Cross_Matrix[0, 1], Cross_Matrix[1, 0] = -a3, a3
        Cross_Matrix[0, 2], Cross_Matrix[2, 0] = a2, -a2
        Cross_Matrix[1, 2], Cross_Matrix[2, 1] = -a1, a1
        return Cross_Matrix

    P1 = K1 @ np.hstack([R1, T1])
    P2 = K2 @ np.hstack([R2, T2])

    # Calculate Pseudo-Inverse
    P1_pI = np.linalg.pinv(P1)

    # Camera center of the first camera
    C1 = np.linalg.solve(P1[:, :-1], -P1[:, -1])

    # Camera center in homogenous coordinates
    C1 = np.hstack([C1, 1])

    # Epipole of C1 into second image
    E2 = P2 @ C1

    # Calculate F matrix
    E2_mat = get_cross_matrix(E2)
    F_mat = E2_mat @ P2 @ P1_pI

    return F_mat

# ------------------------ Sampling on First Image -----------------------------


def sample_on_img(img, num_feature_points=None, intensity_threshold=0.4):
    """
    bounding_box_radius must be ODD number
    """

    # SAMPLE RANDOM INDICES
    h, w, _ = np.shape(img)

    intensity_mask = np.where(np.sum(img, axis=2) > intensity_threshold, 1, 0)

    valid_points_idx = np.where(intensity_mask == 1)
    valid_points = np.array([valid_points_idx[1], valid_points_idx[0]]).T

    # If does not need sampling
    if not num_feature_points:
        return valid_points, intensity_mask
    else:
        sampled_idx = np.random.choice(np.arange(valid_points.shape[0]),
                                       size=num_feature_points, replace=False)
        return valid_points[sampled_idx], intensity_mask


# ------------------------ Retrieve Epipolar Points ----------------------------

def get_right_image_point_cv(imL, imR, impts, x2_on_eline, epsilon,
                             bbr=5, method=cv2.TM_CCOEFF_NORMED,
                             shift=None, show_img=False,
                             matched_start=np.array([]),
                             matched_end=np.array([]), delta=0.):

    # If no impts / epiline given due to out of bound issues
    if impts.size == 0 or x2_on_eline.size == 0:
        return np.array([]), -1

    # delta = 0.00000000001 DEFAULT FOR dinoRing and dinoSparseRing
    h, w = bbr * 2 + 1, bbr * 2 + 1
    x, y = impts
    img = imR.copy()
    img_ref = imL.copy()
    if not shift:  # Define horizontal shift
        shift = (-bbr, bbr)

    # Get template from imL
    imL_padded = np.pad(imL, [(bbr, bbr), (bbr, bbr), (0, 0)], mode='constant')
    template = imL_padded[y:y + h, x:x + w, :]

    result = cv2.matchTemplate(img, template, method)
    result_padded = np.pad(result, [(bbr, bbr), (bbr, bbr)], mode='constant')

    # return location with max score along x2 lines
    shift_template = np.repeat(np.array([0, 1])[None, :], x2_on_eline.shape[0],
                               axis=0)
    shift_diff = shift[1] - shift[0] + 1
    result_eline = np.zeros((shift_diff, x2_on_eline.shape[0]))

    # used to compute the projection of the point to be matched along the line from the
    # match of the start to the match of the end in the right image
    if not (matched_start.size == 0 or matched_end.size == 0):
        line = matched_start - matched_end
        l = 1 / np.sum((line) ** 2) * (line)

    for s in range(shift[0], shift[1]):
        x2_eline_shifted = x2_on_eline + shift_template * s
        x2_eline_shifted = np.where(x2_eline_shifted < 0, 0, x2_eline_shifted)
        x2_eline_shifted_y = np.where(x2_eline_shifted[:, 1] >= img.shape[0],
                                      img.shape[0] - 1, x2_eline_shifted[:, 1])
        x2_eline_shifted_x = np.where(x2_eline_shifted[:, 0] >= img.shape[1],
                                      img.shape[1] - 1, x2_eline_shifted[:, 0])
        result_eline_shifted = result_padded[
            (x2_eline_shifted_y, x2_eline_shifted_x)]
        # Add penality on shifting
        result_eline_shifted -= epsilon * result_eline_shifted * abs(
            2 * s) / shift_diff

        # Add penality on dist from line between matched_start and matched_end to enforce order
        if not (matched_start.size == 0 or matched_end.size == 0):

            # reshaping the points on the epipolar line so we can calculate all the projections at once
            epi_line = np.array([x2_eline_shifted_y, x2_eline_shifted_x])

            # create copies of the sampled line so we can take multiple dot products simultaneously
            line_reshaped = np.tile(line, epi_line.shape[1]).reshape(
                (epi_line.shape[1], epi_line.shape[0]))

            a, b = epi_line.shape
            matched_start_reshaped = \
                np.tile(matched_start, epi_line.shape[1]).reshape(epi_line.shape)

            dot_product = np.dot(epi_line.transpose(),
                                 line_reshaped.transpose())
            scale = np.sum(dot_product, axis=1)
            projections = matched_start_reshaped + \
                          np.multiply(scale, np.tile(l, epi_line.shape[1]).reshape((a, b)))

            diff = projections - epi_line
            penalty = np.sum(np.square(diff), axis=0)

            result_eline_shifted -= delta * penalty

        result_eline[s - shift[0], :] = result_eline_shifted

    score = result_eline.max()
    max_idx = np.where(result_eline == score)
    location_eline = x2_on_eline + shift_template * (max_idx[0][0] + shift[0])
    location = location_eline[max_idx[1][0], :]

    if show_img: # Debugging code for showing the matching points
        bottom_right = (location[0] + w, location[1] + h)
        impts_bottom_right = (x + w, y + h)
        cv2.rectangle(img, location, bottom_right, 255, 5)
        cv2.rectangle(img_ref, impts, impts_bottom_right, 255, 5)

        plt.figure(0, figsize=(10, 4))
        ax81 = plt.subplot(121)
        plt.imshow(img_ref)
        ax82 = plt.subplot(122)
        plt.imshow(img)
        plt.show()

    return location, score

# ------------------ Retrieve Matching Epipolar Pairs  -------------------------

def get_right_epipolar_line(imR, x1, F):
    """
    x1 are image points
    """

    def predict_y(l, x):
        a = -np.divide(l[:, 0], l[:, 1])
        b = -np.divide(l[:, 2], l[:, 1])

        return np.multiply(a[:, None], x) + b[:, None]

    x1_homo = np.append(x1, np.ones(len(x1))[:, None], axis=1)

    l2 = x1_homo @ F.T

    u = np.repeat(np.array([0, imR.shape[1]])[None, :], len(x1), axis=0)

    v = predict_y(l2, u)

    return u, v, l2


def get_left_epipolar_line(imL, x2, F):
    """
    x1 are image points
    """

    def predict_y(l, x):
        a = -np.divide(l[:, 0], l[:, 1])
        b = -np.divide(l[:, 2], l[:, 1])

        return np.multiply(a[:, None], x) + b[:, None]

    x2_homo = np.append(x2, np.ones(len(x2))[:, None], axis=1)

    l1 = x2_homo @ F

    u = np.repeat(np.array([0, imL.shape[1]])[None, :], len(x2), axis=0)

    v = predict_y(l1, u)

    return u, v, l1


def point_and_line(point_index, x1, w, h, l2_list,
                   mask, ignore_precentage = 0.01):
    '''
    Takes a index in x1 and returns the point along with its corresponding epipolar line formatted nicely as input
    for the matching function
    '''
    image_point = np.array([x1[point_index, 0], x1[point_index, 1]])

    # l2 corresponding with the image_point
    l2 = l2_list[point_index, :]

    # Generate all points on epipolar line projected from first image plane
    a = -l2[0] / l2[1]
    b = -l2[2] / l2[1]

    # Produce all the points on the epipolar line
    v_top, v_bot = 0, h
    u_top, u_bot = -b / a, (h - b) / a
    u2 = np.linspace(u_top, u_bot, min([w, h]))
    v2 = np.linspace(v_top, v_bot, min([w, h]))

    # Stack the individual coordinates to make (n x 2) vector
    x2_on_eline = np.vstack((u2, v2)).T
    x2_on_eline = np.rint(x2_on_eline).astype(int)
    image_point = np.rint(image_point).astype(int)

    # Extract all the points on the epipolar line that are within the image
    # (and not restricted by the bounding_box_radius)
    inimg_idx = np.where((u2 < w - 1) & (v2 < h - 1) & (u2 > 1) & (v2 > 1))[0]
    x2_inimg = x2_on_eline[inimg_idx,:]
    x2_inbound = mask[(x2_inimg[:,1], x2_inimg[:,0])]
    x2_on_eline = x2_on_eline[np.where(x2_inbound == 1)]

    # If less than 1% of the points is in bound
    if x2_on_eline.size < min([w, h]) * ignore_precentage:
        return np.array([]), [] # Ignore this pair

    return image_point, x2_on_eline


def get_match_points_linspace(imL, imR, x1, F,
                              intensity_threshold = 0.4,
                              penalty_epsilon = 0.1,
                              bounding_box_radius = 15,
                              epiline_shift = (-20, 20),
                              match_threshold=0.8, delta=0.00000000001):
    # Define some common variables:

    h, w, c = np.shape(imL)

    u2_list, v2_list, l2_list = get_right_epipolar_line(imR, x1, F)

    _, imR_mask = sample_on_img(imR, intensity_threshold=intensity_threshold)

    start, start_line = point_and_line(0, x1, w, h, l2_list, imR_mask)
    end, end_line = point_and_line(-1, x1, w, h, l2_list, imR_mask)

    matched_start, score_start = \
        get_right_image_point_cv(imL, imR, start, start_line,
                                 shift=epiline_shift, bbr=bounding_box_radius,
                                 epsilon=penalty_epsilon)

    matched_end, score_end = \
        get_right_image_point_cv(imL, imR, end, end_line,
                                 shift=epiline_shift, bbr=bounding_box_radius,
                                 epsilon=penalty_epsilon)

    x2_list = []
    for i in range(len(x1)):

        image_point, x2_on_eline = \
            point_and_line(i, x1, w, h, l2_list, imR_mask)

        # Return final_R_point on the right image that is
        # closest to image_point on the left image

        final_R_point, score = \
            get_right_image_point_cv(imL, imR, image_point, x2_on_eline,
                                     shift=epiline_shift, bbr=bounding_box_radius,
                                     epsilon=penalty_epsilon, matched_start=matched_start,
                                     matched_end=matched_end, delta=delta)

        # [-1, -1] represents a match less than threshold or out of bound
        if not final_R_point.size == 0 and score > match_threshold:
            x2_list.append(final_R_point)
        else:
            x2_list.append(np.array([-1, -1]))

    x2 = np.array(x2_list)
    u1_list, v1_list, _ = get_left_epipolar_line(imL, x2, F)

    return x2, u1_list, v1_list, u2_list, v2_list

# -------------------------- Plot Triangluation -------------------------------

def plot_all_triangulated(X_list, C_list, I_list, scale=0.05, angle_list = [-140, 60]):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    if C_list == None and I_list == None:
        for i in range(len(X_list)):

            ax.scatter3D(X_list[i][:, 0], X_list[i][:, 1], X_list[i][:, 2], s = 1)

        ax.view_init(angle_list[0], angle_list[1])
        ax.set_xlim([-scale, scale])
        ax.set_ylim([-scale, scale])
        ax.set_zlim([-scale, scale])
        plt.show()
    else:
        for i in range(len(X_list)):

            ax.scatter3D(X_list[i][:, 0], X_list[i][:, 1], X_list[i][:, 2], s = 1)
            ax.scatter3D(C_list[i][:, 0], C_list[i][:, 1], C_list[i][:, 2],
                         s = 1, color="r")  # CAMERA CENTER
            ax.scatter3D(I_list[i][:, 0], I_list[i][:, 1], I_list[i][:, 2],
                         s = 1, color="g")  # IMAGE CENTER

        ax.view_init(angle_list[0], angle_list[1])
        ax.set_xlim([-scale, scale])
        ax.set_ylim([-scale, scale])
        ax.set_zlim([-scale, scale])
        plt.show()

    return None

# -------------------- Retrieve Epipolar Points (Non-CV) -----------------------

def get_bound_box_intensities(im, point, bounding_box_radius):
    u, v_dec = int(point[0]), point[1]
    v_ceil, v_floor = int(np.ceil(v_dec)), int(np.floor(v_dec))

    i = bounding_box_radius // 2

    im_cropped_list = []

    for v in [v_ceil, v_floor]:
        w_min, w_max = u - i, u + i
        h_min, h_max = v - i, v + i

        im_cropped_list.append(im[h_min:(h_max + 1), w_min:(w_max + 1), :])

    return im_cropped_list[0], v_ceil, im_cropped_list[1], v_floor


def loss_function_abs(image_point, image_point_2, imL_cropped,
                      imR_cropped):  # Depreciated

    # Photoconsistency (?) + (Positional Consistency)
    # Photoconsistency hovers around [0.1, 12]
    # Positional Consistency hovers around [30, 600] (divide by 100 maybe?)
    loss = 70 * np.sum(np.abs(imL_cropped - imR_cropped)) + np.sum(
        np.linalg.norm(image_point - image_point_2))

    return loss


def loss_function_default(image_point, image_point_2, imL_cropped, imR_cropped):
    # Photoconsistency (?) + (Positional Consistency)
    photo_con_loss = np.sum(np.abs(imL_cropped - imR_cropped))
    pos_loss = np.sum(np.abs(image_point[1] - image_point_2[1]))

    loss = photo_con_loss + 3 * pos_loss

    return loss



def get_right_image_point(imL, imR, image_point, x2_on_eline,
                          bbr = 5, loss_function=loss_function_default):

    final_image_point_2_list, final_loss_list = [], []

    # For giving image_point and bounding_box_radius return cropped array
    imL_cropped, _, _, _ = get_bound_box_intensities(imL, image_point,
                                                     bbr)

    for i in range(len(x2_on_eline)):
        # Testing if current point on epipolar line in image 2 works
        image_point_2 = x2_on_eline[i, :]

        # Return the (bounding_box_radius x bounding_box_radius x 3)
        imR_cropped_ceil, v_ceil, imR_cropped_floor, v_floor = \
            get_bound_box_intensities(imR, image_point_2, bbr)

        # Custom loss functino
        image_point_2_list = [np.array([image_point_2[0], v_ceil]),
                              np.array([image_point_2[0], v_floor])]

        # bb_int_L[0] = bb_int_L[1] since the image_point are all ints to begin with
        loss_list = [
            loss_function(image_point, image_point_2_list[0], imL_cropped,
                          imR_cropped_ceil), \
            loss_function(image_point, image_point_2_list[1], imL_cropped,
                          imR_cropped_floor)]

        loss_arg = np.argmin(loss_list)

        image_point_2 = image_point_2_list[loss_arg]
        loss = loss_list[loss_arg]

        final_image_point_2_list.append(image_point_2)
        final_loss_list.append(loss)

    final_R_point = final_image_point_2_list[np.argmin(final_loss_list)]

    return final_R_point
