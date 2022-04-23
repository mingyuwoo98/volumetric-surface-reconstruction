from mylibs.triangluation_utils import *

# ------------------------ Pair Triangulations  -----------------------------

def triangulation(x1, x2, P1, P2):
    A = np.apply_along_axis(get_matrix_A, 1, np.hstack([x1, x2]), P1, P2)

    # least squares for solving linear system A_{0:2} X_{0:2} = - A_3
    A_02 = A[:, :, 0:3]  # the first 3 columns of 4x4 matrix A
    A_3 = A[:, :, 3]  # the last column on 4x4 matrix A

    X = least_squares_est(A_02, -A_3)

    return X


def get_matrix_A(pair_points, P_1, P_2):
    # X = np.append([n_ptsL, n_ptsR])
    A_row_1 = P_1[0, :] - pair_points[0] * P_1[2, :]
    A_row_2 = P_1[1, :] - pair_points[1] * P_1[2, :]
    A_row_3 = P_2[0, :] - pair_points[2] * P_2[2, :]
    A_row_4 = P_2[1, :] - pair_points[3] * P_2[2, :]

    A = np.vstack([A_row_1, A_row_2, A_row_3, A_row_4])

    return A


def least_squares_est(A, b):
    y = np.linalg.inv(A.transpose((0, 2, 1)) @ A) @ (
                A.transpose((0, 2, 1)) @ b[:, :, None])

    return y


def camera_image_center_pairs(P1, P2):
    # Calculate Camera Centers
    R1, T1 = P1[:, 0:3], P1[:, 3]
    R2, T2 = P2[:, 0:3], P2[:, 3]

    # Camera center of the first camera
    C1 = np.linalg.inv(R1) @ (-T1)
    C2 = np.linalg.inv(R2) @ (-T2)

    # Image Plane coordinate
    I1 = np.linalg.inv(R1) @ (
                -(np.array([0, 0, -1]) + T1.reshape((3,))) * np.array(
            [1, 1, -1]))
    I2 = np.linalg.inv(R2) @ (
                -(np.array([0, 0, -1]) + T2.reshape((3,))) * np.array(
            [1, 1, -1]))

    C = np.vstack([C1, C2])
    I = np.vstack([I1, I2])

    return C, I

# ------------------------ Sequential Triangulations  --------------------------

def sequential_triangulation(camera_matrix, image_list, camera_order=None,
                             bounding_box_radius = 15, num_feature_points=None,
                             intensity_threshold = 0.4,
                             penalty_epsilon=0.1, match_threshold=0.9, pkl_name = 'X_C_I_list_final.pkl'):
    X_list = []
    C_list = []
    I_list = []

    if not camera_order:
        camera_order = np.arange(len(camera_matrix)).tolist()

    for i in progressbar(range(len(camera_matrix) - 1), "Comparing Images: "):
        #print(f"Comparing Images {camera_order[i]} and {camera_order[i + 1]}")
        # Calculate the Fundimental Matrix

        C1_params = camera_matrix[camera_order[i]]
        C2_params = camera_matrix[camera_order[i + 1]]

        K1, R1, T1 = C1_params
        K2, R2, T2 = C2_params

        F = get_F_matrix_sourishghosh(K1, R1, T1, K2, R2, T2)

        # Get images
        imL = image_list[camera_order[i]]
        imR = image_list[camera_order[i + 1]]

        # ------------------------- Get Match Points from Edge Points ------------------------- #

        # -- uncomment below for random points on dino
        x1, _ = sample_on_img(imL, num_feature_points, intensity_threshold)

        x2, u1_list, v1_list, u2_list, v2_list = \
            get_match_points_linspace(imL, imR, x1, F,
                                      intensity_threshold=intensity_threshold,
                                      penalty_epsilon=penalty_epsilon,
                                      bounding_box_radius=bounding_box_radius,
                                      match_threshold=match_threshold)

        # ------------------------- Filter Edge Points ------------------------- #

        x2_filtered = x2[np.where(x2.sum(axis=1) != -2)]
        x1_filtered = x1[np.where(x2.sum(axis=1) != -2)]

        P1 = K1 @ np.hstack([R1, T1])
        P2 = K2 @ np.hstack([R2, T2])

        if (x1_filtered.size == 0) or (x2_filtered.size == 0):
            warnings.warn(f"WARNING: no matches are good enough on img {i} with {i+1}")
        else:
            X = triangulation(x1_filtered, x2_filtered, P1, P2)

            C, I = camera_image_center_pairs(P1, P2)

            X_list.append(X)
            C_list.append(C)
            I_list.append(I)

    print('Done')
    # Save parameters
    with open(pkl_name, 'wb') as handle:
        pkl.dump([X_list, C_list, I_list], handle,
                 protocol=pkl.HIGHEST_PROTOCOL)

    return X_list, C_list, I_list
