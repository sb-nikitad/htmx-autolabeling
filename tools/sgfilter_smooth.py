import os, sys
import numpy as np

def filterCoeff(n, k):
    '''
    n: whole window size
    k: polynomial size
    '''
    step = (n - 1) / 2
    X = np.zeros((n, k))

    for i in range(n):
        for j in range(k):
            X[i, j] = np.power(i - step, j)

    tmp1 = np.linalg.inv(np.dot(X.T, X))
    tmp2 = np.dot(X, tmp1)

    return np.dot(tmp2, X.T)

def sgfilter(data, smooth_window_size, polynomial_size):
    whole_window_size = 2 * smooth_window_size + 1

    pad_left = np.array([data[0] for _ in range(smooth_window_size)])
    pad_right = np.array([data[-1] for _ in range(smooth_window_size)])
    pad_data = np.concatenate((pad_left, data, pad_right), axis=0)

    coeff_mat = filterCoeff(whole_window_size, polynomial_size + 1)

    filter_res_mat = np.dot(coeff_mat, pad_data)

    return filter_res_mat[smooth_window_size, :]


def smoothing_2d(keypoints, smooth_window_size, polynomial_size):
    '''
    keypoints: [frame_num, 38, 3]
    '''
    kpts_num = keypoints.shape[1]
    result = np.zeros_like(keypoints)

    for kpt_id in range(kpts_num):
        result[:, kpt_id, 0] = sgfilter(keypoints[:, kpt_id, 0], smooth_window_size, polynomial_size)
        result[:, kpt_id, 1] = sgfilter(keypoints[:, kpt_id, 1], smooth_window_size, polynomial_size)
    result[:, :, 2] = keypoints[:, :, 2]

    return result


def smoothing_3d(keypoints, smooth_window_size, polynomial_size):
    '''
    keypoints: [frame_num, 38, 3]
    '''

    kpts_num = keypoints.shape[1]
    result = np.zeros_like(keypoints)

    for kpt_id in range(kpts_num):
        result[:, kpt_id, 0] = sgfilter(keypoints[:, kpt_id, 0], smooth_window_size, polynomial_size)
        result[:, kpt_id, 1] = sgfilter(keypoints[:, kpt_id, 1], smooth_window_size, polynomial_size)
        result[:, kpt_id, 2] = sgfilter(keypoints[:, kpt_id, 2], smooth_window_size, polynomial_size)

    return result