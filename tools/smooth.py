from matplotlib.pyplot import axis
import numpy as np
from scipy.signal import savgol_filter


def smoothing_2d(keypoints, smooth_window_size, polynomial_size, smooth_2d_club, club_window_size, website):
    frames_num = keypoints.shape[0]
    kpts_num = keypoints.shape[1]
    result = np.zeros_like(keypoints)
    if website:
        group_id = (34, 35, 36, 37, 20, 21, 22, 23, 16, 19)
        if smooth_2d_club:
            for kpt_id in group_id:
                result[:, kpt_id, 0] = savgol_filter(keypoints[:, kpt_id, 0], 2 * club_window_size + 1, polynomial_size, mode='nearest')
                result[:, kpt_id, 1] = savgol_filter(keypoints[:, kpt_id, 1], 2 * club_window_size + 1, polynomial_size, mode='nearest')
        for kpt_id in range(kpts_num):
            if kpt_id not in group_id:
                result[:, kpt_id, 0] = savgol_filter(keypoints[:, kpt_id, 0], 2 * smooth_window_size + 1, polynomial_size, mode='nearest')
                result[:, kpt_id, 1] = savgol_filter(keypoints[:, kpt_id, 1], 2 * smooth_window_size + 1, polynomial_size, mode='nearest')
    else:
        for kpt_id in range(kpts_num):
            result[:, kpt_id, 0] = savgol_filter(keypoints[:, kpt_id, 0], 2 * smooth_window_size + 1, polynomial_size, mode='nearest')
            result[:, kpt_id, 1] = savgol_filter(keypoints[:, kpt_id, 1], 2 * smooth_window_size + 1, polynomial_size, mode='nearest')
        
    dist = np.linalg.norm(result[:, :, :2] - keypoints[:, :, :2], axis=2)
    update_conf = np.exp(-1 / 2 * (dist / 7) ** 2) * keypoints[:, :, 2]
    # result[:, :, 2] = update_conf
    result[:, :, 2] = keypoints[:, :, 2]
    
    # for idx in range(smooth_window_size, frames_num - smooth_window_size):
    #     collection = keypoints[idx - smooth_window_size: idx + smooth_window_size + 1, :, :]
    #     for kpt_id in range(kpts_num):
    #         # # no 2d smoothing 4 points
    #         # if kpt_id >= 34 and kpt_id <= 37:
    #         #     result[idx, kpt_id, :] = collection[smooth_window_size, kpt_id, :]
    #         #     continue

    #         # # no smoothing 38 points
    #         # result[idx, kpt_id, :] = collection[smooth_window_size, kpt_id, :]
    #         # continue

    #         indices = np.arange(collection.shape[0])
    #         data_x = collection[:, kpt_id, 0].reshape(-1)
    #         data_y = collection[:, kpt_id, 1].reshape(-1)
    #         x_coeff = np.polyfit(indices, data_x, polynomial_size)
    #         y_coeff = np.polyfit(indices, data_y, polynomial_size)
    #         x_poly_func = np.poly1d(x_coeff)
    #         y_poly_func = np.poly1d(y_coeff)

    #         # Modify confidence score
    #         update_x = x_poly_func(smooth_window_size)
    #         update_y = y_poly_func(smooth_window_size)
    #         dist = np.sqrt((update_x - keypoints[idx, kpt_id, 0]) ** 2 + (update_y - keypoints[idx, kpt_id, 1]) ** 2)
    #         update_conf = np.exp(-1 / 2 * (dist / 7) ** 2) * keypoints[idx, kpt_id, 2]

    #         result[idx, kpt_id, :] = np.array(
    #             [update_x, update_y, update_conf])

    # for idx in range(smooth_window_size):
    #     result[idx, :, :] = result[smooth_window_size, :, :]

    # for idx in range(frames_num - smooth_window_size, frames_num):
    #     result[idx, :, :] = result[frames_num - smooth_window_size - 1, :, :]
    
    # print(keypoints[20:30, 0, :])
    # print(result[20:30, 0, :])

    return result


def smoothing_3d(keypoints, smooth_window_size, polynomial_size, smooth_3d_club, club_window_size, website):
    '''
    keypoints: [frame_num, 38, 3]
    '''

    frames_num = keypoints.shape[0]
    kpts_num = keypoints.shape[1]
    result = np.zeros_like(keypoints)
    indices = np.arange(2 * smooth_window_size + 1)
    if website:
        group_id = (34, 35, 36, 37, 20, 21, 22, 23, 16, 19)
        if smooth_3d_club:
            for kpt_id in group_id:
                result[:, kpt_id, 0] = savgol_filter(keypoints[:, kpt_id, 0], 2 * club_window_size + 1, polynomial_size, mode='nearest')
                result[:, kpt_id, 1] = savgol_filter(keypoints[:, kpt_id, 1], 2 * club_window_size + 1, polynomial_size, mode='nearest')
                result[:, kpt_id, 2] = savgol_filter(keypoints[:, kpt_id, 2], 2 * club_window_size + 1, polynomial_size, mode='nearest')
        for kpt_id in range(kpts_num):
            if kpt_id not in group_id:
                result[:, kpt_id, 0] = savgol_filter(keypoints[:, kpt_id, 0], 2 * smooth_window_size + 1, polynomial_size, mode='nearest')
                result[:, kpt_id, 1] = savgol_filter(keypoints[:, kpt_id, 1], 2 * smooth_window_size + 1, polynomial_size, mode='nearest')
                result[:, kpt_id, 2] = savgol_filter(keypoints[:, kpt_id, 2], 2 * smooth_window_size + 1, polynomial_size, mode='nearest')
    else:
        for kpt_id in range(kpts_num):
            result[:, kpt_id, 0] = savgol_filter(keypoints[:, kpt_id, 0], 2 * smooth_window_size + 1, polynomial_size, mode='nearest')
            result[:, kpt_id, 1] = savgol_filter(keypoints[:, kpt_id, 1], 2 * smooth_window_size + 1, polynomial_size, mode='nearest')
            result[:, kpt_id, 2] = savgol_filter(keypoints[:, kpt_id, 2], 2 * smooth_window_size + 1, polynomial_size, mode='nearest')
    
    # for idx in range(smooth_window_size, frames_num - smooth_window_size):
    #     collection = keypoints[idx - smooth_window_size: idx + smooth_window_size + 1, :, :]
    #     for kpt_id in range(kpts_num):
    #         # # no 3d smoothing 4 points
    #         # if kpt_id >= 34 and kpt_id <= 37:
    #         #     result[idx, kpt_id, :] = collection[smooth_window_size, kpt_id, :]
    #         #     continue

    #         # # no smoothing 38 points
    #         # result[idx, kpt_id, :] = collection[smooth_window_size, kpt_id, :]
    #         # continue

    #         data_x = collection[:, kpt_id, 0].reshape(-1)
    #         data_y = collection[:, kpt_id, 1].reshape(-1)
    #         data_z = collection[:, kpt_id, 2].reshape(-1)
    #         x_coeff = np.polyfit(indices, data_x, polynomial_size)
    #         y_coeff = np.polyfit(indices, data_y, polynomial_size)
    #         z_coeff = np.polyfit(indices, data_z, polynomial_size)
    #         x_poly_func = np.poly1d(x_coeff)
    #         y_poly_func = np.poly1d(y_coeff)
    #         z_poly_func = np.poly1d(z_coeff)
    #         result[idx, kpt_id, :] = np.array(
    #             [x_poly_func(smooth_window_size), y_poly_func(smooth_window_size), z_poly_func(smooth_window_size)])

    # for idx in range(smooth_window_size):
    #     result[idx, :, :] = result[smooth_window_size, :, :]

    # for idx in range(frames_num - smooth_window_size, frames_num):
    #     result[idx, :, :] = result[frames_num - smooth_window_size - 1, :, :]
    
    # print(keypoints[20:30, 0, :])
    # print(result[20:30, 0, :])

    return result
