import os
import torch
import tensorflow as tf
import numpy as np
import math
import cv2
import copy
import json
import skvideo.io
import matplotlib.pyplot as plt
from tools import inference_2d_tf
import libs.videopose3d.common.modeltf_attn as model3d


def inference_2d_golf_club_line_segment(img_idx, ignore_frames, img, results_2d, conf_2d, ref_club, ref_club_length,
                                        ref_direction, ref_img_idx, fps=120):
    def line_association(line_1, line_2):
        '''
        line: [x1, y1, x2, y2]
        '''
        line_1 = np.array(line_1[0])
        line_2 = np.array(line_2[0])

        suppose_direction = line_1[2:] - line_1[:2]
        suppose_direction = suppose_direction / np.linalg.norm(suppose_direction)

        direction1 = line_2[2:] - line_2[:2]
        direction1 = direction1 / np.linalg.norm(direction1)

        direction2 = line_2[:2] - line_1[2:]
        direction2 = direction2 / np.linalg.norm(direction2)

        angle1 = direction1.dot(suppose_direction)
        angle2 = direction2.dot(suppose_direction)
        if 1 - angle1 < 0.0025:
            dist = np.linalg.norm(line_2[:2] - line_1[2:])
            if dist > 10 and 1 - angle2 > 0.0025:  # 4 degree
                return np.inf
            else:
                return dist
        else:
            return np.inf

    def get_foot_point(point, line_p1, line_p2):
        """
        point, line_p1, line_p2 : [x, y]
        """
        x0 = point[0]
        y0 = point[1]

        x1 = line_p1[0]
        y1 = line_p1[1]

        x2 = line_p2[0]
        y2 = line_p2[1]

        k = -((x1 - x0) * (x2 - x1) + (y1 - y0) * (y2 - y1)) / \
            ((x2 - x1) ** 2 + (y2 - y1) ** 2) * 1.0

        xn = k * (x2 - x1) + x1
        yn = k * (y2 - y1) + y1

        return np.array([xn, yn])

    def calc_angle(direction1, direction2):
        angle = direction1.dot(direction2)
        angle = angle / (np.linalg.norm(direction1) * np.linalg.norm(direction2))

        return angle

    def check_hosel(results_2d, conf_2d):
        start = results_2d[0, 35, :]
        end = results_2d[0, 34, :]
        hosel = results_2d[0, 36, :]

        direction1 = end - start
        direction2 = hosel - start

        angle = calc_angle(direction1, direction2)

        if angle < 0.94:
            conf_2d[0, 36, 0] = 0

        return conf_2d

    if fps == 240:
        ref_dir_interval_thres = 6
    else:
        ref_dir_interval_thres = 3

    current_direction = results_2d[0, 34, :] - results_2d[0, 35, :]     # 34: mid hands     35: top of handle
    current_direction = current_direction / np.linalg.norm(current_direction)

    if ref_direction is None:
        suppose_direction = current_direction
    elif img_idx - ref_img_idx > ref_dir_interval_thres:
        angle = ref_direction.dot(current_direction) / \
                (np.linalg.norm(ref_direction) * np.linalg.norm(current_direction))
        if 1 - abs(angle) < 0.065:
            suppose_direction = ref_direction
        else:
            ref_direction = None
            ref_club = None
            ref_club_length = None
            suppose_direction = current_direction
    else:
        suppose_direction = ref_direction

    suppose_direction = suppose_direction / np.linalg.norm(suppose_direction)

    min_edge = min(img.shape[0], img.shape[1])
    scale_factor = min_edge / 720 * 1.0

    # bbox: [up_left_corner, bottom_right_corner]
    if img_idx - ignore_frames == 0 or ref_direction is None:
        bbox = np.array([np.clip(results_2d[0, 35, 0] - current_direction[0] * 50 * scale_factor, 0, img.shape[1]),
                         np.clip(results_2d[0, 35, 1] - current_direction[1] * 50 * scale_factor, 0, img.shape[0]),
                         np.clip(results_2d[0, 35, 0] + current_direction[0] * 250 * scale_factor, 0, img.shape[1]),
                         np.clip(results_2d[0, 35, 1] + current_direction[1] * 250 * scale_factor, 0, img.shape[0])])
        bbox = np.array([max(min(bbox[0], bbox[2]) - 50, 0),
                         max(min(bbox[1], bbox[3]) - 50, 0),
                         min(max(bbox[0], bbox[2]) + 50, img.shape[1]),
                         min(max(bbox[1], bbox[3]) + 50, img.shape[0])])

    else:
        bbox = np.array([ref_club[0],
                         ref_club[1],
                         ref_club[2],
                         ref_club[3]])
        bbox = np.array([max(min(bbox[0], bbox[2]) - 75 * scale_factor, 0),
                         max(min(bbox[1], bbox[3]) - 75 * scale_factor, 0),
                         min(max(bbox[0], bbox[2]) + 75 * scale_factor, img.shape[1]),
                         min(max(bbox[1], bbox[3]) + 75 * scale_factor, img.shape[0])])

    # line segment detection requires GRAY format
    try:
        crop_img = img[int(bbox[1]): int(bbox[3]), int(bbox[0]): int(bbox[2]), :]
    except:
        conf_2d = check_hosel(results_2d, conf_2d)
        return results_2d, conf_2d, ref_club, ref_club_length, ref_direction, ref_img_idx

    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

    fld = cv2.ximgproc.createFastLineDetector()
    lines = fld.detect(crop_img)

    if lines is None:
        conf_2d = check_hosel(results_2d, conf_2d)
        return results_2d, conf_2d, ref_club, ref_club_length, ref_direction, ref_img_idx

    # Filter lines by directions
    interested_lines = []
    for line in lines:
        np_line = np.array(line[0])

        direction = np_line[2:] - np_line[:2]
        direction = direction / np.linalg.norm(direction)
        angle = direction.dot(suppose_direction)

        if abs(angle) > 0.95:
            if angle < 0:
                np_line = np.array([np_line[2], np_line[3], np_line[0], np_line[1]])

            np_line = np.array([[np_line[0] + bbox[0], np_line[1] + bbox[1],
                                 np_line[2] + bbox[0], np_line[3] + bbox[1]]])

            direction1 = np_line[0][:2] - results_2d[0, 34, :]
            direction1 = direction1 / np.linalg.norm(direction1)
            angle1 = direction1.dot(current_direction)

            if angle1 > 0.93:
                interested_lines.append(np_line)

    # Find lines whose stating point is closest to mid hand
    all_dist_interested_lines = []
    for line in interested_lines:
        np_line = np.array(line[0])
        all_dist_interested_lines.append(np.linalg.norm(np_line[:2] - results_2d[0, 34, :]))
    sorted_idx = np.argsort(all_dist_interested_lines).tolist()
    if len(sorted_idx) > 20:
        sorted_idx = sorted_idx[:20]

    lines = []
    initial_lines_idx = copy.deepcopy(sorted_idx)

    for idx1 in initial_lines_idx:
        if all_dist_interested_lines[idx1] > 100:
            break
        traj_lines_idx = [idx1]
        not_traj_lines_idx = [i for i in range(len(interested_lines))]
        not_traj_lines_idx.remove(idx1)

        while len(not_traj_lines_idx) != 0:
            start_idx = traj_lines_idx[0]
            end_idx = traj_lines_idx[-1]
            line_1 = np.array([[interested_lines[start_idx][0][0], interested_lines[start_idx][0][1],
                                interested_lines[end_idx][0][2], interested_lines[end_idx][0][3]]])

            all_dist = np.full(len(not_traj_lines_idx), np.inf)
            for i in range(len(not_traj_lines_idx)):
                idx2 = not_traj_lines_idx[i]
                line_2 = interested_lines[idx2]
                all_dist[i] = line_association(line_1, line_2)

            min_idx = np.argmin(all_dist)
            if all_dist[min_idx] < 50:
                idx2 = not_traj_lines_idx[min_idx]
                traj_lines_idx.append(idx2)
                not_traj_lines_idx.remove(idx2)
            elif all_dist[min_idx] < 100:
                idx2 = not_traj_lines_idx[min_idx]
                tmp_line = interested_lines[idx2][0]
                if np.linalg.norm(tmp_line[2:] - tmp_line[:2]) > all_dist[min_idx]:
                    traj_lines_idx.append(idx2)
                    not_traj_lines_idx.remove(idx2)
                else:
                    break
            else:
                break

        lines.append(traj_lines_idx)

    if len(lines) == 0:
        print("This frame has no golf club 0: ", img_idx)
        conf_2d = check_hosel(results_2d, conf_2d)
        return results_2d, conf_2d, ref_club, ref_club_length, ref_direction, ref_img_idx

    # Filter by angle and length
    club_line = []
    for i in range(len(lines)):
        traj_lines_idx = lines[i]
        start_idx = traj_lines_idx[0]
        end_idx = traj_lines_idx[-1]
        traj = np.array([interested_lines[start_idx][0][0], interested_lines[start_idx][0][1],
                         interested_lines[end_idx][0][2], interested_lines[end_idx][0][3]])

        club_line.append(traj)

    length = np.zeros(len(club_line))
    for i in range(len(club_line)):
        line = club_line[i]
        length[i] = np.linalg.norm(line[2:] - line[:2])

    max_length_idx = np.argsort(length)[-1]
    club_line = np.array(club_line[max_length_idx])

    # Final filter
    club_length = np.linalg.norm(club_line[2:] - club_line[:2])
    if img_idx - ignore_frames == 0:
        ref_club_length = club_length

    if ref_club_length != None:
        if club_length < ref_club_length * 0.6:
            club_line = np.zeros((1, 4))

    if np.all(club_line == 0):
        print("This frame has no golf club 1: ", img_idx)
        conf_2d = check_hosel(results_2d, conf_2d)
        pass
    else:
        # Compare the results from LPN and LSD, and merge them together
        lpn_hosel = copy.deepcopy(results_2d[0, 36, :])
        lsd_start = copy.deepcopy(club_line[:2])
        lsd_end = copy.deepcopy(club_line[2:])
        foot_point = get_foot_point(lpn_hosel, lsd_start, lsd_end)

        direction1 = lsd_end - lsd_start
        direction2 = lpn_hosel - lsd_start

        angle = calc_angle(direction1, direction2)
        if angle < 0.94:
            results_2d[0, 36, :] = lsd_end
            conf_2d[0, 36, 0] = 1
        else:
            results_2d[0, 36, :] = foot_point

        ref_club = np.append(results_2d[0, 35, :], results_2d[0, 36, :])
        ref_club_length = club_length
        ref_direction = results_2d[0, 36, :] - results_2d[0, 35, :]
        ref_direction = ref_direction / np.linalg.norm(ref_direction)
        ref_img_idx = img_idx

    return results_2d, conf_2d, ref_club, ref_club_length, ref_direction, ref_img_idx


# 3D Inference functions
def bbox_detection(model, image, flag=0):
    # flag = 0 means RGB format
    if flag == 0:
        img = image
    else:
        img = image[:, :, ::-1]

    # Inference
    results = model(img, size=640)  # includes NMS

    df = results.pandas().xyxy[0]  # img1 predictions (pandas)
    bbox = np.zeros(4)
    for index, row in df.iterrows():
        if row['name'] == 'person' and row['confidence'] > 0.85:
            bbox = np.array([row['xmin'], row['ymin'], row['xmax'] - row['xmin'], row['ymax'] - row['ymin']])

    return bbox


def adjust_bbox(x, y, w, h, alpha, beta, video_info):
    x = x - alpha * w if x - alpha * w > 0 else 0
    y = y - beta * h if y - beta * h > 0 else 0
    w = (1 + 2 * alpha) * w if (1 + 2 * alpha) * w < video_info['width'] else video_info['width']
    h = (1 + 2 * beta) * h if (1 + 2 * beta) * h < video_info['height'] else video_info['height']
    return np.array([x, y, w, h])


def smoothing_2d(keypoints, smooth_window_size, polynomial_size):
    frames_num = keypoints.shape[0]
    kpts_num = keypoints.shape[1]
    result = np.zeros_like(keypoints)
    for idx in range(smooth_window_size, frames_num - smooth_window_size):
        collection = keypoints[idx - smooth_window_size: idx + smooth_window_size + 1, :, :]
        for kpt_id in range(kpts_num):
            indices = np.arange(collection.shape[0])
            data_x = collection[:, kpt_id, 0].reshape(-1)
            data_y = collection[:, kpt_id, 1].reshape(-1)
            x_coeff = np.polyfit(indices, data_x, polynomial_size)
            y_coeff = np.polyfit(indices, data_y, polynomial_size)
            x_poly_func = np.poly1d(x_coeff)
            y_poly_func = np.poly1d(y_coeff)

            # Modify confidence score
            update_x = x_poly_func(smooth_window_size)
            update_y = y_poly_func(smooth_window_size)
            dist = np.sqrt((update_x - keypoints[idx, kpt_id, 0]) ** 2 + (update_y - keypoints[idx, kpt_id, 1]) ** 2)
            update_conf = np.exp(-1 / 2 * (dist / 7) ** 2) * keypoints[idx, kpt_id, 2]

            result[idx, kpt_id, :] = np.array(
                [update_x, update_y, update_conf])

    for idx in range(smooth_window_size):
        result[idx, :, :] = result[smooth_window_size, :, :]

    for idx in range(frames_num - smooth_window_size, frames_num):
        result[idx, :, :] = result[frames_num - smooth_window_size - 1, :, :]

    return result


def smoothing_3d(keypoints, smooth_window_size, polynomial_size):
    '''
    keypoints: [frame_num, 39, 3]
    '''

    frames_num = keypoints.shape[0]
    kpts_num = keypoints.shape[1]
    result = np.zeros_like(keypoints)
    for idx in range(smooth_window_size, frames_num - smooth_window_size):
        collection = keypoints[idx - smooth_window_size: idx + smooth_window_size + 1, :, :]
        for kpt_id in range(kpts_num):
            indices = np.arange(collection.shape[0])
            data_x = collection[:, kpt_id, 0].reshape(-1)
            data_y = collection[:, kpt_id, 1].reshape(-1)
            data_z = collection[:, kpt_id, 2].reshape(-1)
            x_coeff = np.polyfit(indices, data_x, polynomial_size)
            y_coeff = np.polyfit(indices, data_y, polynomial_size)
            z_coeff = np.polyfit(indices, data_z, polynomial_size)
            x_poly_func = np.poly1d(x_coeff)
            y_poly_func = np.poly1d(y_coeff)
            z_poly_func = np.poly1d(z_coeff)
            result[idx, kpt_id, :] = np.array(
                [x_poly_func(smooth_window_size), y_poly_func(smooth_window_size), z_poly_func(smooth_window_size)])

    for idx in range(smooth_window_size):
        result[idx, :, :] = result[smooth_window_size, :, :]

    for idx in range(frames_num - smooth_window_size, frames_num):
        result[idx, :, :] = result[frames_num - smooth_window_size - 1, :, :]

    return result


def normalize(skeleton, bbox):
    normalize_skeleton = np.zeros((skeleton.shape[0], skeleton.shape[1], 2))

    for i in range(skeleton.shape[0]):
        points = skeleton[i, :, :2]
        box = bbox[i, :]
        norm_points = np.zeros_like(points)

        for j in range(points.shape[0]):
            norm_points[j, 0] = (points[j, 0] - box[0]) / box[2] * 2 - 1
            norm_points[j, 1] = (points[j, 1] - box[1]) / box[2] * 2 - box[3] / box[2]
        normalize_skeleton[i, :, :] = norm_points

    return normalize_skeleton


def compute_rescale_translation(skeletons3d, unnormalize_data_2d, camera_intrinsics, measurement, frame_id,
                                scale_factor, height, height_flag, translation):
    # skeletons3d: (1, 1, 39 ,3)
    # unnormalize_results_2d: (39, 2)

    max_y = (skeletons3d[0, 0, 5, 1] + skeletons3d[0, 0, 6, 1]) / 2
    min_y = np.min(skeletons3d[0, 0, [25, 27], 1])
    h = abs(max_y - min_y)

    # 1. Rescale 3D skeleton
    hip_diff_3d = (skeletons3d[0, 0, 12, :] - skeletons3d[0, 0, 10, :]).reshape(-1, 1)

    if frame_id == 0:
        dist = np.linalg.norm(hip_diff_3d)
        scale_factor = measurement['hip_distance'] / dist

    # 2. Compute left and right hip under camera coordinate
    left_hip_2d = np.append(unnormalize_data_2d[10, :], 1).reshape(-1, 1)
    left_hip_cam = np.linalg.inv(camera_intrinsics).dot(left_hip_2d)
    right_hip_2d = np.append(unnormalize_data_2d[12, :], 1).reshape(-1, 1)
    right_hip_cam = np.linalg.inv(camera_intrinsics).dot(right_hip_2d)

    hip_diff_2d = right_hip_2d - left_hip_2d

    if height_flag:
        scale_factor = height / h

    if abs(hip_diff_2d[0]) > 10:
        # 3. Construct least square solution to get translation
        hip_diff_3d = hip_diff_3d * scale_factor
        A = np.concatenate((right_hip_cam, -1 * left_hip_cam), axis=-1)
        b = hip_diff_3d
        depth = np.linalg.inv(A.T.dot(A)).dot(A.T.dot(b))  # depth: (2, 1)

        # 4. Get hip camera coordinates with depth
        right_hip_cam = right_hip_cam * depth[0, 0]
        left_hip_cam = left_hip_cam * depth[1, 0]

        # 5. Get translation vector
        center = (left_hip_cam + right_hip_cam) / 2

        # Not fixing the depth
        translation = copy.deepcopy(center)

        # # # Fix the depth
        # if frame_id == 0:
        #     translation = copy.deepcopy(center)
        # else:
        #     translation = copy.deepcopy(center) * (translation[2, 0] / center[2, 0])

    # 6. Perform scale and translation on all key points on human body
    for i in range(3):
        skeletons3d[0, 0, :, i] = skeletons3d[0, 0, :, i] * scale_factor + translation[i, 0]

    if not height_flag:
        height_flag = True
        max_y = (skeletons3d[0, 0, 5, 1] + skeletons3d[0, 0, 6, 1]) / 2
        min_y = np.min(skeletons3d[0, 0, [25, 27], 1])
        height = abs(max_y - min_y)

    return skeletons3d, scale_factor, height, height_flag, translation


def compute_rescale_translation_pnp(skeletons3d, unnormalize_data_2d, camera_intrinsics, measurement, frame_id,
                                    scale_factor, height, height_flag, translation):
    # skeletons3d: (1, 1, 38 ,3)
    # unnormalize_results_2d: (38, 2)

    max_y = (skeletons3d[0, 0, 5, 1] + skeletons3d[0, 0, 6, 1]) / 2
    min_y = np.min(skeletons3d[0, 0, [25, 27], 1])
    h = abs(max_y - min_y)

    # 1. Rescale 3D skeleton
    hip_diff_3d = (skeletons3d[0, 0, 12, :] - skeletons3d[0, 0, 10, :]).reshape(-1, 1)

    if frame_id == 0:
        dist = np.linalg.norm(hip_diff_3d)
        scale_factor = measurement['hip_distance'] / dist
        height_flag = False

    # 2. Compute left and right hip under camera coordinate
    left_hip_2d = np.append(unnormalize_data_2d[10, :], 1).reshape(-1, 1)
    right_hip_2d = np.append(unnormalize_data_2d[12, :], 1).reshape(-1, 1)
    hip_diff_2d = right_hip_2d - left_hip_2d

    if height_flag:
        scale_factor = height / h

    kpts_3d_idx = [1]
    kpts_3d_idx.extend([i for i in range(5, 34)])

    if abs(hip_diff_2d[0]) > 10:
        # 3. Perform rescale on all body keypoints
        all_joints_3d = skeletons3d[0, 0, kpts_3d_idx, :] * scale_factor
        all_joints_2d = unnormalize_data_2d[kpts_3d_idx, :]
        dist_coeffs = np.zeros((4, 1))

        # 4. Construct PnP problem to compute rotation and translation
        _, rvec, tvec, inliers = cv2.solvePnPRansac(all_joints_3d,
                                                    all_joints_2d,
                                                    camera_intrinsics,
                                                    dist_coeffs,
                                                    flags=cv2.SOLVEPNP_EPNP)
        translation = tvec

    # 5. Perform scale and translation on all key points on human body
    skeletons3d[0, 0, :, :] = skeletons3d[0, 0, :, :] * scale_factor + translation.T

    if not height_flag:
        height_flag = True
        max_y = (skeletons3d[0, 0, 5, 1] + skeletons3d[0, 0, 6, 1]) / 2
        min_y = np.min(skeletons3d[0, 0, [25, 27], 1])
        height = abs(max_y - min_y)

    return skeletons3d, scale_factor, height, height_flag, translation


def compute_rescale_translation_pnp_1(skeletons3d, unnormalize_data_2d, camera_intrinsics, measurement, frame_id,
                                      scale_factor, height, height_flag, translation, conf2d):
    # skeletons3d: (1, 1, 38 ,3)
    # unnormalize_results_2d: (38, 2)
    # conf2d: (38, )

    max_y = (skeletons3d[0, 0, 5, 1] + skeletons3d[0, 0, 6, 1]) / 2
    min_y = np.min(skeletons3d[0, 0, [25, 27], 1])
    h = abs(max_y - min_y)

    # 1. Rescale 3D skeleton
    hip_diff_3d = (skeletons3d[0, 0, 12, :] - skeletons3d[0, 0, 10, :]).reshape(-1, 1)

    if frame_id == 0:
        dist = np.linalg.norm(hip_diff_3d)
        scale_factor = measurement['hip_distance'] / dist
        height_flag = False

    # 2. Compute left and right hip under camera coordinate
    left_hip_2d = np.append(unnormalize_data_2d[10, :], 1).reshape(-1, 1)
    right_hip_2d = np.append(unnormalize_data_2d[12, :], 1).reshape(-1, 1)
    hip_diff_2d = right_hip_2d - left_hip_2d

    if height_flag:
        scale_factor = height / h

    kpts_3d_idx = [1]
    kpts_3d_idx.extend([i for i in range(5, 34)])
    # kpts_3d_idx = [5, 6, 10, 12, 15, 16, 18, 19, 24, 25, 26, 27]

    if abs(hip_diff_2d[0]) > 10:
        # 3. Perform rescale on all body keypoints
        # all_joints_3d: (38, 3), all_joints_2d: (38, 2)
        all_joints_3d = skeletons3d[0, 0, :, :] * scale_factor
        all_joints_2d = unnormalize_data_2d
        all_joints_2d = np.concatenate((all_joints_2d, np.ones((all_joints_2d.shape[0], 1))), axis=1)
        all_joints_2d = (np.linalg.inv(camera_intrinsics) @ all_joints_2d.T).T

        A = np.zeros((len(kpts_3d_idx) * 2, 3))
        b = np.zeros((len(kpts_3d_idx) * 2, 1))
        for i, kpts_idx in enumerate(kpts_3d_idx):
            all_joints_2d[kpts_idx] = all_joints_2d[kpts_idx] / all_joints_2d[kpts_idx, 2]
            A[i * 2, 0] = -1
            A[i * 2, 2] = all_joints_2d[kpts_idx, 0]
            A[i * 2 + 1, 1] = -1
            A[i * 2 + 1, 2] = all_joints_2d[kpts_idx, 1]
            b[i * 2, 0] = all_joints_3d[kpts_idx, 0] - all_joints_3d[kpts_idx, 2] * all_joints_2d[kpts_idx, 0]
            b[i * 2 + 1, 0] = all_joints_3d[kpts_idx, 1] - all_joints_3d[kpts_idx, 2] * all_joints_2d[kpts_idx, 1]

            A[i * 2, :] *= conf2d[kpts_idx]
            A[i * 2 + 1, :] *= conf2d[kpts_idx]
            b[i * 2, :] *= conf2d[kpts_idx]
            b[i * 2 + 1, :] *= conf2d[kpts_idx]

        ATA = np.dot(A.T, A)
        ATb = np.dot(A.T, b)
        translation = np.dot(np.linalg.inv(ATA), ATb)

    # 5. Perform scale and translation on all key points on human body
    skeletons3d[0, 0, :, :] = skeletons3d[0, 0, :, :] * scale_factor + translation.T

    if not height_flag:
        height_flag = True
        max_y = (skeletons3d[0, 0, 5, 1] + skeletons3d[0, 0, 6, 1]) / 2
        min_y = np.min(skeletons3d[0, 0, [25, 27], 1])
        height = abs(max_y - min_y)

    return skeletons3d, scale_factor, height, height_flag, translation


def compute_rescale(skeletons3d, unnormalize_data_2d, camera_intrinsics, measurement, frame_id,
                    scale_factor, height, height_flag, translation, conf2d):
    # skeletons3d: (1, 1, 38 ,3)
    # unnormalize_results_2d: (38, 2)
    # conf2d: (38, )

    max_y = (skeletons3d[0, 0, 5, 1] + skeletons3d[0, 0, 6, 1]) / 2
    min_y = np.min(skeletons3d[0, 0, [25, 27], 1])
    h = abs(max_y - min_y)

    # 1. Rescale 3D skeleton
    hip_diff_3d = (skeletons3d[0, 0, 12, :] - skeletons3d[0, 0, 10, :]).reshape(-1, 1)

    if frame_id == 0:
        dist = np.linalg.norm(hip_diff_3d)
        scale_factor = measurement['hip_distance'] / dist
        height_flag = False

    # 2. Compute left and right hip under camera coordinate
    left_hip_2d = np.append(unnormalize_data_2d[10, :], 1).reshape(-1, 1)
    right_hip_2d = np.append(unnormalize_data_2d[12, :], 1).reshape(-1, 1)
    hip_diff_2d = right_hip_2d - left_hip_2d

    if height_flag:
        scale_factor = height / h

    # 5. Perform scale and translation on all key points on human body
    skeletons3d[0, 0, :, :] = skeletons3d[0, 0, :, :] * scale_factor

    if not height_flag:
        height_flag = True
        max_y = (skeletons3d[0, 0, 5, 1] + skeletons3d[0, 0, 6, 1]) / 2
        min_y = np.min(skeletons3d[0, 0, [25, 27], 1])
        height = abs(max_y - min_y)

    return skeletons3d, scale_factor, height, height_flag, translation


def compute_3d_points_on_head(head_2d, camera_intrinsics, measurement):
    model_points = np.array([measurement['top_of_head'],
                             measurement['nose'],
                             measurement['left_ear'],
                             measurement['right_ear']], dtype=np.float64)
    image_points = np.squeeze(head_2d)
    dist_coeffs = np.zeros((4, 1))

    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points,
                                                                  image_points,
                                                                  camera_intrinsics,
                                                                  dist_coeffs,
                                                                  flags=cv2.SOLVEPNP_EPNP)
    rotM = cv2.Rodrigues(rotation_vector)[0]

    head_3d = rotM.dot(model_points.T) + translation_vector

    return head_3d.T


def inference_3d_skeleton(video_path, bbox_detector, model_2d_path, model_3d_path, ignore_seconds,
                          camera_intrinsics=None, hip_distance=None, num_kpts=38):
    metadata = skvideo.io.ffprobe(video_path)
    frames_num = int(metadata["video"]['@nb_frames'])
    frames_rate = int(metadata['video']['@r_frame_rate'].split('/')[0])
    video_info = {"width": 0, "height": 0, "frames_rate": frames_rate, "ignore_seconds": ignore_seconds,
                  "frames_num": frames_num}

    results_2d = np.zeros((1, num_kpts, 3))
    bbox_2d = np.zeros((1, 4))

    # 2D inference
    # Fix bbox as the location in the first frame
    v = skvideo.io.vreader(video_path)
    for i, frame in enumerate(v):
        if i < ignore_seconds * frames_rate:
            continue
        height, width = frame.shape[:2]
        video_info["height"] = height
        video_info["width"] = width
        tmp_bbox = bbox_detection(bbox_detector, frame, flag=0)
        if tmp_bbox[0] != 0 and tmp_bbox[1] != 0 and tmp_bbox[2] != 0 and tmp_bbox[3] != 0:
            x, y, w, h = tmp_bbox
            tmp_bbox = adjust_bbox(x, y, w, h, 0.2, 0.1, video_info)
            bbox_2d = tmp_bbox.reshape(1, 4)
            break

    print(video_info)
    ref_club = None
    ref_club_length = None
    ref_direction = None
    ref_img_idx = 0
    print("----------Doing 2D inference----------")

    # skvideo.io results in RGB format
    # 120 fps
    v = skvideo.io.vreader(video_path)
    for i, frame in enumerate(v):
        if i < frames_rate * ignore_seconds:
            continue

        bbox = bbox_2d[-1, :]
        preds, conf = inference_2d_tf.inference_2d_skeleton(frame, model_2d_path, bbox, flag=0)

        # Use line segment to detect golf club
        preds, conf, ref_club, ref_club_length, ref_direction, ref_img_idx = \
            inference_2d_golf_club_line_segment(i, frames_rate * ignore_seconds, frame, preds, conf,
                                                ref_club, ref_club_length, ref_direction, ref_img_idx, fps=frames_rate)

        results_2d = np.append(results_2d, np.concatenate((preds, conf), axis=-1), axis=0)

        x = np.min(results_2d[-1, :, 0])
        y = np.min(results_2d[-1, :, 1])
        w = np.max(results_2d[-1, :, 0]) - np.min(results_2d[-1, :, 0])
        h = np.max(results_2d[-1, :, 1]) - np.min(results_2d[-1, :, 1])

        new_bbox = adjust_bbox(x, y, w, h, 0.15, 0.15, video_info)
        if i < frames_num:
            if abs(new_bbox[2] - bbox[2]) > 0.1 * bbox[2]:
                bbox_2d = np.append(bbox_2d, new_bbox.reshape(1, 4), axis=0)
            else:
                bbox_2d = np.append(bbox_2d, bbox.reshape(1, 4), axis=0)
        else:
            break

    results_2d = np.delete(results_2d, 0, axis=0)

    # # 2D smoothing
    # print("----------Doing 2D Smoothing----------")
    # smooth_window_size = 15
    # polynomial_size = 2
    # results_2d = smoothing_2d(results_2d, smooth_window_size, polynomial_size)

    frames_num = results_2d.shape[0]
    smooth_window_size = 15
    polynomial_size = 2

    print("----------Doing 3D inference----------")
    # Adjust bbox for 3D normalization
    x = np.min(results_2d[0, :, 0])
    y = np.min(results_2d[0, :, 1])
    w = np.max(results_2d[0, :, 0]) - np.min(results_2d[0, :, 0])
    h = np.max(results_2d[0, :, 1]) - np.min(results_2d[0, :, 1])

    bbox_2d = np.tile(adjust_bbox(x, y, w, h, 0.15, 0.15, video_info), (frames_num, 1))

    # 3D inference
    unnormalize_results_2d = copy.deepcopy(results_2d[:, :, :2])  # (frames_num, 38, 2)
    normalize_results_2d = normalize(results_2d, bbox_2d)  # (frames_num, 38, 2), no confidence
    normalize_results_2d = np.concatenate((normalize_results_2d,
                                           results_2d[:, :, 2][:, :, np.newaxis]), axis=2)  # add confidence

    results_3d_raw = np.zeros((frames_num, num_kpts, 3))
    results_3d = np.zeros((frames_num, num_kpts, 3))

    w_size = 27
    center = w_size // 2

    if camera_intrinsics is None:
        default_fx, default_fy = 2900, 2900
        default_cx, default_cy = 2000, 1500
        cx, cy = video_info["width"] // 2, video_info["height"] // 2
        ratio = min(default_cx / cx, default_cy / cy)

        camera_intrinsics = np.array([[default_fx / ratio, 0, cx],
                                      [0, default_fy / ratio, cy],
                                      [0, 0, 1]], dtype=np.float32)

    measurement = {'top_of_head': [0, 100, 0], 'nose': [0, 0, 40], 'left_ear': [-70, 0, 10], 'right_ear': [70, 0, 10],
                   'hip_distance': 190.69 if hip_distance is None else hip_distance, 'height': 0, 'wing_length': 0}

    scale_factor = 0
    height = 0
    height_flag = False
    translation = np.zeros((3, 1))
    num_kpts_vp3d = 33

    for i in range(frames_num):
        skeletons2d = np.zeros((1, w_size, num_kpts, 3))
        skeletons3d = np.zeros((1, 1, num_kpts, 3))

        for j in range(max(-i, -center), min(center, (normalize_results_2d.shape[0] - i))):
            skeletons2d[0, center + j, :, :] = normalize_results_2d[i + j, :, :]

        # Pad left
        for j in range(0, center - i):
            skeletons2d[0, j, :, :] = normalize_results_2d[0, :, :]

        # Pad right
        for j in range(center + normalize_results_2d.shape[0] - i, w_size):
            skeletons2d[0, j, :, :] = normalize_results_2d[normalize_results_2d.shape[0] - 1, :, :]

        # Only use 33 keypoints for videopose3d
        model = model3d.TemporalModelTF(num_joints_in=num_kpts_vp3d, in_features=2, num_joints_out=num_kpts_vp3d,
                                        filter_widths=[3 for i in range(3)])
        model(np.zeros((1, 27, num_kpts_vp3d, 2)))
        model.load_weights(model_3d_path)

        input_data = np.zeros((1, w_size, num_kpts_vp3d, 3), dtype=np.float32)
        input_data[0, :, :, :] = np.concatenate((skeletons2d[0, :, 1:2, :], skeletons2d[0, :, 5:37, :]), axis=-2)

        output_data = model(input_data)

        skeletons3d[0, 0, 1, :] = output_data[0, 0, 0, :]
        skeletons3d[0, 0, 5:37, :] = output_data[0, 0, 1:, :]

        # Compute translation
        skeletons3d, scale_factor, height, height_flag, translation = compute_rescale_translation_pnp_1(
            skeletons3d, unnormalize_results_2d[i, :, :], camera_intrinsics,
            measurement, i, scale_factor, height, height_flag, translation, conf)

        # # Compute 3D coordinates on head
        # head = compute_3d_points_on_head(unnormalize_results_2d[i, :4, :], camera_intrinsics, measurement)
        # skeletons3d[0, 0, :4, :] = head

        results_3d[i, :, :] = np.squeeze(skeletons3d, axis=1)

    # # Subtract pelvis
    # pelvis = (results_3d[:, 10, :] + results_3d[:, 12, :]) / 2
    # results_3d = results_3d - pelvis[:, np.newaxis, :]

    # 3D smoothing
    print("----------Doing 3D Smoothing----------")
    results_3d = smoothing_3d(results_3d, smooth_window_size, polynomial_size)

    return results_2d, bbox_2d, results_3d, video_info


def read_2d_json(json_path):
    json2d_file = []
    with open(json_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            json2d_file.append(json.loads(line))
        frames_num = len(json2d_file)
        results_2d = np.ones((frames_num, 38, 3))
        for i in range(frames_num):
            for j in range(38):
                results_2d[i, j, :] = np.array(
                    [json2d_file[i]['joints'][str(j)]['x'],
                     json2d_file[i]['joints'][str(j)]['y'],
                     json2d_file[i]['joints'][str(j)]['z']])
    return results_2d


def write_2d_json(skeletons2d, bbox2d, json_2d_path, video_info, fps=120):
    json_2d = []
    starting_idx = video_info['frames_rate'] * video_info['ignore_seconds']
    for i in range(skeletons2d.shape[0]):
        dic = {'boundingBox': list(bbox2d[i, :]),
               'fps': fps,
               'frameId': starting_idx + i,
               'imageSize': {"height": video_info['height'], "width": video_info['width']},
               'joints': {}, 'score': np.mean(skeletons2d[i, :, 2])}
        for j in range(skeletons2d.shape[1]):
            dic['joints'][str(j)] = {'x': skeletons2d[i, j, 0],
                                     'y': skeletons2d[i, j, 1],
                                     'z': skeletons2d[i, j, 2]}
        json_2d.append(dic)

    json.dump(json_2d, open(json_2d_path, 'w'))

    # with open(json_2d_path, 'w', encoding='utf-8') as f:
    #     for i in range(len(json_2d)):
    #         json.dump(json_2d[i], f)
    #         f.write('\n')


def write_3d_json(skeletons3d, json_3d_path, video_info, fps=120):
    json_3d = []
    starting_idx = video_info['frames_rate'] * video_info['ignore_seconds']
    for i in range(skeletons3d.shape[0]):
        dic = {'fps': fps,
               'frameNum': starting_idx + i,
               'joints': {},
               'score': 1.0}
        for j in range(skeletons3d.shape[1]):
            dic['joints'][str(j)] = {'x': skeletons3d[i, j, 0],
                                     'y': skeletons3d[i, j, 1],
                                     'z': skeletons3d[i, j, 2]}
        json_3d.append(dic)

    json.dump(json_3d, open(json_3d_path, 'w'))


if __name__ == "__main__":
    model_2d_path = './models/tensorflow_model/lpn_pose2d_tf_38_140k.h5'
    model_3d_path = "./models/tensorflow_model/attn_videopose3d_33_1000K_deg0_90.h5"

    bbox_detector = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    ignore_seconds = 0

    root = "./data/tracker_accuracy_test/Swing Detection/202203_Sunridge_Canyon_AMM_Collection/exp_2"

    # Prepare to read AMM json file
    amm_data = {}
    amm_root = os.path.join(root, "results/AMM/data")
    for file in os.listdir(amm_root):
        if file == '.DS_Store':
            continue
        full_name = file.split('.')[0]
        amm_data[full_name] = os.path.join(amm_root, file)
    print(amm_data.keys())

    for player_name in os.listdir(os.path.join(root, "videos")):
        if player_name == '.DS_Store':
            continue

        videos_root = os.path.join(root, "videos", player_name)
        results_root = os.path.join(root, "results", player_name)
        if not os.path.exists(results_root):
            os.makedirs(os.path.join(results_root, "2d"))
            os.makedirs(os.path.join(results_root, "3d"))

        for file in os.listdir(videos_root):
            if file == '.DS_Store':
                continue

            video_path = os.path.join(videos_root, file)
            video_name = (video_path.split('/')[-1]).split('.')[0]

            # Read hip distance from AMM data
            # amm_filename = amm_data[video_name][0]
            # amm_path = amm_data[video_name][1]
            amm_path = amm_data[player_name]
            amm_json = json.load(open(amm_path))
            frame = amm_json[0]
            hip = np.array([frame['joints']['12']['x'] - frame['joints']['10']['x'],
                            frame['joints']['12']['y'] - frame['joints']['10']['y'],
                            frame['joints']['12']['z'] - frame['joints']['10']['z']])
            hip_distance = np.linalg.norm(hip)
            print("Hip distance for {} is:\t {}".format(player_name, hip_distance))

            camera_intrinsics = np.array([[1371.8, 0., 539.4],
                                          [0., 1366.5, 973.2],
                                          [0., 0., 1.]], dtype=np.float32)
            skeletons2d, bbox2d, skeletons3d, video_info = inference_3d_skeleton(
                video_path,
                bbox_detector,
                model_2d_path,
                model_3d_path,
                ignore_seconds,
                camera_intrinsics,
                hip_distance,
                num_kpts=38)
            print(skeletons3d.shape[0])

            fps = int(video_info['frames_rate'])

            json_2d_path = os.path.join(results_root, '2d/{}.json'.format(video_name))
            json_3d_path = os.path.join(results_root, '3d/{}.json'.format(video_name))
            write_2d_json(skeletons2d, bbox2d, json_2d_path, video_info, fps)
            write_3d_json(skeletons3d, json_3d_path, video_info, fps)

    # # Single video test
    # name = 'Swing_1646346017364'
    # video_path = './data/tracker_accuracy_test/Swing Detection/videos/{}.mp4'.format(name)
    # json_2d_path = './{}_2d.json'.format(name)
    # json_3d_path = './{}_3d.json'.format(name)
    # ignore_seconds = 4.5
    # fps = 120
    #
    # skeletons2d, bbox2d, skeletons3d, video_info = inference_3d_skeleton(
    #     video_path,
    #     bbox_detector,
    #     model_2d_path,
    #     model_3d_path,
    #     ignore_seconds,
    #     camera_intrinsics=None,
    #     hip_distance=None,
    #     num_kpts=38)
    #
    # write_2d_json(skeletons2d, bbox2d, json_2d_path, video_info, fps)
    # write_3d_json(skeletons3d, json_3d_path, video_info, fps)

    # # Multi video test
    # root = "./data/tracker_accuracy_test/Swing Detection/202203_Sunridge_Canyon_AMM_Collection/Brandi Luedtke"
    # video_root = os.path.join(root, "videos")
    # results_root = os.path.join(root, "results")
    #
    # ignore_seconds = 0
    # fps = 120
    #
    # if not os.path.exists(os.path.join(results_root, "2d")):
    #     os.makedirs(os.path.join(results_root, "2d"))
    #     os.makedirs(os.path.join(results_root, "3d"))
    #
    # for file in os.listdir(video_root):
    #     if file == ".DS_Store":
    #         continue
    #     print(file)
    #     name = file.split('.')[0]
    #     video_path = os.path.join(video_root, file)
    #     json_2d_path = os.path.join(results_root, '2d', "{}_2d.json".format(name))
    #     json_3d_path = os.path.join(results_root, '3d', "{}_3d.json".format(name))
    #
    #     skeletons2d, bbox2d, skeletons3d, video_info = inference_3d_skeleton(
    #         video_path,
    #         bbox_detector,
    #         model_2d_path,
    #         model_3d_path,
    #         ignore_seconds,
    #         camera_intrinsics=None,
    #         hip_distance=None,
    #         num_kpts=38)
    #
    #     write_2d_json(skeletons2d, bbox2d, json_2d_path, video_info)
    #     write_3d_json(skeletons3d, json_3d_path, video_info)
