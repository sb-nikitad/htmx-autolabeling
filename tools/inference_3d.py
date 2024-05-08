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

from inference_2d import *
from golf_club_detection import *
from smooth import *
from utils_lib import *


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


def compute_rescale_translation_pnp(skeletons3d, unnormalize_data_2d, camera_intrinsics, measurement, frame_id,
                                    translation_pelvis, translation_root, root_y, height, club_length, conf2d):
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
    else:
        scale_factor = height / h

    # 2. Compute left and right hip under camera coordinate
    left_hip_2d = np.append(unnormalize_data_2d[10, :], 1).reshape(-1, 1)
    right_hip_2d = np.append(unnormalize_data_2d[12, :], 1).reshape(-1, 1)
    hip_diff_2d = right_hip_2d - left_hip_2d

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
        translation_pelvis = np.dot(np.linalg.inv(ATA), ATb).T

    # 5. Perform scale and translation on all key points on human body with FixRoot Z-Coordinate
    root_idx = 30
    root = copy.deepcopy(skeletons3d[0, 0, root_idx, :])  # left foot outside
    if frame_id == 0:
        translation_root = copy.deepcopy(
            skeletons3d[0, 0, root_idx, :] * scale_factor + translation_pelvis)
        root_y = copy.deepcopy(translation_root[0, 1])

    skeletons3d[0, 0, :, :2] = skeletons3d[0, 0, :, :2] * scale_factor + translation_pelvis[0, :2]
    skeletons3d[0, 0, :, 2] = (skeletons3d[0, 0, :, 2] - root[2]) * scale_factor + translation_root[0, 2]

    # 6. Constrain root point won't move negatively in vertical direction
    y_diff = skeletons3d[0, 0, root_idx, 2] - root_y
    if y_diff > 0:
        skeletons3d[0, 0, :, 1] -= root_y

    # 7. fix player height as the same as in the first frame
    if frame_id == 0:
        max_y = (skeletons3d[0, 0, 5, 1] + skeletons3d[0, 0, 6, 1]) / 2
        min_y = np.min(skeletons3d[0, 0, [25, 27], 1])
        height = abs(max_y - min_y)

    # fix club length as the same in the first frame
    club = skeletons3d[0, 0, 36, :] - skeletons3d[0, 0, 34, :]
    if frame_id == 0:
        club_length = np.linalg.norm(club)
    else:
        club_dir = club / np.linalg.norm(club)
        skeletons3d[0, 0, 36, :] = skeletons3d[0, 0, 34, :] + club_length * club_dir

    return skeletons3d, translation_pelvis, translation_root, root_y, height, club_length


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
        preds, conf = inference_2d_skeleton(frame, model_2d_path, bbox, flag=0)
        # Use line segment to detect golf club
        preds, conf, ref_club, ref_club_length, ref_direction, ref_img_idx = \
            inference_2d_golf_club_line_segment(i, frames_rate * ignore_seconds, frame, preds[0], conf[0],
                                                ref_club, ref_club_length, ref_direction, ref_img_idx, fps=frames_rate)
        preds = np.array([preds])
        conf = np.array([conf])
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

    # 2D smoothing
    print("----------Doing 2D Smoothing----------")
    smooth_window_size = 15
    polynomial_size = 2
    results_2d = smoothing_2d(results_2d, smooth_window_size, polynomial_size)

    frames_num = results_2d.shape[0]
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

    translation_pelvis = None
    translation_root = None
    root_y = None
    height = None
    club_length = None
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
        input_data = np.zeros((1, w_size, 33, 3), dtype=np.float32)
        input_data[0, :, :, :] = np.concatenate((skeletons2d[0, :, 1:2, :], skeletons2d[0, :, 5:37, :]), axis=-2)
        conf = normalize_results_2d[i, :, 2]

        interpreter = tf.lite.Interpreter(model_path=model_3d_path)

        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]['index'], input_data)

        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]['index'])
        # output_oks_score = interpreter.get_tensor(output_details[1]['index'])
        skeletons3d[0, 0, 1, :] = output_data[0, 0, 0, :]
        skeletons3d[0, 0, 5:37, :] = output_data[0, 0, 1:, :]
        # conf[1] = output_oks_score[0, 0, 0]
        # conf[5:37] = output_oks_score[0, 0, 1:]

        # Compute translation
        skeletons3d, translation_pelvis, translation_root, root_y, height, club_length = \
            compute_rescale_translation_pnp(
                skeletons3d, unnormalize_results_2d[i, :, :], camera_intrinsics,
                measurement, i, translation_pelvis, translation_root, root_y, height, club_length, conf)

        # # Compute 3D coordinates on head
        # head = compute_3d_points_on_head(unnormalize_results_2d[i, :4, :], camera_intrinsics, measurement)
        # skeletons3d[0, 0, :4, :] = head

        results_3d[i, :, :] = np.squeeze(skeletons3d, axis=1)

    # 3D smoothing
    print("----------Doing 3D Smoothing----------")
    results_3d = smoothing_3d(results_3d, smooth_window_size, polynomial_size)

    return results_2d, bbox_2d, results_3d, video_info


def get_3d_result(
        video_path,
        hip_distance,
        fps=120,
        camera_intrinsics=np.array([[1371.8, 0., 539.4],
                                    [0., 1366.5, 973.2],
                                    [0., 0., 1.]], dtype=np.float32),
):
    """
        video_path:         the absolute or relative path of the video, str
        hip_distance:       the length of the hip, float
        fps:                the frame rate per second of the video, int
        camera_intrinsics:  the camera intrinsic parameter of the video, 3x3 numpy array
                            [[fx,   0,  cx],
                             [ 0,  fy,  cy],
                             [ 0,   0,   1]]
                            fx, fy the focal length of x and y axes
                            cx, cy the coordinate of the sensor's center
    """
    model_2d_path = './models/lpn_pose2d_tflite_38_140k.tflite'
    model_3d_path = "./models/attn_videopose3d_33_1000k.tflite"

    bbox_detector = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    ignore_seconds = 0

    skeletons2d, bbox2d, skeletons3d, video_info = inference_3d_skeleton(
        video_path,
        bbox_detector,
        model_2d_path,
        model_3d_path,
        ignore_seconds,
        camera_intrinsics,
        hip_distance, num_kpts=38)

    json2d = generate_2d_json(skeletons2d, bbox2d, video_info, fps)
    json3d = generate_3d_json(skeletons3d, video_info, fps)

    return json2d, json3d


if __name__ == "__main__":
    model_2d_path = './models/lpn_pose2d_tflite_38_140k.tflite'
    model_3d_path = "./models/attn_videopose3d_33_1000k.tflite"

    bbox_detector = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    ignore_seconds = 0
    fps = 120

    root = './data/Smoothing Test/AB_Swings_Smoothing_Test'
    videos_root = root + '/videos/'
    results_root = root + '/results/4 points/9 frames/no2d_no3d'

    # Prepare to read AMM json file
    amm_data = {}
    amm_root = os.path.join(root, "AMM")
    for file in os.listdir(amm_root):
        if file == '.DS_Store':
            continue
        full_name = file.split('.')[0].split()
        amm_data[full_name[0][0] + full_name[1][0]] = os.path.join(amm_root, file)
    print(amm_data.keys())

    # Prepare output folder
    if not os.path.exists(results_root):
        os.makedirs(os.path.join(results_root, "2d"))
        os.makedirs(os.path.join(results_root, "3d"))

    for file in os.listdir(videos_root):
        if file == '.DS_Store':
            continue

        video_path = os.path.join(videos_root, file)
        video_name = (video_path.split('/')[-1]).split('.')[0]
        print(video_name)

        # Read hip distance from AMM data
        amm_path = amm_data[video_name[:2]]
        amm_json = json.load(open(amm_path))
        frame = amm_json[0]
        hip = np.array([frame['joints']['12']['x'] - frame['joints']['10']['x'],
                        frame['joints']['12']['y'] - frame['joints']['10']['y'],
                        frame['joints']['12']['z'] - frame['joints']['10']['z']])
        hip_distance = np.linalg.norm(hip)

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
            hip_distance, num_kpts=38)
        print(skeletons3d.shape[0])

        json_2d_path = os.path.join(results_root, '2d/{}.json'.format(video_name))
        json_3d_path = os.path.join(results_root, '3d/{}.json'.format(video_name))
        generate_2d_json(skeletons2d, bbox2d, json_2d_path, video_info, fps)
        generate_3d_json(skeletons3d, json_3d_path, video_info, fps)

    # # Single video test
    # video_path = '/Users/haoruiji/Downloads/IMG_8579.mov'
    # ignore_seconds = 0
    # fps = 120
    # if not os.path.exists('./vis'):
    #     os.makedirs('./vis')
    #
    # inference_3d_skeleton(
    #     video_path,
    #     bbox_detector,
    #     model_2d_path,
    #     model_3d_path,
    #     ignore_seconds,
    #     num_kpts=38)
