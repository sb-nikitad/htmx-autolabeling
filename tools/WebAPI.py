import os
import time
import torch
import tensorflow as tf
import numpy as np
import math
import cv2
import copy
import json
import skvideo.io
import matplotlib.pyplot as plt
import sys
from multiprocessing import Pool
import ffmpeg

sys.path.append(os.getcwd())

from tools.inference_2d_tf import *
from tools.golf_club_detection import *
from tools.smooth import *
from libs.utils import *
from libs.lpn.architecture.lpn_tf import *
from libs.videopose3d.common.modeltf_attn import *


class Inference(object):
    
    def __init__(self, GPU=[0, 1]):
        self.statuses = [{'stage': 'IDLE', 'num_frame': 0, 'current_frame': 0, 'req_id': 0} for i in range(len(GPU))]
        
        self.model_2d_path = 'models/tensorflow_model/faceOn2d.h5'
        self.model_3d_path = "models/tensorflow_model/attn_videopose3d_33_1000K_deg0.h5"

        self.bbox_detector = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        
        self.preprocessing_time = 0
        self.inference2d_time = 0
        self.post2d_time = 0
        
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
        for gpuId in GPU:
            config = tf.config.experimental.set_memory_growth(physical_devices[gpuId], True)
        # config = tf.config.experimental.set_memory_growth(physical_devices[1], True)
        # config = tf.config.experimental.set_memory_growth(physical_devices[2], True)
        # self.pool = Pool(processes=10)
        self.model2d = []
        self.model3d = []
        for i in range(len(GPU)):
            with tf.device('/GPU:%d' % GPU[i]):
                self.model2d.append(get_pose_net())
                self.model2d[i](np.zeros((1, 256, 192, 3)))
                self.model2d[i].load_weights(self.model_2d_path)
            
            with tf.device('/GPU:%d' % GPU[i]):
                self.model3d.append(TemporalModelTF(num_joints_in=33, in_features=2, num_joints_out=33,
                                            filter_widths=[3 for i in range(3)]))
                self.model3d[i](np.zeros((1, 27, 33, 3)))
                self.model3d[i].load_weights(self.model_3d_path)
        
    def reset(self, gpu=-1):
        if (gpu == -1):
            self.statuses = [{'stage': 'IDLE', 'num_frame': 0, 'current_frame': 0, 'req_id': 0} for i in range(len(self.statuses))]
        else:
            self.statuses[gpu]={'stage': 'IDLE', 'num_frame': 0, 'current_frame': 0, 'req_id': 0}
        self.preprocessing_time = 0
        self.inference2d_time = 0
        self.post2d_time = 0
        
    def allocate_gpu(self):
        for i in range(len(self.statuses)):
            if self.statuses[i]['stage'] == 'IDLE':
                self.statuses[i]['stage'] = 'RESERVED'
                return i
        return -1 
    
    def inference_2d_skeleton(self, image, bbox, flag=0, gpu=0):
        # flag = 0 means RGB format
        if flag == 0:
            image = image
        else:
            image = image[:, :, ::-1]

        # 2D inference
        aspect_ratio = 192 / 256

        center, scale = box2cs(bbox[0], bbox[1], bbox[2], bbox[3], aspect_ratio)
        transformImage = preProcessImage(image, center, scale)

        input_data = np.float32(transformImage)
        input_data = input_data[np.newaxis, ...]
        output_data = self.model2d[gpu](input_data)
        
        output_data = np.transpose(output_data, (0, 3, 1, 2))
        preds, maxvals = get_final_preds(output_data, [center], [scale])

        return preds, maxvals

    def inference_2d_skeleton_batch(self, image, bbox, batch_size, flag=0, gpu=0):
        # flag = 0 means RGB format
        if flag == 0:
            image = image
        else:
            image = image[:, :, ::-1]

        # 2D inference
        aspect_ratio = 192 / 256

        center, scale = box2cs(bbox[0], bbox[1], bbox[2], bbox[3], aspect_ratio)
        transformImage = np.zeros((batch_size, 256, 192, 3))
        tic = time.time()
        # results = []
        for i in range(batch_size):
        # result = self.pool.starmap_async(preProcessImage, [(image[i], center, scale) for i in range(batch_size)])
        # transformImage = result.get()
            transformImage[i] = preProcessImage(image[i], center, scale)
        # transformImage = np.array(transformImage)
        # exit()
        self.preprocessing_time += time.time() - tic

        input_data = np.array(transformImage).astype(np.float32)
        # input_data = input_data[np.newaxis, ...]
        tic = time.time()
        output_data = self.model2d[gpu](input_data)
        self.inference2d_time += time.time() - tic
        
        tic = time.time()
        output_data = np.transpose(output_data, (0, 3, 1, 2))
        preds, maxvals = get_final_preds(output_data, [center for _ in range(batch_size)], [scale for _ in range(batch_size)])
        self.post2d_time += time.time() - tic

        return preds, maxvals

    # 3D Inference functions
    def bbox_detection(self, image, flag=0):
        # flag = 0 means RGB format
        if flag == 0:
            img = image
        else:
            img = image[:, :, ::-1]

        # Inference
        results = self.bbox_detector(img, size=640)  # includes NMS

        df = results.pandas().xyxy[0]  # img1 predictions (pandas)
        bbox = np.zeros(4)
        largest_area = 0
        for index, row in df.iterrows():
            if row['name'] == 'person' and row['confidence'] > 0.85:
                area = (row['xmax'] - row['xmin']) * (row['ymax'] - row['ymin'])
                if area > largest_area:
                    bbox = np.array([row['xmin'], row['ymin'], row['xmax'] - row['xmin'], row['ymax'] - row['ymin']])
                    largest_area = area

        return bbox


    def adjust_bbox(self, x, y, w, h, alpha, beta, video_info):
        x = x - alpha * w if x - alpha * w > 0 else 0
        y = y - beta * h if y - beta * h > 0 else 0
        w = (1 + 2 * alpha) * w if (1 + 2 * alpha) * w < video_info['width'] else video_info['width']
        h = (1 + 2 * beta) * h if (1 + 2 * beta) * h < video_info['height'] else video_info['height']
        return np.array([x, y, w, h])


    def normalize(self, skeleton, bbox):
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


    def compute_rescale_translation_pnp(self, skeletons3d, unnormalize_data_2d, camera_intrinsics, measurement, frame_id,
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

        # kpts_3d_idx = [1]
        kpts_3d_idx = [i for i in range(5, 34)]
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


    def compute_rescale(self, skeletons3d, unnormalize_data_2d, camera_intrinsics, measurement, frame_id,
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


    def compute_3d_points_on_head(self, head_2d, camera_intrinsics, measurement):
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


    def inference_3d_skeleton(self, video_path, ignore_seconds,
                            camera_intrinsics=None, hip_distance=None, num_kpts=38,
                            fps=120,
                            smooth2d=True,
                            ws2d=31,
                            smooth3d=True,
                            ws3d=31,
                            smooth_type='polynomial',
                            smooth2d_club=True,
                            smooth3d_club=True,
                            ws2d_club=31,
                            ws3d_club=31,
                            website=False,
                            gpu=0):
        metadata = skvideo.io.ffprobe(video_path)
        frames_rate = int(metadata['video']['@r_frame_rate'].split('/')[0])

        results_2d = np.zeros((1, num_kpts, 3))
        bbox_2d = np.zeros((1, 4))

        # cap = cv2.VideoCapture(video_path)
        # frames_num = 0
        # frames = []
        # self.statuses[gpu]['stage'] = "Loading the Video"
        # print("----------Loading Video----------")
        # tic = time.time()
        # while(cap.isOpened()):
        #     ret, frame = cap.read()
        #     if not ret:
        #         break
        #     frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        #     frames_num += 1
        # # plt.imsave('test.jpg', frames[0])
        # print('load video', time.time() - tic)
        
        self.statuses[gpu]['stage'] = "Loading the Video"
        print("----------Loading Video----------")
        cap = cv2.VideoCapture(video_path)
        frames_num = 0
        frames = []
        while(cap.isOpened()):
            ret, frame = cap.read()
            height = frame.shape[0]
            width = frame.shape[1]
            break
        
        tic = time.time()
        process1 = (
            ffmpeg
            .input(video_path)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run_async(pipe_stdout=True)
        )
        
        while True:
            in_bytes = process1.stdout.read(height * width * 3)
            if not in_bytes:
                break
            frames.append(
                np
                .frombuffer(in_bytes, np.uint8)
                .reshape([height, width, 3])
            )
        frames = np.array(frames)
        frames_num = frames.shape[0]
        print('load video', time.time() - tic)
            
        video_info = {"width": 0, "height": 0, "frames_rate": frames_rate, "ignore_seconds": ignore_seconds,
                    "frames_num": frames_num}
            
        # 2D inference
        # Fix bbox as the location in the first frame
        # v = skvideo.io.vreader(video_path)
        for i, frame in enumerate(frames):
            if i < ignore_seconds * frames_rate:
                continue
            height, width = frame.shape[:2]
            video_info["height"] = height
            video_info["width"] = width
            tmp_bbox = self.bbox_detection(frame, flag=0)
            # exit()
            if tmp_bbox[0] != 0 and tmp_bbox[1] != 0 and tmp_bbox[2] != 0 and tmp_bbox[3] != 0:
                x, y, w, h = tmp_bbox
                tmp_bbox = self.adjust_bbox(x, y, w, h, 1.2, 0.3, video_info)
                bbox_2d = tmp_bbox.reshape(1, 4)
                break
        if np.sum(bbox_2d) < 1e-4:
            return None, None, None, None

        # print(video_info)
        ref_club = None
        ref_club_length = None
        ref_direction = None
        ref_img_idx = 0
        print("----------Doing 2D inference----------")

        # skvideo.io results in RGB format
        # 120 fps
        # v = skvideo.io.vreader(video_path)
        tic = time.time()
        self.statuses[gpu]['stage'] = "Processing 2d"
        self.statuses[gpu]['num_frame'] = frames_num
        line_time = 0
        batch_size = 32
        input_data = np.zeros((batch_size, video_info["height"], video_info["width"], 3))
        current_batch = 0
        for i, frame in enumerate(frames):
            self.statuses[gpu]['current_frame'] = i 

            bbox = bbox_2d[-1, :]
            # preds, conf = self.inference_2d_skeleton(frame, bbox, flag=0)
            # if i % batch_size != batch_size - 1 and i + 1 != len(frames):
            #     input_data[current_batch] = frame
            #     current_batch += 1
            #     continue
            # else:
            #     input_data[current_batch] = frame
            #     current_batch += 1
            #     preds_list, conf_list = self.inference_2d_skeleton_batch(input_data[:current_batch], bbox, current_batch)
            preds_list, conf_list = self.inference_2d_skeleton(frame, bbox, current_batch, gpu)

            # print(preds_list.shape, conf_list.shape)
            # Use line segment to detect golf club
            for fid in range(preds_list.shape[0]):
                if conf_list[fid, 36, 0] > 0.8:
                    line_tic = time.time()
                    preds_list[fid], conf_list[fid], ref_club, ref_club_length, ref_direction, ref_img_idx = \
                        inference_2d_golf_club_line_segment(i - current_batch + fid + 1, frames_rate * ignore_seconds, frame, preds_list[fid], conf_list[fid],
                                                            ref_club, ref_club_length, ref_direction, ref_img_idx, fps=frames_rate)
                    line_time += time.time() - line_tic
                else:
                    # preds_list[fid, 36] = np.array([0, 0])
                    conf_list[fid, 36] = 0
            current_batch = 0
            # print(preds_list.shape, conf_list.shape)
            results_2d = np.concatenate((results_2d, np.concatenate((preds_list, conf_list), axis=-1)), axis=0)

            x = np.min(results_2d[-1, results_2d[-1, :, 2] > 0, 0])
            y = np.min(results_2d[-1, results_2d[-1, :, 2] > 0, 1])
            w = np.max(results_2d[-1, :, 0]) - x
            h = np.max(results_2d[-1, :, 1]) - y

            new_bbox = self.adjust_bbox(x, y, w, h, 0.15, 0.15, video_info)
            if i < frames_num:
                if abs(new_bbox[2] - bbox[2]) > 0.1 * bbox[2]:
                    bbox_2d = np.append(bbox_2d, new_bbox.reshape(1, 4), axis=0)
                else:
                    bbox_2d = np.append(bbox_2d, bbox.reshape(1, 4), axis=0)
            else:
                break

        results_2d = np.delete(results_2d, 0, axis=0)

        # 2D smoothing
        print("2D inference:", time.time() - tic)
        print('line segment:', line_time)
        print('preprocessing:', self.preprocessing_time)
        print('inference:', self.inference2d_time)
        print('post processing:', self.post2d_time)
        smooth_window_size = ws2d // 2
        polynomial_size = 2
        if smooth2d:
            print("----------Doing 2D Smoothing----------")
            tic = time.time()
            self.statuses[gpu]['stage'] = "Smoothing 2d"
            if smooth_type == 'polynomial':
                results_2d = smoothing_2d(results_2d, smooth_window_size, polynomial_size, smooth2d_club, ws2d_club // 2, website)
            elif smooth_type == 'btw':
                json2d = generate_2d_json(results_2d, bbox_2d, video_info, fps)
                buildInfo = {
                        "app_version": "None",
                        "actionSdk_version": "None",
                        "unity_version": "None",
                        "pose2DModel_version": "lpn_pose2d_tf_38_148k",
                        "pose3DModel_version": "attn_videopose3d_33_1000K_deg0_90",
                        "app_buildType": "None",
                        "system_info": "None"
                    }
                json.dump({"buildInfo": buildInfo, "frames": json2d}, open('webapiJSON/json_2d.json', 'w'))
                os.system('java -jar trackers-1.0-SNAPSHOT-all.jar GENERATE_TRACKER_BATCH webapiJSON webapiJSON')
        
            print("2D smoothing:", time.time() - tic)

        frames_num = results_2d.shape[0]
        print("----------Doing 3D inference----------")
        tic = time.time()
        # Adjust bbox for 3D normalization
        x = np.min(results_2d[0, results_2d[0, :, 2] > 0, 0])
        y = np.min(results_2d[0, results_2d[0, :, 2] > 0, 1])
        w = np.max(results_2d[0, :, 0]) - x
        h = np.max(results_2d[0, :, 1]) - y
        
        bbox_3d = np.tile(self.adjust_bbox(x, y, w, h, 0.15, 0.15, video_info), (frames_num, 1))

        # 3D inference
        unnormalize_results_2d = copy.deepcopy(results_2d[:, :, :2])  # (frames_num, 38, 2)
        normalize_results_2d = self.normalize(results_2d, bbox_3d)  # (frames_num, 38, 2), no confidence
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
        self.statuses[gpu]['stage'] = "Processing 3d"

        pad2d_left = np.array([normalize_results_2d[0] for i in range(w_size // 2)])
        pad2d_right = np.array([normalize_results_2d[-1] for i in range(w_size // 2)])
        normalize_results_2d = np.concatenate([pad2d_left, normalize_results_2d, pad2d_right], axis=0)
        input_data = np.zeros((1, normalize_results_2d.shape[0], 33, 3), dtype=np.float32)
        input_data[0, :, :, :] = np.concatenate((normalize_results_2d[:, 0:1, :], normalize_results_2d[:, 5:37, :]), axis=-2)
        skeletons3d = np.zeros((1, frames_num, num_kpts, 3))
        
        output_data = self.model3d[gpu](input_data)
        
        skeletons3d[0, :, 0, :] = output_data[0, :, 0, :]
        skeletons3d[0, :, 1, :] = output_data[0, :, 0, :]
        skeletons3d[0, :, 5:37, :] = output_data[0, :, 1:, :]
        print("3D inference:", time.time() - tic)
        tic = time.time()
        
        for i in range(frames_num):
            self.statuses[gpu]['num_frame'] = i
            # skeletons2d = np.zeros((1, w_size, num_kpts, 3))
            # skeletons3d = np.zeros((1, 1, num_kpts, 3))

            # for j in range(max(-i, -center), min(center, (normalize_results_2d.shape[0] - i))):
            #     skeletons2d[0, center + j, :, :] = normalize_results_2d[i + j, :, :]

            # # Pad left
            # for j in range(0, center - i):
            #     skeletons2d[0, j, :, :] = normalize_results_2d[0, :, :]

            # # Pad right
            # for j in range(center + normalize_results_2d.shape[0] - i, w_size):
            #     skeletons2d[0, j, :, :] = normalize_results_2d[normalize_results_2d.shape[0] - 1, :, :]

            # # Only use 33 keypoints for videopose3d
            # input_data = np.zeros((1, w_size, 33, 3), dtype=np.float32)
            # input_data[0, :, :, :] = np.concatenate((skeletons2d[0, :, 1:2, :], skeletons2d[0, :, 5:37, :]), axis=-2)
            # conf = normalize_results_2d[i, :, 2]

            # output_data = self.model3d[gpu](input_data)
            # output_oks_score = interpreter.get_tensor(output_details[1]['index'])
            # skeletons3d[0, 0, 1, :] = output_data[0, 0, 0, :]
            # skeletons3d[0, 0, 5:37, :] = output_data[0, 0, 1:, :]
            # conf[1] = output_oks_score[0, 0, 0]
            # conf[5:37] = output_oks_score[0, 0, 1:]

            conf = results_2d[i, :, 2]
            # Compute translation
            skeleton3d, translation_pelvis, translation_root, root_y, height, club_length = \
                self.compute_rescale_translation_pnp(
                    skeletons3d[:, i:i+1, :, :], unnormalize_results_2d[i, :, :], camera_intrinsics,
                    measurement, i, translation_pelvis, translation_root, root_y, height, club_length, conf)

            # # Compute 3D coordinates on head
            # head = compute_3d_points_on_head(unnormalize_results_2d[i, :4, :], camera_intrinsics, measurement)
            # skeletons3d[0, 0, :4, :] = head

            results_3d[i, :, :] = np.squeeze(skeleton3d, axis=1)
        print("3D PnP:", time.time() - tic)

        # 3D smoothing
        smooth_window_size = ws3d // 2
        polynomial_size = 2
        if smooth3d:
            print("----------Doing 3D Smoothing----------")
            tic = time.time()
            self.statuses[gpu]['stage'] = "Smoothing 3d"
            results_3d = smoothing_3d(results_3d, smooth_window_size, polynomial_size, smooth3d_club, ws3d_club // 2, website)
            
            print("3D Smoothing:", time.time() - tic)

        return results_2d, bbox_2d, results_3d, video_info


    def get_3d_result(
            self, 
            video_path,
            hip_distance,
            fps=120,
            camera_intrinsics=np.array([[1371.8, 0., 539.4],
                                        [0., 1366.5, 973.2],
                                        [0., 0., 1.]], dtype=np.float32),
            smooth2d=True,
            ws2d=31,
            smooth3d=True,
            ws3d=31,
            smooth_type='polynomial',
            smooth2d_club=True,
            smooth3d_club=True,
            ws2d_club=31,
            ws3d_club=31,
            website=False,
            gpu=0
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
        ignore_seconds = 0

        skeletons2d, bbox2d, skeletons3d, video_info = self.inference_3d_skeleton(
            video_path,
            ignore_seconds,
            camera_intrinsics,
            hip_distance, num_kpts=38,
            fps=fps,
            smooth2d=smooth2d,
            ws2d=ws2d,
            smooth3d=smooth3d,
            ws3d=ws3d,
            smooth_type=smooth_type,
            smooth2d_club=smooth2d_club,
            smooth3d_club=smooth3d_club,
            ws2d_club=ws2d_club,
            ws3d_club=ws3d_club,
            website=website, 
            gpu=gpu)

        if skeletons2d is None:
            return None, None

        self.statuses[gpu]['stage'] = "Generating Results"
        json2d = generate_2d_json(skeletons2d, bbox2d, video_info, fps)
        json3d = generate_3d_json(skeletons3d, video_info, fps)
        self.reset(gpu)

        return json2d, json3d


if __name__ == "__main__":
    model_3d_path = "models/tensorflow_model/attn_videopose3d_33_1000K_deg0_90.h5"
    model3d = [0]
    model3d[0] = TemporalModelTF(num_joints_in=33, in_features=2, num_joints_out=33,
                                    filter_widths=[3 for i in range(3)])
    print(model3d[0](np.zeros((1, 30, 33, 3))).shape)
