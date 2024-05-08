

import json
import argparse
import os,sys
import pprint
import shutil
# from videos_to_frames import save_frame
from glob import glob
import os
import copy
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2
import torch
from pathlib import Path
import cv2 as cv
from datetime import date

from smooth import *

from coco2via import coco2csv

import torchvision.transforms as transforms
from pose_hrnet import get_pose_net
from config import cfg
from config import update_config
from swing_stage_detection import swing_stage_detection

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')

    parser.add_argument('--dataDir',
                    help='Videos data directory',
                    type=str,
                    required=True,
                    default='')

    parser.add_argument('--output',
                        help='Output directory',
                        type=str,
                        default='')
    
    parser.add_argument('--model',
                    help='Model path',
                    type=str,
                    default='')
    parser.add_argument('--conf_thresh',
                    help='Output directory',
                    type=int,
                    default=0.8)

    parser.add_argument('--of_thresh',
                help='Output directory',
                type=int,
                default=0.7)

    args = parser.parse_args()

    return args


def save_coco(file, info, licenses, images, annotations, categories):
    with open(file, 'wt', encoding='UTF-8') as coco:
        json.dump({ 'info': info, 'licenses': licenses, 'images': images, 
            'annotations': annotations, 'categories': categories}, coco, indent=2, sort_keys=True)

def box2cs(x, y, width, height, aspect_ratio):
    w = width
    h = height

    pixel_std = 200

    center = np.array([x + w * 0.5, y + h * 0.5])

    if (w > aspect_ratio * h):
        h = (w * 1.0 / aspect_ratio)
    elif (w < aspect_ratio * h):
        w = h * aspect_ratio

    scale = np.array([w / pixel_std, h / pixel_std])
    if (center[0] != -1):
        for i in range(len(scale)):
            scale[i] = scale[i] * 1.25

    return [center, scale]


def preProcessImage(srcImage, center, scale, image_size=(192, 256)):
    # image_size = [192, 256]
    r = 0

    affineTransMat = get_affine_transform(center, scale, r, image_size)

    affineTransImg = cv2.warpAffine(srcImage,
                                    affineTransMat,
                                    image_size,
                                    flags=cv2.INTER_LINEAR)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    affineTransImg = (affineTransImg / 255 - mean) / std

    return affineTransImg

def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_affine_transform(
        center, scale, rot, output_size,
        shift=np.array([0, 0], dtype=np.float32), inv=0
):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def softmax(x):
    '''
    x: numpy.ndarray([])
    '''
    y = np.exp(x - np.max(x))
    f_x = y / np.sum(np.exp(x))
    return f_x

def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))

    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def get_final_preds(batch_heatmaps, center, scale):
    coords, maxvals = get_max_preds(batch_heatmaps[:, :, ...])

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-processing
    for n in range(coords.shape[0]):
        for p in range(coords.shape[1]):
            hm = batch_heatmaps[n][p]
            px = int(math.floor(coords[n][p][0] + 0.5))
            py = int(math.floor(coords[n][p][1] + 0.5))
            if 1 < px < heatmap_width - 1 and 1 < py < heatmap_height - 1:
                diff = np.array(
                    [
                        hm[py][px + 1] - hm[py][px - 1],
                        hm[py + 1][px] - hm[py - 1][px]
                    ]
                )
                coords[n][p] += np.sign(diff) * .25

    preds = coords.copy()

    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(
            coords[i], center[i], scale[i], [heatmap_width, heatmap_height]
        )

    return preds, maxvals


def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]



def inference_2d_skeleton(image, model_path, bbox, flag=0):
    # flag = 0 means RGB format
    if flag == 0:
        image = image
    else:
        image = image[:, :, ::-1]

    # 2D inference
    aspect_ratio = 192 / 256

    center, scale = box2cs(bbox[0], bbox[1], bbox[2], bbox[3], aspect_ratio)
    transformImage = preProcessImage(image, center, scale)

    interpreter = tf.lite.Interpreter(model_path=model_path)
    input_data = np.float32(transformImage)
    input_data = input_data[np.newaxis, ...]

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    output_data = np.transpose(output_data, (0, 3, 1, 2))

    preds, maxvals = get_final_preds(output_data, [center], [scale])

    return preds, maxvals

device = torch.device('cuda')

def get_model_pytorch(args):
    update_config(cfg, args)
    model = get_pose_net(cfg, is_train=False)
    model.load_state_dict(torch.load(args.model))
    model.eval()

    model.to(device)
    return model

def inference_2d_skeleton_pytorch(image, model_path, bbox, model = None, flag=0):
    # flag = 0 means RGB format
    if flag == 0:
        image = image
    else:
        image = image[:, :, ::-1]

    # 2D inference
    aspect_ratio = 192 / 256
    #     print(bbox)
    center, scale = box2cs(bbox[0], bbox[1], bbox[2], bbox[3], aspect_ratio)

    transformImage = preProcessImage(image, center, scale)

    
    input_data = np.float32(transformImage)
    # input_data = input_data[np.newaxis, ...]

    # model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
    #     cfg, is_train=False
    # )
    
    # logger.info('=> loading model from {}'.format(model_path))

    # model.load_state_dict(torch.load(model_state_file))

    # model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
    # normalize = transforms.Normalize(
    #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    # )
    transform = transforms.Compose([
        transforms.ToTensor(),
        # normalize,
    ])

    
    with torch.no_grad():
        input_t = transform(input_data)
        input_tensors = torch.unsqueeze(input_t, dim=0)
        output_data = model(input_tensors.to(device))
        # print(output_data.shape)
        # torch.Size([1, 38, 64, 48])
       # (0, 1, )
        #TF
        # (1, 64, 48, 38)
        # after transpose [1, 38, 64, 48]

        output_data = output_data.cpu().numpy()

    preds, maxvals = get_final_preds(output_data, [center], [scale])
    
    return preds, maxvals

bbox_detector = torch.hub.load('ultralytics/yolov5', 'yolov5s')

def bbox_detection(image, flag=0):
    # flag = 0 means RGB format
    if flag == 0:
        img = image
    else:
        img = image[:, :, ::-1]

    # Inference
    results = bbox_detector(img, size=640)  # includes NMS

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

from qc import quality_estimation
import albumentations as A

def video_to_coco(images, annotations, resfile):
    info = dict()
    info['date_created'] = "Mon, 22 Aug 2022 21:31:53 +0000"
    info['url'] = 'unknown'
    info['year'] = 2021

    licenses = dict()
    licenses["id"] = 1
    licenses['name'] = "unknown"
    licenses['url'] = "unknown"

    categories = list()
    categorie = dict()
    categorie['id'] = 1
    categorie['keypoints'] = [
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "11",
            "12",
            "13",
            "14",
            "15",
            "16",
            "17",
            "18",
            "19",
            "20",
            "21",
            "22",
            "23",
            "24",
            "25",
            "26",
            "27",
            "28",
            "29",
            "30",
            "31",
            "32",
            "33",
            "34",
            "35",
            "36",
            "37"
        ]
    categorie['name'] = "human"
    categorie['skeleton'] = []
    categorie['supercategory'] = "human"
    categories.append(categorie)
    print(f'Saving results to {resfile}')
    save_coco(resfile, info, licenses, images, annotations, categories)

import scipy.stats
import random
import time

import numpy as np
import copy
import cv2

import ffmpeg

def adjust_bbox(x, y, w, h, alpha, beta, height, width):
    x = x - alpha * w if x - alpha * w > 0 else 0
    y = y - beta * h if y - beta * h > 0 else 0
    w = (1 + 2 * alpha) * w if (1 + 2 * alpha) * w < width else width
    h = (1 + 2 * beta) * h if (1 + 2 * beta) * h < height else height
    return np.array([x, y, w, h])

if __name__ == "__main__":
    args = parse_args()
    
    # test_image = plt.imread('test.jpg')
    CONF_THRESH = args.conf_thresh
    OF_THRESH = args.of_thresh
    vid_dir = args.dataDir
    output = args.output
    model_2d_path = args.model
    
    video_paths = os.listdir(vid_dir)
    print(video_paths)

    dstamp = date.today()
    run_date = dstamp.strftime("%Y%m%d")

    
    model_name = os.path.basename(model_2d_path)
    run_folder = os.path.join(output, f'{run_date}_{model_name}')
    Path(run_folder).mkdir(parents=True, exist_ok=True)
    output = run_folder
    # swing_stages = swing_stage_detection(vid_dir, '/home/gpuuser/HumanPose/quality-pipeline/json3d')
    # print(swing_stages)
    INTERVAL_BUFFER = 10
    INTERVAL_UPSWING = 5
    INTERVAL_DOWNSWING = 1
    INTERVAL_FINISH = 8

    for path in video_paths:
        print(path)
        output_dir = os.path.join(output, path)
        # video_to_coco(list(np.array(images)[top_indices]), list(np.array(annotations)[top_indices]), resfile)
        # video_to_coco(images, annotations, resfile)


        Path(output_dir).mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(os.path.join(vid_dir, path))
        # json2d, json3d = inference_api.get_3d_result(os.path.join(vid_dir, path), 121, 120, smooth_type='polynomial')
        
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frames_idxs = list(range(0,length))
        # stages = swing_stages[path]
        # start = stages['start']
        # end = stages['end']
        # stage = stages['position']

        prvs = None
        frm_num = 0
        gq_num = 0
        idx = 0
        kpts = []

        images = []
        annotations = []

        dist = scipy.stats.norm(0.5, 0.2)
        dist_list = [dist.pdf(t/length)/2 for t in range(0, length)]
        
        # normal_sampling = np.sort(np.random.choice(frames_idxs, int(length//3), p=dist_list))
        # print(normal_sampling)
        bbox_2d = np.zeros((1, 4))
        frames_num = 0
        frames = []
        while(cap.isOpened()):
            ret, frame = cap.read()
            height = frame.shape[0]
            width = frame.shape[1]
            break
        # while True:
        #     ret, frame = cap_b.read()
            
        #     if ret == False:
        #         cap_b.release()
        #         break
        #     height = frame.shape[0]
        #     width = frame.shape[1]

        #     tmp_bbox = bbox_detection(frame, flag=0)

        #     if tmp_bbox[0] != 0 and tmp_bbox[1] != 0 and tmp_bbox[2] != 0 and tmp_bbox[3] != 0:
        #         x, y, w, h = tmp_bbox
        #         bbox = tmp_bbox.reshape(1, 4)
        #         break
        #     else:
        #         continue
        # print(bbox)

        # cv.imwrite(os.path.join(output_dir, 'bbox.jpg'), cv.flip(frame[int(bbox_c[1]):int(bbox_c[1]+bbox_c[3]) , int(bbox_c[0]):int(bbox_c[0]+bbox_c[2]), :], -1))
        mse_scores = []
        i = 0

        process1 = (
            ffmpeg
            .input(os.path.join(vid_dir, path))
            .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run_async(pipe_stdout=True, quiet=True)

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

        for i, frame in enumerate(frames):
            # if i < ignore_seconds * frames_rate:
            #     continue
            height, width = frame.shape[:2]
            tmp_bbox = bbox_detection(frame, flag=0)
            # exit()
            if tmp_bbox[0] != 0 and tmp_bbox[1] != 0 and tmp_bbox[2] != 0 and tmp_bbox[3] != 0:
                x, y, w, h = tmp_bbox
                tmp_bbox = adjust_bbox(x, y, w, h, 0.15, 0.1, height, width)
                bbox_2d = tmp_bbox.reshape(1, 4)
                break
        if np.sum(bbox_2d) < 1e-4:
            continue

        batch_size = 16
        interval = 1
        for i, frame in enumerate(frames):
            bbox = bbox_2d[-1, :]
            idx+=1
            
            # if idx < stage['ADR'] or idx > stage['FIN']:
            #     continue    
            
            # if i < stage['ADR']:
            #     interval = INTERVAL_BUFFER
            # if i >= stage['ADR']:
            #     interval = INTERVAL_UPSWING
            # if i >= stage['DCV']:
            #     interval = INTERVAL_DOWNSWING
            # if i >= stage['FCV']:
            #     interval = INTERVAL_FINISH
            print(idx)
            # if i % interval != 0:
            #     continue
            # ret, frame = cap.read()
            
            # if ret == False:
            #     cap.release()
            #     break


            # r = random.random()
            # print(f'rand:{r}')
            # print(f'Dist: {dist_list[idx-1]}')
            # print(r/dist_list[idx-1])
            # if r/dist_list[idx-1]>1:
            #     print('Skipped')
            #     continue

            # threshold = 100.
            # if prvs is not None:
            #     diff = np.sum(np.absolute(frame[int(bbox_c[1]):int(bbox_c[1]+bbox_c[3]) , int(bbox_c[0]):int(bbox_c[0]+bbox_c[2]), :]-prvs[int(bbox_c[1]):int(bbox_c[1]+bbox_c[3]) , int(bbox_c[0]):int(bbox_c[0]+bbox_c[2]), :]))/np.size(frame[int(bbox_c[1]):int(bbox_c[1]+bbox_c[3]) , int(bbox_c[0]):int(bbox_c[0]+bbox_c[2]), :])
            #     # print(diff)
            #     if diff > threshold:
            #         image_name = f'{path}_{idx}.jpg'
            #         cv.imwrite(os.path.join(output_dir, image_name), cv.flip(frame, -1))
            #     else:
            #         continue
            
            # print('Processing')
            # print(idx)
            # if idx%2!=0:
            #     continue

            frm_num += 1
            # bbox = np.zeros((1, 4))
            # tmp_bbox = bbox_detection(frame, flag=0)
            # # exit()
            # print(tmp_bbox)
            # if tmp_bbox[0] != 0 and tmp_bbox[1] != 0 and tmp_bbox[2] != 0 and tmp_bbox[3] != 0:
            #     x, y, w, h = tmp_bbox
            #     bbox = tmp_bbox.reshape(1, 4)
            # else:
            #     continue
            # print(bbox)
            
            # bbox = [frame.shape[1] / 4, 0, frame.shape[1] / 2, frame.shape[0]]
            # args.cfg = 'tools/w48_256x192_coco_golf_no_aug.yaml'
            # model = get_model_pytorch(args)
            preds, conf = inference_2d_skeleton(frame, model_2d_path, bbox, flag=0)
            # preds, conf = inference_2d_skeleton_pytorch(frame, model_2d_path, bbox, model = model, flag=0)

            # results_2d = np.zeros((1, 38, 3))

             # Use line segment to detect golf club
            # frames_rate = 120
            # ignore_seconds = 0
            # ref_club = None
            # ref_club_length = None
            # ref_direction = None
            # ref_img_idx = 0
            # current_batch = 0

            # for fid in range(preds.shape[0]):
            #     if conf[fid, 36, 0] > 0.8:
            #         # line_tic = time.time()
                    
            #         preds[fid], conf[fid], ref_club, ref_club_length, ref_direction, ref_img_idx = \
            #             inference_2d_golf_club_line_segment(i - current_batch + fid + 1, frames_rate * ignore_seconds, frame, preds[fid], conf[fid],
            #                                                 ref_club, ref_club_length, ref_direction, ref_img_idx, fps=frames_rate)
            #         # line_time += time.time() - line_tic
            #     else:
            #         # preds_list[fid, 36] = np.array([0, 0])
            #         conf[fid, 36] = 0
            # results_2d = np.concatenate((results_2d, np.concatenate((preds, conf), axis=-1)), axis=0)
            # results_2d = np.delete(results_2d, 0, axis=0)
            # results_2d = smoothing_2d(results_2d, 31, 2, True, 31 // 2, False)
            # preds, conf = inference_2d_skeleton(frame, model_2d_path, bbox[0],flag=0)
            augmented_frames = []
            test_image = frame.copy()

            transform = A.Compose(
            [A.HorizontalFlip(p=1)], 
            keypoint_params=A.KeypointParams(format='xy'),
            bbox_params=A.BboxParams(format='coco', label_fields=[])
            )
            preds_t = preds.copy()
            preds_t[preds_t<0] = 0
            try:
                transformed = transform(image=frame, keypoints=preds_t[0], bboxes = [bbox])
            except:
                continue
            aug_frame, aug_kp, aug_bbox = transformed['image'], transformed['keypoints'], transformed['bboxes']
            # preds, conf = inference_2d_skeleton_pytorch(aug_frame, model_2d_path, list(aug_bbox[0]), model = model, flag=0)
            preds, conf = inference_2d_skeleton(aug_frame, model_2d_path, list(aug_bbox[0]), flag=0)
            mse = (np.square((aug_kp - preds[0])))
            # mse = np.abs((aug_kp - preds[0])/[width, height])
            mse_mean = np.mean(mse)
            print(f'mse: {mse_mean}')
            mse_scores.append(mse_mean)

            # for i in range(38):
            #     if 5 < preds[0, i, 0] < test_image.shape[1] - 5 and 5 < preds[0, i, 1] < test_image.shape[0] - 5:
            #         test_image[int(preds[0, i, 1] - 5):int(preds[0, i, 1] + 5), int(preds[0, i, 0]) - 5:int(preds[0, i, 0] + 5)] = [255, 0, 0]

            # for p in preds[0]:
            #     cv2.circle(test_image, (int(p[0]), int(p[1])), 8, (0, 0, 255), -1)
            # if conf[0].mean()>0.9:
            #     continue
            

            if conf[0][-2:].mean()<0.9:
                if mse[-2:].mean()>100:
                    image_name = f'{path}_{idx:04}.jpg'
                    
                    
                    cv.imwrite(os.path.join(output_dir, image_name), cv2.cvtColor(cv2.flip(test_image, -1), cv2.COLOR_BGR2RGB))
                    
        
                
                    keypoints = []
                    for i, p in enumerate(preds[0]):
                        keypoints.append(int(p[0]))
                        keypoints.append(int(p[1]))
                        # keypoints.append(str(p[0]))
                        # keypoints.append(str(p[1]))
                        if i in range(33, 38):
                            keypoints.append(str(0))
                        else:
                            keypoints.append(str(conf[0][i][0]))
                        # keypoints.append(int(mse[i].mean()))
                        # keypoints.append(int(mse[i].mean()))
                        # if conf[0][i]<0.8:
                        #     keypoints.append(0)
                        # else:
                        #     keypoints.append(1)
                    
            

                    image_json = dict()
                    ann = dict()
                    ann['keypoints'] = list(keypoints) 
                    ann['category_id'] = 1
                    ann['id'] = idx
                    ann['image_id']= idx
                    ann['iscrowd'] = 0
                    ann['num_keypoints'] = 37
                    ann['bbox'] = list(bbox)
                    ann['area'] = int(ann['bbox'][2]*ann['bbox'][3])
                    
                    annotations.append(ann)
                    
                    image_json['coco_url'] = "unknown"
                    image_json['date_captured'] = "unknown"
                    image_json['flickr_url'] = "unknown"
                    image_json['coco_url'] = "unknown"
                    image_json['height'] = frame.shape[0]
                    image_json['id'] = idx
                    image_json['license'] = 1
                    image_json['next_image_id'] = -1
                    image_json['prev_image_id'] = -1
                    image_json['video_name'] = path
                    image_json['width'] = frame.shape[1]
                    image_json['file_size'] = os.path.getsize(os.path.join(output_dir, image_name))
                    images.append(image_json)
                
                
           

            
            # test_image = aug_frame.copy()
            
            # for i in range(38):
            #     if 5 < preds[0, i, 0] < test_image.shape[1] - 5 and 5 < preds[0, i, 1] < test_image.shape[0] - 5:
            #         test_image[int(preds[0, i, 1] - 5):int(preds[0, i, 1] + 5), int(preds[0, i, 0]) - 5:int(preds[0, i, 0] + 5)] = [255, 0, 0]
            
            # cv.imwrite(os.path.join(output_dir, f'{path}_{idx}_flip.jpg'), cv.flip(test_image, -1))
                               
            # kq, okq = quality_estimation(preds, conf[0], frame, prvs=prvs, CONF_THRESH=0.8, OF_THRESH=0.7)
            # print(okq)
            # print(np.array(okq).mean())
            # print(conf[0].mean())
            # if conf[0].mean() > CONF_THRESH:
            #     if len(okq) > 1:
            #         if np.array(okq).mean() > OF_THRESH:
            #             image_name = f'{path}_{idx}.jpg'
            #             cv.imwrite(os.path.join(output_dir, image_name), cv.flip(frame, -1))
                            
            #             # kpts.append({
            #             #     'keypoints': kpt,
            #             #     'score': conf[idx],
            #             #     'image': imname
            #             # })
            #             keypoints = kq

            #             image_json = dict()
            #             ann = dict()
            #             ann['keypoints'] = keypoints    
            #             ann['category_id'] = 1
            #             ann['id'] = idx
            #             ann['image_id']= idx
            #             ann['iscrowd'] = 0
            #             ann['num_keypoints'] = 37
            #             ann['bbox'] = list(bbox)
            #             print(bbox)
            #             ann['area'] = int(ann['bbox'][2]*ann['bbox'][3])
                        
            #             annotations.append(ann)
                        
            #             image_json['coco_url'] = "unknown"
            #             image_json['date_captured'] = "unknown"
            #             image_json['flickr_url'] = "unknown"
            #             image_json['coco_url'] = "unknown"
            #             image_json['height'] = frame.shape[0]
            #             image_json['id'] = idx
            #             image_json['license'] = 1
            #             image_json['next_image_id'] = -1
            #             image_json['prev_image_id'] = -1
            #             image_json['video_name'] = path
            #             image_json['width'] = frame.shape[1]
            #             images.append(image_json)
                        
            #             continue

            gq_num += 1

            prvs = frame.copy()
        
        # top_percent = 0.4
        # top_scores = np.argsort(mse_scores)[::-1]
        # top_indices = top_scores[:int(top_percent*len(top_scores))]
        # print(f'Taking top {int(top_percent*len(images))} frames')
        # Coco for video saving
        resfile = os.path.join(output_dir, f'{path}.json')
        csvpath = os.path.join(output_dir, f'{path}.csv')
        # video_to_coco(list(np.array(images)[top_indices]), list(np.array(annotations)[top_indices]), resfile)
        video_to_coco(images, annotations, resfile)
        coco2csv(resfile, csvpath)
        print(idx)
        print(gq_num)
        print(frm_num)
