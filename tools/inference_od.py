from unittest import result
import torch
import matplotlib.pyplot as plt
import skvideo.io
import argparse
import sys
import os
import numpy as np
from nanodet.data.transform import Pipeline
from nanodet.data.batch_process import stack_batch_img
from nanodet.data.collate import naive_collate
import pickle
sys.path.append(os.getcwd())

import libs.utils as utils

def adjust_bbox(x, y, w, h, alpha, beta, width, height):
    x = x - alpha * w if x - alpha * w > 0 else 0
    y = y - beta * h if y - beta * h > 0 else 0
    w = (1 + 2 * alpha) * w if x + (1 + 2 * alpha) * w < width else width - x
    h = (1 + 2 * beta) * h if y + (1 + 2 * beta) * h < height else height - y
    return np.array([x, y, w, h])

def batch_golfer_detect(frames, model, pipeline):
    meta_list = []
    for fid, frame in enumerate(frames):
        height, width = frames.shape[1:3]
        img_info = {"id": fid}
        img_info["height"] = height
        img_info["width"] = width
        meta = dict(img_info=img_info, raw_img=frame, img=frame)
        meta = pipeline(None, meta, [416, 416])
        meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1)).cuda()
        meta_list.append(meta)
        
    meta_list = naive_collate(meta_list)
    meta_list["img"] = stack_batch_img(meta_list["img"], divisible=32)
    results = model.inference(meta_list)
    # print(results)
    # exit()
    bboxs = np.zeros((frames.shape[0], 4))
    for fid in range(frames.shape[0]):
        if len(results[fid][0]) > 0:
            largest_area = 0
            for i in range(len(results[fid][0])):
                area = (results[fid][0][i][2] - results[fid][0][i][0]) * (results[fid][0][i][3] - results[fid][0][i][1])
                if results[fid][0][i][4] > 0.7 and area > largest_area:
                    bboxs[fid] = results[fid][0][i][:4]
                    largest_area = area
            x, y, w, h = bboxs[fid]
            bboxs[fid] = adjust_bbox(x, y, w, h, 0.1, 0.1, width, height)
    bboxs[:, 2] = bboxs[:, 2] - bboxs[:, 0]
    bboxs[:, 3] = bboxs[:, 3] - bboxs[:, 1]
    
    return bboxs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Object Detection Inference')
    parser.add_argument('--video', type=str, help='video path for inference')
    args = parser.parse_args()

    model = torch.load('models/nanodet.model').cuda()
    model.eval()

    pipeline = Pipeline(pickle.load(open('models/cfg.pipeline', 'rb')), True)
    
    batch_golfer_detect(np.zeros((10, 1920, 1080, 3)), model, pipeline)
    detect_result = []
    # v = skvideo.io.vreader(args.video)
    # print(args.video)
    # for i, frame in enumerate(v):
    #     print(i, end='\r')

    #     height, width = frame.shape[:2]
    #     img_info = {"id": 0}
    #     img_info["height"] = height
    #     img_info["width"] = width
    #     meta = dict(img_info=img_info, raw_img=frame, img=frame)
    #     meta = pipeline(None, meta, [416, 416])

    #     meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1))
    #     meta = naive_collate([meta])
    #     meta["img"] = stack_batch_img(meta["img"], divisible=32)
    #     result = model.inference(meta)

    #     detect_result.append({'frame_id': i, "bbox": result[0]})
    
    # json.dump(detect_result, open('%s.json' % args.video.split('.')[0], 'w'))
        