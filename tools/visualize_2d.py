import matplotlib.pyplot as plt
import os
from pycocotools.coco import COCO
import numpy as np
import json
import cv2

# root = './data/results/Friday_Michael V_0High_iPhone XR'
# name = 'IMG_0002'
# json_file = json.load(open(os.path.join(root, '3d', '{}_3d.json'.format(name))))
# json2d_file = []
# with open(os.path.join(root, '2d', '{}_2d.json'.format(name)), 'r', encoding='utf-8') as f:
#     for line in f.readlines():
#         json2d_file.append(json.loads(line))

# # Single video test
# root = './'
# name = 'Swing 2mp4'
# json_file = json.load(open(os.path.join(root, '{}_3d.json'.format(name))))
# json2d_file = []
# with open(os.path.join(root, '{}_2d.json'.format(name)), 'r', encoding='utf-8') as f:
#     for line in f.readlines():
#         json2d_file.append(json.loads(line))

# Single video test
# root = "./data/tracker_accuracy_test/results/new_test_set_result/120.8K-2d-120fps_attn_vp3d_200K_sigma_range0.2-3d-33/2d"
# name = 'IMG_9019_2d.json'
root = "/home/gpuuser/HumanPose/quality-pipeline/output/20221013_lpn_pose2d_tflite_38_140k.tflite/trim.4E8D19BF-8177-4EFE-9D7E-D1C1FA5A7BF3.MOV/"
name = 'trim.4E8D19BF-8177-4EFE-9D7E-D1C1FA5A7BF3.MOV.json'
json_file = []
with open(os.path.join(root, name), 'r', encoding='utf-8') as f:
    for line in f.readlines():
        json_file.append(json.loads(line))

all_keypoint = []
num_kpts = 38

for fid, frame in enumerate(json_file):
    if fid < 1:
        continue
    if fid % 2 != 0:
        continue
    print(fid)

    keypoints2d = np.zeros((num_kpts, 2))
    for i in range(num_kpts):
        keypoints2d[i] = np.array([json_file[fid]['joints'][str(i)]['x'],
                                   json_file[fid]['joints'][str(i)]['y']])

    image = np.zeros((1920 // 4, 1080 // 4, 3)) + 255
    image = np.uint8(image)
    for i in range(num_kpts):
        cv2.circle(image, (int(keypoints2d[i, 0]), int(keypoints2d[i, 1])), 2, (0, 255, 0), -1)

    # feet = [[27, 28], [27, 29], [28, 29], [30, 31], [31, 32], [30, 32]]
    # for pair in feet:
    #     cv2.line(image, (int(keypoints[pair[0], 0]) // 4, int(keypoints[pair[0], 1]) // 4),
    #              (int(keypoints[pair[1], 0]) // 4, int(keypoints[pair[1], 1]) // 4), (255, 0, 0), thickness=1)
    # print(keypoints2d)
    plt.cla()
    plt.imshow(image)
    plt.draw()
    plt.pause(0.000000000000001)
