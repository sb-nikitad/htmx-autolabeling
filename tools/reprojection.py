import matplotlib.pyplot as plt
import os
from pycocotools.coco import COCO
import numpy as np
import json
import cv2

# Single video test
root = './test'
name = 'Swing 2'
json_file = json.load(open(os.path.join(root, '{}_3d.json'.format(name))))
json2d_file = []
with open(os.path.join(root, '{}_2d.json'.format(name)), 'r', encoding='utf-8') as f:
    for line in f.readlines():
        json2d_file.append(json.loads(line))

pchip = None
plankle = None
prankle = None

dhip = []
dlankle = []
drankle = []

all_keypoint = []

intrisic = np.array([[1200, 0, 540], [0, 1200, 960], [0, 0, 1]])

for fid, frame in enumerate(json_file):
    if fid % 2 != 0:
        continue
    print(fid)
    keypoints = np.zeros((39, 3))
    for i in range(39):
        keypoints[i] = np.array([frame['joints'][str(i)]['x'], frame['joints'][str(i)]['y'], frame['joints'][str(i)]['z']])
    if fid == 0:
        depth = keypoints[9, 2] + keypoints[11, 2] / 2
    current_depth = keypoints[9, 2] + keypoints[11, 2] / 2
    keypoints[:, 2] = keypoints[:, 2] * depth / current_depth
    keypoints = intrisic.dot(keypoints.T).T
    keypoints[:, :2] = (keypoints[:, :2].T / keypoints[:, 2]).T
    keypoints2d = np.zeros((39, 2))
    for i in range(39):
        keypoints2d[i] = np.array([json2d_file[fid]['joints'][str(i)]['position']['x'], json2d_file[fid]['joints'][str(i)]['position']['y']])
        # keypoints2d[i] = np.array([json2d_file[fid]['joints'][str(i)]['x'], json2d_file[fid]['joints'][str(i)]['y']])

    image = np.zeros((1920 // 4, 1080 // 4, 3)) + 255
    image = np.uint8(image)
    # bbox = np.array([797, 198.81, 350, 815])
    # cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), (0, 0, 255))
    for i in range(39):
        cv2.circle(image, (int(keypoints[i, 0]) // 4, int(keypoints[i, 1]) // 4), 2, (255, 0, 0), -1)
        cv2.circle(image, (int(keypoints2d[i, 0]) // 4, int(keypoints2d[i, 1]) // 4), 2, (0, 255, 0), -1)

    # feet = [[27, 28], [27, 29], [28, 29], [30, 31], [31, 32], [30, 32]]
    # for pair in feet:
    #     cv2.line(image, (int(keypoints[pair[0], 0]) // 4, int(keypoints[pair[0], 1]) // 4),
    #              (int(keypoints[pair[1], 0]) // 4, int(keypoints[pair[1], 1]) // 4), (255, 0, 0), thickness=1)
    # print(keypoints2d)
    plt.cla()
    plt.imshow(image)
    plt.draw()
    plt.pause(0.000000000000001)
