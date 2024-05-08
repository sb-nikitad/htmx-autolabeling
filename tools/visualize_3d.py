import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from pycocotools.coco import COCO
import numpy as np
import json
import cv2

# Single video test
# root = "/Users/haoruiji/Downloads"
# name = 'test_3d.json'

root = "./data/tracker_accuracy_test/AMM Data Seattle Feb 2022/results/140K-2d-120fps_attn_vp3d_1000K-3d-33/3d"
name = 'fleera7i1.json'

json_path = os.path.join(root, name)
json_file = json.load(open(json_path))

output_path = "./test_1/"
if not os.path.exists(output_path):
    os.makedirs(output_path)

for fid, frame in enumerate(json_file):
    if fid % 2 != 0:
        continue
    print(fid)

    keypoints = np.zeros((38, 3))
    for i in range(38):
        keypoints[i] = np.array([frame['joints'][str(i)]['x'], frame['joints'][str(i)]['y'], frame['joints'][str(i)]['z']])

    keypoints /= 1000

    visualize_idx = [0, 5, 6, 15, 16, 18, 19, 10, 12, 24, 25, 26, 27, 34, 36, 20, 21, 22, 23]
    skeletons = ((5, 6), (16, 20), (16, 21), (20, 21), (19, 22), (19, 23), (22, 23), (10, 12), (5, 10),
                 (6, 12), (10, 24), (24, 25), (12, 26), (26, 27), (34, 36))
    visualize_kpts = keypoints[visualize_idx, :]

    # 3D
    plt.cla()
    plt.figure(figsize=(16, 16))
    ax = plt.axes(projection='3d')
    # ax.view_init(elev=0, azim=0)

    max_range = np.array([2, 2, 2]).max() / 2.0
    mid_x = 0.0
    mid_y = 3.0
    mid_z = 0.0
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.scatter(visualize_kpts[:, 0], visualize_kpts[:, 2], -visualize_kpts[:, 1], cmap='Blues', s=20)

    red_index = [len(skeletons) - 1, 3, 6]
    for index, skeleton in enumerate(skeletons):
        if index not in red_index :
            ax.plot([keypoints[skeleton[0], 0], keypoints[skeleton[1], 0]],
                    [keypoints[skeleton[0], 2], keypoints[skeleton[1], 2]],
                    [-keypoints[skeleton[0], 1], -keypoints[skeleton[1], 1]], 'b')
        else:
            ax.plot([keypoints[skeleton[0], 0], keypoints[skeleton[1], 0]],
                    [keypoints[skeleton[0], 2], keypoints[skeleton[1], 2]],
                    [-keypoints[skeleton[0], 1], -keypoints[skeleton[1], 1]], 'r')

    # # Proj 2D
    # plt.cla()
    # plt.figure(figsize=(16, 16))
    # ax = plt.axes()
    # max_range = np.array([2, 2, 2]).max() / 2.0
    # mid_x = 0.25
    # # mid_y = 0
    # mid_z = 0
    # ax.set_xlim(mid_x - max_range, mid_x + max_range)
    # ax.set_ylim(mid_z - max_range, mid_z + max_range)
    # # ax.set_zlim(mid_z - max_range, mid_z + max_range)
    #
    # center_pelvis = (keypoints[10, :] + keypoints[12, :]) / 2
    # neck = (keypoints[5, :] + keypoints[6, :]) / 2
    # spine_center = (center_pelvis + neck) / 2
    #
    # # ax.scatter(visualize_kpts[:, 0], -visualize_kpts[:, 1], cmap='Blues')
    # ax.scatter(center_pelvis[0], -center_pelvis[1], cmap='Blues')
    # ax.scatter(neck[0], -neck[1], cmap='Blues')
    # ax.scatter(spine_center[0], -spine_center[1], cmap='Blues')
    #
    # for index, skeleton in enumerate(skeletons):
    #     if index != len(skeletons) - 1:
    #         ax.plot([keypoints[skeleton[0], 0], keypoints[skeleton[1], 0]],
    #                 # [keypoints[skeleton[0], 2], keypoints[skeleton[1], 2]],
    #                 [-keypoints[skeleton[0], 1], -keypoints[skeleton[1], 1]], 'b')
    #     # else:
    #     #     ax.plot([keypoints[skeleton[0], 0], keypoints[skeleton[1], 0]],
    #     #             # [keypoints[skeleton[0], 2], keypoints[skeleton[1], 2]],
    #     #             [-keypoints[skeleton[0], 1], -keypoints[skeleton[1], 1]], 'r')
    #
    plt.savefig(os.path.join(output_path, "%06d.jpg" % (fid // 2)))



