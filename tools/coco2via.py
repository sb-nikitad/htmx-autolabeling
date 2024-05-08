import csv
import json
import numpy as np
import time
from pycocotools import coco

def coco2csv(jsonpath, csvpath):
    with open(csvpath, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        coco_set = coco.COCO(jsonpath)
        row = ["filename","file_size","file_attributes","region_count","region_id","region_shape_attributes","region_attributes"]
        csvwriter.writerow(row)
        image_ids = sorted(coco_set.getImgIds())

        imgs = coco_set.loadImgs(coco_set.getImgIds())

        keypoints_map = {
                "top of head": 0,
                "forehead": 1,
                "nose": 2,
                "left ear": 3,
                "right ear": 4,
                "left AC joint": 5,
                "right AC joint": 6,
                "right rib cage mid": 7,
                "right rib cage low": 8,
                "left top of pelvis": 9,
                "left hip": 10,
                "left GT": 11,
                "right hip": 12,
                "right GT": 13,
                "left shoulder": 14,
                "left elbow": 15,
                "left wrist": 16,
                "right shoulder": 17,
                "right elbow": 18,
                "right wrist": 19,
                "left index knuckle": 20,
                "left pinky knuckle": 21,
                "right index knuckle": 22,
                "right pinky knuckle": 23,
                "left knee": 24,
                "left ankle": 25,
                "right knee": 26,
                "right ankle": 27,
                "left heel": 28,
                "left mid toe": 29,
                "left foot outside": 30,
                "right heel": 31,
                "right mid toe": 32,
                "right foot outside": 33,
                "club mid hands": 34,
                "club top of handle": 35,
                "club hosel": 36,
                "club toe top": 37}
        keypoints_map = sorted(keypoints_map.items(), key=lambda x: x[1]) 

        for j, image_id in enumerate(image_ids):
            annIds = coco_set.getAnnIds(imgIds=image_id, catIds=[1], iscrowd=None)
            anns = coco_set.loadAnns(annIds)[0]
            keypoints = np.array(anns['keypoints'])
            
            keypoint_num = 38
            count = 0
            filename = f'{imgs[j]["video_name"]}_{image_id:04}.jpg'
            file_size = imgs[j]['file_size']
            for i in range(38):
                if float(keypoints[i * 3 + 2]) > 0.2:
                    
                    row = [filename, file_size, '{}', keypoint_num, count, "{\"name\":\"point\",\"cx\":%d,\"cy\":%d,\"conf\":%.2f}" % (int(float(keypoints[i * 3])), int(float(keypoints[i * 3 + 1])), float(keypoints[i * 3 + 2])), \
                        "{\"type\":\"%s\"}" % keypoints_map[i][0]]
                    csvwriter.writerow(row)
                    count += 1
                    
def coco2csv_baseball(jsonpath, csvpath):
    with open(csvpath, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        coco_set = coco.COCO(jsonpath)
        row = ["filename","file_size","file_attributes","region_count","region_id","region_shape_attributes","region_attributes"]
        csvwriter.writerow(row)
        image_ids = sorted(coco_set.getImgIds())

        imgs = coco_set.loadImgs(coco_set.getImgIds())

        keypoints_map = {
                "top of head": 0,
                "forehead": 1,
                # "nose": 2,
                "left ear": 2,
                "right ear": 3,
                "left AC joint": 4,
                "right AC joint": 5,
                "right rib cage mid": 6,
                "right rib cage low": 7,
                "left top of pelvis": 8,
                "left hip": 9,
                "left GT": 10,
                "right hip": 11,
                "right GT": 12,
                "left shoulder": 13,
                "left elbow": 14,
                "left wrist": 15,
                "right shoulder": 16,
                "right elbow": 17,
                "right wrist": 18,
                "left index knuckle": 19,
                "left pinky knuckle": 20,
                "right index knuckle": 21,
                "right pinky knuckle": 22,
                "left knee": 23,
                "left ankle": 24,
                "right knee": 25,
                "right ankle": 26,
                "left heel": 27,
                "left mid toe": 28,
                "left foot outside": 29,
                "right heel": 30,
                "right mid toe": 31,
                "right foot outside": 32,
                "bat mid hands": 33,
                "bat top of handle": 34,
                "bat end": 35}
                # "club toe top": 37}
        keypoints_map = sorted(keypoints_map.items(), key=lambda x: x[1]) 

        for j, image_id in enumerate(image_ids):
            annIds = coco_set.getAnnIds(imgIds=image_id, catIds=[1], iscrowd=None)
            anns = coco_set.loadAnns(annIds)[0]
            keypoints = np.array(anns['keypoints'])
            
            keypoint_num = 36
            count = 0
            filename = f'{imgs[j]["video_name"]}_{image_id:04}.jpg'
            file_size = imgs[j]['file_size']
            for i in range(36):
                if float(keypoints[i * 3 + 2]) > 0.2:
                    
                    row = [filename, file_size, '{}', keypoint_num, count, "{\"name\":\"point\",\"cx\":%d,\"cy\":%d,\"conf\":%.2f}" % (int(float(keypoints[i * 3])), int(float(keypoints[i * 3 + 1])), float(keypoints[i * 3 + 2])), \
                        "{\"type\":\"%s\"}" % keypoints_map[i][0]]
                    csvwriter.writerow(row)
                    count += 1