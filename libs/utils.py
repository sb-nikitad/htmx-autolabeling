import numpy as np
import cv2
import json
import os


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
            scale[i] = scale[i]

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

def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

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


def generate_2d_json(skeletons2d, bbox2d, video_info, fps=120):
    json_2d = []
    starting_idx = video_info['frames_rate'] * video_info['ignore_seconds']
    for i in range(skeletons2d.shape[0]):
        dic = {'boundingBox': list(bbox2d[i, :]),
               'fps': fps,
               'frameId': starting_idx + i,
               'imageSize': {"x": video_info['width'], "y": video_info['height']},
               'joints': {}, 'score': np.mean(skeletons2d[i, :, 2])}
        for j in range(skeletons2d.shape[1]):
            dic['joints'][str(j)] = {'x': skeletons2d[i, j, 0],
                                     'y': skeletons2d[i, j, 1],
                                     'z': skeletons2d[i, j, 2]}
        json_2d.append(dic)

    return json_2d

    # with open(json_2d_path, 'w', encoding='utf-8') as f:
    #     for i in range(len(json_2d)):
    #         json.dump(json_2d[i], f)
    #         f.write('\n')


def generate_3d_json(skeletons3d, video_info, fps=120):
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

    return json_3d

def generate_ai_json(path3d):
    json3d = json.load(open(path3d))
    # json3d = json.loads(json3d)
    # json3d = json.dump(json3d, open('json3d.json', 'w'))
    # print(len(json3d))
    # print(json.dumps(json3d[:5], separators=(',', ':')))
    # with open('test.sh', 'w') as f:
    #     f.write('java -jar trackers-1.0-SNAPSHOT-all.jar GENERATE_TRACKER %s tracker_output.json' % json.dumps(json3d, separators=(',', ':'))
    os.system('java -jar ../trackers-1.0-SNAPSHOT-all.jar GENERATE_TRACKER %s tracker_output.json' % json.dumps(json3d[:5], separators=(',', ':')))
    jsonai = json.load(open('tracker_output.json'))
    return jsonai

if __name__ == "__main__":
    generate_ai_json('/home/gpuuser/HumanPose/desktop-pipeline/json_3d.json')
