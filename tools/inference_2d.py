import os
import copy
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2

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


def preProcessImage(srcImage, center, scale):
    image_size = [192, 256]
    r = 0

    affineTransMat = get_affine_transform(center, scale, r, image_size)

    affineTransImg = cv2.warpAffine(srcImage,
                                    affineTransMat,
                                    (192, 256),
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

if __name__ == "__main__":
    test_image = plt.imread('059783.jpg')
    model_2d_path = '/home/gpuuser/HumanPose/ssl/golf-2d-pose-estimation/converted_tflite_model_2kp.tflite'
    bbox = [test_image.shape[1] / 4, 0, test_image.shape[1] / 2, test_image.shape[0]]

    preds, conf = inference_2d_skeleton(test_image, model_2d_path, bbox, flag=0)

    test_image = test_image.copy()
    for i in range(2):
        if 5 < preds[0, i, 0] < test_image.shape[1] - 5 and 5 < preds[0, i, 1] < test_image.shape[0] - 5:
            test_image[int(preds[0, i, 1] - 5):int(preds[0, i, 1] + 5), int(preds[0, i, 0]) - 5:int(preds[0, i, 0] + 5)] = [255, 0, 0]
    plt.imsave('result.jpg', test_image)
