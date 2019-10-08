#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import os, glob
import json
import shutil
import numpy as np
from utils.Code_dictionary import CodeDictionary
import pickle


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return round(float(obj), 2)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def bboxes_iou(boxes1, boxes2):
    """
    boxes: [xmin, ymin, xmax, ymax, score, class] format coordinates.
    """
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    ious = np.maximum(1.0 * inter_area / union_area, 0.0)

    return ious


def nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
    """
    Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
          https://github.com/bharatsingh430/soft-nms
    """
    best_bboxes = []

    while len(bboxes) > 0:
        max_ind = np.argmax(bboxes[:, 4])
        best_bbox = bboxes[max_ind]
        best_bboxes.append(best_bbox)

        bboxes = np.concatenate([bboxes[: max_ind], bboxes[max_ind + 1:]])
        iou = bboxes_iou(best_bbox[np.newaxis, :4], bboxes[:, :4])
        weight = np.ones((len(iou),), dtype=np.float32)

        assert method in ['nms', 'soft-nms']
        if method == 'nms':
            iou_mask = iou > iou_threshold
            weight[iou_mask] = 0.0
        if method == 'soft-nms':
            weight = np.exp(-(1.0 * iou ** 2 / sigma))

        bboxes[:, 4] = bboxes[:, 4] * weight
        score_mask = bboxes[:, 4] > 0.
        bboxes = bboxes[score_mask]

    return best_bboxes

def model_test(pkl_file,
               score_thr=0.1):

    with open(pkl_file, 'rb') as f:
        results = pickle.load(f)

    output_bboxes = []
    json_dict = []
    for i, result in enumerate(results):#loop for images
        img_path = imgs[i].replace('\\', '/')
        img_name = img_path.split('/')[-1]
        print(i, img_name)

        total_bbox = []
        for id, boxes in enumerate(result):#loop for categories
            category_id = id + 1
            if len(boxes) != 0:
                for box in boxes:#loop for bbox
                    conf = box[4]
                    if conf > score_thr:
                        total_bbox.append(list(box) + [category_id])

        bboxes = np.array(total_bbox)
        best_bboxes = bboxes
        output_bboxes.append(best_bboxes)
        for bbox in best_bboxes:
            coord = [round(i, 2) for i in bbox[:4]]
            conf, category_id = bbox[4], int(bbox[5])
            json_dict.append({'name': img_name, 'category': category_id, 'bbox': coord, 'score': conf})

    return output_bboxes, json_dict


def show_and_save_images(img_path, bboxes, code_dict, out_dir=None):
    img = cv2.imread(img_path)
    img_path = img_path.replace('\\', '/')
    img_name = img_path.split('/')[-1]
    for bbox in bboxes:
        bbox_int = bbox[:4].astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        code = code_dict.id2code(int(bbox[5]))
        label_txt = code + ': ' + str(round(bbox[4], 2))
        cv2.rectangle(img, left_top, right_bottom, (0, 0, 255), 1)
        cv2.putText(img, label_txt, (bbox_int[0], max(bbox_int[1] - 2, 0)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    if out_dir is not None:
        cv2.imwrite(os.path.join(out_dir, img_name), img)


if __name__ == '__main__':
    imgs = glob.glob(r'E:\whtm\21101bt_part2_2000\*.jpg')
    pkl_file = r'D:\Project\WHTM\result\21101\bt2\21101_bt2_result_v2.pkl'
    output_bboxes, json_dict = model_test(pkl_file,
                                          score_thr=0.05)
    with open(r'21101_bt2_result_v2.json', 'w') as f:
        json.dump(json_dict, f, indent=4)

    code_file = r'D:\Project\WHTM\code\21101code.xlsx'
    code = CodeDictionary(code_file)

    out_dir = None
    if out_dir is not None:
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir)
        for i in range(len(imgs)):
            img_path = imgs[i]
            bboxes = output_bboxes[i]
            show_and_save_images(img_path, bboxes, code, out_dir)
