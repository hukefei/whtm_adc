#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import os, glob
import json
import shutil
import numpy as np
from utils.Code_dictionary import CodeDictionary
import pickle
import random
from utils.utils import bboxes_iou, nms


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




def model_test(pkl_file,
               score_thr=0.1,
               NMS=False,
               nms_thr=0.9):
    with open(pkl_file, 'rb') as f:
        results = pickle.load(f)

    output_bboxes = []
    json_dict = []
    for i, result in enumerate(results):  # loop for images
        img_path = imgs[i].replace('\\', '/')
        img_name = img_path.split('/')[-1]
        print(i, img_name)

        total_bbox = []
        for id, boxes in enumerate(result):  # loop for categories
            category_id = id + 1
            if len(boxes) != 0:
                for box in boxes:  # loop for bbox
                    conf = box[4]
                    if conf > score_thr:
                        total_bbox.append(list(box) + [category_id])

        bboxes = np.array(total_bbox)
        if NMS:
            best_bboxes = nms(bboxes, nms_thr)
        else:
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
        cv2.putText(img, label_txt, (bbox_int[0], max(bbox_int[1] - 2, 0)),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    if out_dir is not None:
        cv2.imwrite(os.path.join(out_dir, img_name), img)


if __name__ == '__main__':
    imgs = glob.glob(r'/data/sdv1/whtm/data/1GE02/1GE02_train_test_data/test/*.jpg')
    pkl_file = r'/data/sdv1/whtm/result/1GE02/1GE02_v1_at.pkl'
    output_bboxes, json_dict = model_test(pkl_file,
                                          score_thr=0.05,
                                          NMS=False)
    with open(r'/data/sdv1/whtm/result/1GE02/1GE02_v1_at.json', 'w') as f:
        json.dump(json_dict, f, indent=4)

    code_file = r'/data/sdv1/whtm/document/1GE02.xlsx'
    code = CodeDictionary(code_file)

    out_dir = r'1GEO2_AT'
    if out_dir is not None:
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir)
        for i in range(len(imgs)):
            img_path = imgs[i]
            bboxes = output_bboxes[i]
            show_and_save_images(img_path, bboxes, code, out_dir)
