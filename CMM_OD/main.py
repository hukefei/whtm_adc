import cv2
import os
import numpy as np
import glob
from mmdet.apis import inference_detector, init_detector
import tqdm
import itertools
import pandas as pd

PRIO_ORDER = ['CP03', 'CE03', 'CB02', 'FALSE']
PRIO_WEIGHT = 0.6
THRESHOLD = 0.3


def predict(img, **kwargs):
    assert isinstance(img, str)
    model = kwargs.get('model')
    classes = kwargs.get('classes')

    image_name = img.replace('\\', '/').split('/')[-1]
    image_color = cv2.imread(img, 1)
    image_color = cv2.resize(image_color, (512, 512))

    mmdet_result = inference_detector(model, image_color)
    mmdet_result = all_classes_nms(mmdet_result, 0.7)

    draw_bboxes(image_color, mmdet_result, classes, thr=THRESHOLD)
    code, bbox, score, image = judge_code(mmdet_result, image_color, image_name, classes)

    return code, bbox, score, image


def init_model(config, checkpoint, **kwargs):
    model = init_detector(config, checkpoint, **kwargs)
    return model


def process(path):
    files = os.listdir(path)
    assert 'config.py' in files, 'mmdet config file shoule be included'
    assert 'model.pth' in files, 'checkpoint file should be included'
    assert 'classes.txt' in files, 'classes file should be included'
    config = os.path.join(path, 'config.py')
    checkpoint = os.path.join(path, 'model.pth')
    model = init_model(config, checkpoint)
    with open(os.path.join(path, 'classes.txt'), 'r') as f:
        classes = f.read().splitlines()

    return {'model': model, 'classes': classes}


def filter_code(df, code, thr=1, replace=None):
    if isinstance(code, list):
        for c in code:
            df = filter_code(df, c, thr, replace)
    elif code == 'all':
        df = df[df['score'] > thr]
    else:
        check_code = df[(df['category'] == code) & (df['score'] <= thr)]
        if replace is None:
            df = df.drop(index=check_code.index)
        else:
            df.loc[check_code.index, 'category'] = replace
    return df


def model_test(result,
               img_name,
               codes,
               score_thr=0.05):
    output_bboxes = []
    json_dict = []

    total_bbox = []
    for id, boxes in enumerate(result):  # loop for categories
        category_id = id + 1
        if len(boxes) != 0:
            for box in boxes:  # loop for bbox
                conf = box[4]
                if conf > score_thr:
                    total_bbox.append(list(box) + [category_id])

    bboxes = np.array(total_bbox)
    best_bboxes = bboxes
    output_bboxes.append(best_bboxes)
    for bbox in best_bboxes:
        coord = [round(i, 2) for i in bbox[:4]]
        conf, category = bbox[4], codes[int(bbox[5]) - 1]
        json_dict.append({'name': img_name, 'category': category, 'bbox': coord, 'score': conf, 'bbox_score': bbox[:5],
                          'xmin': coord[0], 'ymin': coord[1], 'xmax': coord[2], 'ymax': coord[3],
                          'ctx': (coord[0] + coord[2]) / 2, 'cty': (coord[1] + coord[3]) / 2,
                          'size': (coord[2] - coord[0]) * (coord[3] - coord[1]), 'remark': {}})

    return json_dict


def judge_code(mmdet_result, image_color, image_name, classes):
    d_ = model_test(mmdet_result, image_name, classes, score_thr=THRESHOLD)
    det_df = pd.DataFrame(d_, columns=['name', 'category', 'bbox', 'score', 'bbox_score',
                                       'xmin', 'ymin', 'xmax', 'ymax', 'ctx', 'cty', 'size', 'remark'])
    det_df = det_df.sort_values(by='score', ascending=False)

    # final judge
    code, bbox, score = final_judge(det_df, PRIO_ORDER, prio_weight=PRIO_WEIGHT)

    return code, bbox, score, image_color


def cpu_nms(dets, thresh, mode='iou'):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    order = dets[:, 4].argsort()[::-1]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        if mode == 'iou':
            over = (w * h) / (area[i] + area[order[1:]] - w * h)
        elif mode == 'iof':
            over = (w * h) / (area[i])
        index = np.where(over <= thresh)[0]
        order = order[index + 1]
    return keep


def all_classes_nms(mmdet_result, thr=0.8, mode='iou'):
    num = len(mmdet_result)
    bboxes_ = []
    labels_ = []
    for i, bboxes in enumerate(mmdet_result):
        for bbox in bboxes:
            bboxes_.append(bbox)
            labels_.append(i)
    bboxes_ = np.array(bboxes_)
    labels_ = np.array(labels_)
    if len(bboxes) == 0:
        return mmdet_result
    keep = cpu_nms(bboxes_, thr, mode)
    keep_bboxes = bboxes_[keep]
    keep_labels = labels_[keep]
    final_ = []
    for _ in range(num):
        final_.append([])
    for bbox, label in zip(keep_bboxes, keep_labels):
        final_[label].append(bbox)
    return final_


def final_judge(det_df, prio_order=None, prio_weight=0.5):
    if len(det_df) != 0:
        Max_conf = det_df['score'].values.max()
        prio_thr = Max_conf * prio_weight
        filtered_final = det_df[det_df['score'] >= prio_thr]

        final_code = prio_check(prio_order, list(filtered_final['category']))
        if final_code == 'NOT_INPLEMENTED':
            return final_code, [], 0
        best_score = filtered_final.loc[filtered_final['category'] == final_code, 'score'].max()
        best_idx = int(filtered_final.loc[filtered_final['category'] == final_code, 'score'].idxmax())
        best_bbox = filtered_final.loc[best_idx, 'bbox']
    else:
        final_code = 'FALSE'
        best_bbox = []
        best_score = 0
    return final_code, best_bbox, best_score


def prio_check(prio_lst, code_lst):
    idx_lst = []
    for code in code_lst:
        if code not in prio_lst:
            # print('{} not in priority list'.format(code))
            continue
        idx = prio_lst.index(code)
        idx_lst.append(idx)
    if idx_lst == []:
        return 'NOT_INPLEMENTED'
    final_code = prio_lst[min(idx_lst)]
    return final_code


def draw_bboxes(img, mmdet_results, codes, thr=0):
    for i, bboxes in enumerate(mmdet_results):
        for bbox in bboxes:
            bbox = np.array(bbox)
            bbox_int = bbox[:4].astype(int)
            left_top = (bbox_int[0], bbox_int[1])
            right_bottom = (bbox_int[2], bbox_int[3])
            if bbox[4] < thr:
                continue
            code = codes[i]
            label_txt = code + ': ' + str(round(bbox[4], 2))
            cv2.rectangle(img, left_top, right_bottom, (0, 0, 255), 1)
            cv2.putText(img, label_txt, (bbox_int[0], max(bbox_int[1] - 2, 0)),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    return img
