import numpy as np
import os
import itertools
import pandas as pd
import json
import pickle
import cv2
import numpy as np


def bboxes_iou(boxes1, boxes2):
    """
    boxes: [xmin, ymin, xmax, ymax, score, class] format coordinates.
    """
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:4], boxes2[..., 2:4])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    ious = np.maximum(1.0 * inter_area / union_area, 0.0)

    return ious


def check_contact(boxes1, boxes2, thr=0):
    """
    boxes: [xmin, ymin, xmax, ymax] format coordinates.
    check if boxes1 contact boxes2
    """
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:4], boxes2[..., 2:4])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    if inter_area > thr:
        return True
    else:
        return False


def show_and_save_images(img_path, img_name, bboxes, codes, out_dir=None):
    img = cv2.imread(os.path.join(img_path, img_name))
    for i, bbox in enumerate(bboxes):
        bbox = np.array(bbox)
        bbox_int = bbox[:4].astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        code = codes[i]
        label_txt = code + ': ' + str(round(bbox[4], 2))
        cv2.rectangle(img, left_top, right_bottom, (0, 0, 255), 1)
        cv2.putText(img, label_txt, (bbox_int[0], max(bbox_int[1] - 2, 0)),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    if out_dir is not None:
        cv2.imwrite(os.path.join(out_dir, img_name), img)

    return img


def prio_check(prio_lst, code_lst):
    idx_lst = []
    for code in code_lst:
        assert code in prio_lst, '{} should be in priority file'.format(code)
        idx = prio_lst.index(code)
        idx_lst.append(idx)
    final_code = prio_lst[min(idx_lst)]
    return final_code


def filter_code(df, code, thr, replace=None):
    check_code = df[(df['category'] == code) & (df['score'] < thr)]
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
    pattern = img_name.split('_')[1][0]

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
        json_dict.append({'name': img_name, 'category': category, 'bbox': coord, 'score': conf, 'bbox_score': bbox[:5], 'pattern': pattern})

    return json_dict


def concat_rule(det_rets, img_names, config, codes, **kwargs):
    total_code = []
    total_bbox = []
    total_score = []
    for ret, img_name in zip(*(det_rets, img_names)):
        final_code, best_bbox, best_score, _ = default_rule(ret, None, img_name, config, codes, False, **kwargs)
        total_code.extend(final_code)
        total_bbox.extend(best_bbox)
        total_score.extend(best_score)
    if 'V04' in total_code:
        for i, code in enumerate(total_code):
            if code in ('V01', 'V03', 'V07'):
                total_code.pop(i)
                total_score.pop(i)
                total_bbox.pop(i)
    if 'V01' in total_code:
        for i, code in enumerate(total_code):
            if code in ('V03', 'V07'):
                total_code.pop(i)
                total_bbox.pop(i)
                total_score.pop(i)
    if 'M07' in total_code:
        for i, code in enumerate(total_code):
            if code == 'M07-64':
                total_code.pop(i)
                total_bbox.pop(i)
                total_score.pop(i)
    return total_code, total_bbox, total_score


def default_rule(det_lst, img_path, img_name, config, codes, draw_img=False, **kwargs):
    """
    postprocessing rule
    """

    # get product id
    product = kwargs.get('product', '')

    # convert list result to dict
    json_dict = model_test(det_lst, img_name, codes)
    det_df = pd.DataFrame(json_dict, columns=['name', 'category', 'bbox', 'score', 'bbox_score'])

    # change other name using threshold
    det_df.loc[det_df['score'] < config['other_thr'], 'category'] = config['other_name']

    # filter pattern for r, g, b
    for k, v in config['filtering_thr'].items():
        det_df = filter_code(det_df, k, v)

    # filter M97
    chip = img_name.split('_')[0]
    position = chip[-4:]
    if (position[:2] not in ('01', '05')) and (position[2:] not in ('01', '18')):
        det_df = filter_code(det_df, 'M97', 1.1)

    # filter V01
    if 'V04' in det_df['category']:
        det_df = filter_code(det_df, 'V01', 1.1)
        det_df = filter_code(det_df, 'V03', 1.1)
        det_df = filter_code(det_df, 'V07', 1.1)
    if 'V01' in det_df['category']:
        det_df = filter_code(det_df, 'V03', 1.1)
        det_df = filter_code(det_df, 'V07', 1.1)


    # judge C08
    if ('639' in product) and ('notch' in det_df['category'].values):
        notch_bbox = det_df.loc[det_df['category'] == 'notch', 'bbox'].values[0]
        for idx in det_df.index:
            cate = det_df.loc[idx, 'category']
            if cate in ('L01', 'L02', 'L09', 'L10'):
                bbox = det_df.loc[idx, 'bbox']
                if check_contact(bbox, notch_bbox):
                    det_df.loc[idx, 'category'] = 'C08'

    # judge L17
    cates = det_df['category'].values
    idx_lst = []
    if ('L01' in cates or 'L02' in cates) and ('L09' in cates or 'L10' in cates):
        for idx1 in det_df[(det_df['category'] == 'L01') | (det_df['category'] == 'L02')].index:
            for idx2 in det_df[(det_df['category'] == 'L09') | (det_df['category'] == 'L10')].index:
                bbox1 = det_df.loc[idx1, 'bbox']
                bbox2 = det_df.loc[idx2, 'bbox']
                if check_contact(bbox1, bbox2):
                    idx_lst.append(idx1)
                    idx_lst.append(idx2)
    idx_lst = list(set(idx_lst))
    det_df.loc[idx_lst, 'category'] = 'L17'

    # delect notch
    det_df = filter_code(det_df, 'notch', 1.1)

    # ET judge
    final_code = list(det_df['category'].values)
    best_bbox = list(det_df['bbox'].values)
    best_score = list(det_df['score'].values)

    # draw images
    img = None

    return final_code, best_bbox, best_score, img


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    config_file = r'C:\Users\huker\Desktop\G6_21101-V1.0\G6_21101-V1.0.json'
    code_file = r'C:\Users\huker\Desktop\G6_21101-V1.0\G6_21101-V1.0.txt'
    img_path = r'C:\Users\huker\Desktop\G6_21101-V1.0'
    img_name = r'w97kp1222a0216_-142801_-511998_before.jpg'
    result_pkl = r'C:\Users\huker\Desktop\G6_21101-V1.0\21101_testimg.pkl'

    with open(result_pkl, 'rb') as f:
        result_lst = pickle.load(f)

    main_code, bbox, score, img = default_rule(result_lst, img_path, img_name, config_file, code_file)
    print(main_code, bbox, score)
    b, g, r = cv2.split(img)
    img2 = cv2.merge([r, g, b])
    plt.imshow(img2)
    plt.show()
