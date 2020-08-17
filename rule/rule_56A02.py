import numpy as np
import os
import itertools
import pandas as pd
import json
import pickle
import cv2
import random

PRIO_WEIGHT = 0.6
PRIO_LIST = ['MKL1', 'MPLO', 'MPTO', 'MRM3', 'AZ02', 'MBU1', 'MLS1', 'MPL1', 'WBU1', 'IRP1', 'MDR1', 'MRM2',
             'PFBA', 'AZ06', 'DAL1', 'IDT1', 'IRMR', 'ISC3', 'TPT1', 'IFB1', 'MDF1', 'FALSE']
FALSE_NAME = 'FALSE'


def show_and_save_images(img_path, img_name, bboxes, codes, out_dir=None, random_color=False):
    img = cv2.imread(img_path)
    cor_l = 0
    color = (0, 0, 255)
    for i, bbox in enumerate(bboxes):
        if random_color:
            color = tuple([random.randint(125, 255) for _ in range(3)])
        bbox = np.array(bbox)
        bbox_int = bbox[:4].astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        code = codes[i]
        label_txt = code + ': ' + str(round(bbox[4], 2))
        cv2.rectangle(img, left_top, right_bottom, color, 1)
        cor_l += 30
        cv2.putText(img, label_txt, (0, cor_l),
                    cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
        cv2.line(img, (0, cor_l), (bbox_int[0], bbox_int[1]), color)

    if out_dir is not None:
        cv2.imwrite(os.path.join(out_dir, img_name), img)

    return img


def prio_check(prio_lst, code_lst, score_lst=None):
    idx_lst = []
    for code in code_lst:
        # assert code in prio_lst, '{} should be in priority file'.format(code)
        if code not in prio_lst:
            print('{} not in priority list, using best score result'.format(code))
            final_code = code_lst[np.argmax(score_lst)]
            return final_code
        idx = prio_lst.index(code)
        idx_lst.append(idx)
    final_code = prio_lst[min(idx_lst)]
    return final_code


def filter_code(df, code, thr, replace=None):
    check_code = df[(df['category'] == code) & (df['score'] <= thr)]
    if replace is None:
        df = df.drop(index=check_code.index)
    else:
        df.loc[check_code.index, 'category'] = replace
    return df


def judge_size(df, code, code1, code2, size, thr):
    if size >= thr:
        df = filter_code(df, code, 1.0, code2)
    else:
        df = filter_code(df, code, 1.0, code1)
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
                          'size': (coord[2] - coord[0]) * (coord[3] - coord[1])})

    return json_dict


def default_rule(det_lst, img_path, img_name, config, codes, draw_img=False, **kwargs):
    """

    :param det_lst: list,
    :param img_path: str,
    :param img_name: str,
    :param size: float, size from .gls file
    :param json_config_file: str, config parameters in json format
    :param code_file: str, code file in txt format
    :param draw_img: Boolean,
    :return: main code, bbox, score, image
    """

    # get size
    size = kwargs.get('size', 0)
    sizeX = kwargs.get('sizeX', 0)
    sizeY = kwargs.get('sizeY', 0)
    unit_id = kwargs.get('unit_id', None)

    size = 0 if size is None else size
    sizeX = 0 if sizeX is None else sizeX
    sizeY = 0 if sizeY is None else sizeY
    if unit_id == 'A1AOI800':
        size = max(size, sizeX, sizeY)

    img = cv2.imread(img_path)
    if img is None:
        raise ValueError

    h, w, c = img.shape

    # analyse pkl file to get det result
    json_dict = model_test(det_lst, img_name, codes)
    det_df = pd.DataFrame(json_dict, columns=['name', 'category', 'bbox', 'score', 'bbox_score',
                                              'xmin', 'ymin', 'xmax', 'ymax', 'ctx', 'cty', 'size'])

    det_df = filter_code(det_df, 'MPL', 1.0, 'MPLO')
    det_df = filter_code(det_df, 'IRMR', 0.6)
    det_df = filter_code(det_df, 'IDT1', 0.5)
    det_df = filter_code(det_df, 'DRMT', 0.7)
    det_df = filter_code(det_df, 'DAL1', 0.4)
    det_df = filter_code(det_df, 'ISC3', 0.4)
    det_df = filter_code(det_df, 'MDF1', 0.3)
    det_df = filter_code(det_df, 'MDR1', 0.4)
    det_df = filter_code(det_df, 'WBU1', 0.3)
    det_df = filter_code(det_df, 'PFBA', 0.3)
    det_df = filter_code(det_df, 'TPT1', 0.1)
    det_df = filter_code(det_df, 'MPL1', 0.2)
    det_df = filter_code(det_df, 'MPLO', 0.2)
    det_df = filter_code(det_df, 'MRM2', 0.2)
    det_df = filter_code(det_df, 'MRM3', 0.2)
    det_df = filter_code(det_df, 'MLS1', 0.2)
    det_df = filter_code(det_df, 'MKL1', 0.2)

    # det_df = judge_size(det_df, 'TPT1', 'TPT1', 'MPTO', size, 100)
    # det_df = judge_size(det_df, 'MPTO', 'TPT1', 'MPTO', size, 100)
    # det_df = judge_size(det_df, 'MRM2', 'MRM2', 'MRM3', size, 100)
    # det_df = judge_size(det_df, 'MRM3', 'MRM2', 'MRM3', size, 100)

    for idx, row in det_df.iterrows():
        if row['category'] == 'TPT1':
            if size >= 100:
                det_df.loc[idx, 'category'] = 'MPTO'

        elif row['category'] == 'MPTO':
            if size < 100 and row['size'] / (h * w) < 0.3:
                det_df.loc[idx, 'category'] = 'TPT1'

        elif row['category'] == 'MRM2':
            if size >= 100:
                det_df.loc[idx, 'category'] = 'MRM3'

        elif row['category'] == 'MRM3':
            if size < 100 and row['size'] / (h * w) < 0.3:
                det_df.loc[idx, 'category'] = 'MRM2'

        elif row['category'] == 'MPL1':
            if (row['xmax'] - row['xmin']) / w >= 0.8 or (row['ymax'] - row['ymin']) / h >= 0.8:
                det_df.loc[idx, 'category'] = 'MPLO'

        elif row['category'] == 'MPLO':
            if (row['xmax'] - row['xmin']) / w < 0.7 and (row['ymax'] - row['ymin']) / h < 0.7:
                det_df.loc[idx, 'category'] = 'MPL1'


    code, bbox, score = prio_judge(det_df, prio_weight=PRIO_WEIGHT, prio_lst=PRIO_LIST, false_name=FALSE_NAME)

    # draw images
    if draw_img:
        img = show_and_save_images(img_path,
                                   img_name,
                                   det_df.bbox_score.values,
                                   det_df.category.values)
    else:
        img = None

    return code, bbox, score, img


def prio_judge(det_df, **kwargs):
    # PRIO CHECK
    if len(det_df) != 0:
        Max_conf = det_df['score'].values.max()
        prio_thr = Max_conf * kwargs.get('prio_weight')
        det_df = det_df[det_df['score'] >= prio_thr]

        if kwargs.get('prio_lst') is None or kwargs.get('prio_lst') == []:
            final_code = det_df.loc[det_df['score'].argmax(), 'category']
        else:
            final_code = prio_check(kwargs.get('prio_lst'), list(det_df['category']), list(det_df['score']))
        best_score = det_df.loc[det_df['category'] == final_code, 'score'].max()
        best_idx = det_df.loc[det_df['category'] == final_code, 'score'].argmax()
        best_bbox = det_df.loc[best_idx, 'bbox']
    else:
        final_code = kwargs.get('false_name')
        best_bbox = []
        best_score = 1

    return final_code, best_bbox, best_score


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
