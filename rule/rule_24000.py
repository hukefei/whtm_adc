import numpy as np
import os
import itertools
import pandas as pd
import json
import pickle
import cv2


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


def nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
    """
    Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
          https://github.com/bharatsingh430/soft-nms
    """
    best_bboxes_idx = []

    while len(bboxes) > 0:
        max_ind = np.argmax(bboxes[:, 4])
        best_bbox = bboxes[max_ind]
        best_bboxes_idx.append(max_ind)

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

    return best_bboxes_idx


def show_and_save_images(img_path, img_name, bboxes, codes, out_dir=None):
    img = cv2.imread(img_path)
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
                          'ctx': (coord[0]+coord[2])/2, 'cty': (coord[1]+coord[3])/2,
                          'size': (coord[2]-coord[0])*(coord[3]-coord[1])})

    return json_dict


def default_rule(det_lst, img_path, img_name, config, codes, draw_img=False, **kwargs):
    """
    default rule for station 21101
    :param det_lst: detetion result of mmdetection in list format
    :param img_path: image directory
    :param img_name: image name
    :param json_config_file: config file in json format
    :param code_file: categories in txt format
    :return:
    main code
    bbox
    score
    image with bbox and score
    """

    # analyse pkl file to get det result
    json_dict = model_test(det_lst, img_name, codes)
    det_df = pd.DataFrame(json_dict, columns=['name', 'category', 'bbox', 'score', 'bbox_score',
                                              'xmin', 'ymin', 'xmax', 'ymax', 'ctx', 'cty', 'size'])

    #prio parameters
    prio_weight = config['prio_weight']
    prio_lst = config['prio_order']
    if config['false_name'] not in prio_lst:
        prio_lst.append(config['false_name'])
    if config['other_name'] not in prio_lst:
        prio_lst.append(config['other_name'])

    #change other name using threshold
    det_df.loc[det_df['score'] < config['other_thr'], 'category'] = config['other_name']

    # judge RES04
    check = (det_df['category'] == 'RES05') & (det_df['score'] >= 0.2) # adc judge res04
    # check1 = (det_df['category'] == 'RES05') & (det_df['score'] >= 0.3) # transfer to manual path res04
    if sum(check) >= 3:
        det_df.loc[check, 'category'] = 'RES04'
        det_df.loc[check, 'score'] = np.average(det_df.loc[check, 'score'])


    # CHECK COM01/COM17
    if 'COM01' in det_df['category'].values:
        for idx in det_df[det_df['category'] == 'COM01'].index:
            bbox = det_df.loc[idx, 'bbox']
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            if max(width / 5, height / 5) >= 100:
                det_df.loc[idx, 'category'] = 'COM17'

    # CHECK COM22/COM18
    if 'COM22' in det_df['category'].values:
        for idx in det_df[det_df['category'] == 'COM22'].index:
            bbox = det_df.loc[idx, 'bbox']
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            if max(width / 5, height / 5) >= 100:
                det_df.loc[idx, 'category'] = 'COM18'


    # prio_check
    if len(det_df) != 0:
        Max_conf = det_df['score'].values.max()
        prio_thr = Max_conf * prio_weight
        filtered_final = det_df[det_df['score'] >= prio_thr]

        final_code = prio_check(prio_lst, list(filtered_final['category']))
        best_score = filtered_final.loc[filtered_final['category'] == final_code, 'score'].max()
        best_idx = int(filtered_final.loc[filtered_final['category'] == final_code, 'score'].argmax())
        best_bbox = filtered_final.loc[best_idx, 'bbox']
    else:
        final_code = config['false_name']
        best_bbox = []
        best_score = 1


    # draw images
    if draw_img:
        img = show_and_save_images(img_path,
                                   img_name,
                                   det_df.bbox_score.values,
                                   det_df.category.values)
    else:
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
    # open config file
    with open(config_file) as f:
        config = json.load(f)
    with open(code_file) as f:
        codes = f.read().splitlines()

    main_code, bbox, score, img = default_rule(result_lst, img_path, img_name, config_file, code_file)
    print(main_code, bbox, score)
    b, g, r = cv2.split(img)
    img2 = cv2.merge([r, g, b])
    plt.imshow(img2)
    plt.show()
