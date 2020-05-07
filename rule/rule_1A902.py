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
    img = cv2.imread(os.path.join(img_path))
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
        json_dict.append({'name': img_name, 'category': category, 'bbox': coord, 'score': conf, 'bbox_score': bbox[:5]})

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
    # open config file
    # with open(json_config_file) as f:
    #     config = json.load(f)
    # with open(code_file) as fp:
    #     codes = fp.read().splitlines()


    # get size
    size = kwargs.get('size', None)

    # get product_id
    product_id = kwargs.get('product_id', None)


    pd_flag = img_name.split('.')[0].split('_')[3]
    if pd_flag in ['pd', 'pad', 'cm', 'ms']:
        pad_flag = 1
    else:
        pad_flag = 0

    # analyse pkl file to get det result
    json_dict = model_test(det_lst, img_name, codes)
    det_df = pd.DataFrame(json_dict, columns=['name', 'category', 'bbox', 'score', 'bbox_score'])


    # prio parameters
    prio_weight = config['prio_weight']
    prio_lst = config['prio_order']
    if config['false_name'] not in prio_lst:
        prio_lst.append(config['false_name'])
    if config['other_name'] not in prio_lst:
        prio_lst.append(config['other_name'])

    # change other name using threshold
    det_df.loc[det_df['score'] < config['other_thr'], 'category'] = config['other_name']

    category_list = det_df['category'].values


    # judge size
    if size:
        if 'STR05' in category_list or 'STR02' in category_list:
            if size >= 40:
                 det_df.loc[det_df['category'] == 'STR05', 'category'] = 'STR05X'
                 det_df.loc[det_df['category'] == 'STR02', 'category'] = 'STR02X'
            else:
                det_df.loc[det_df['category'] == 'STR05', 'category'] = 'STR05O'
                det_df.loc[det_df['category'] == 'STR02', 'category'] = 'STR02O'

    filter_code(det_df, 'AZ09_H', 0.3, 'AZ09_O')
    filter_code(det_df, 'AZ09_L', 0.3, 'AZ09_O')


    # jauge AZ09 and AZ08
    flag_AZ08 = 0
    flag_AZ09_DET = 0
    flag_AZ09_PHT = 0
    flag_STR03 = 0
    for item in det_df['category'].values:
        if 'AZ09_H' == item:
            flag_AZ09_DET += 1
        if 'AZ09_L' == item:
            flag_AZ09_PHT += 1
        if 'AZ08_N' == item:
            flag_AZ08 += 1
        if 'STR03' == item:
            flag_STR03 += 1


    if flag_AZ09_DET >= 2:
        det_df.loc[det_df['category'] == 'AZ09_H', 'category'] = 'ILS03_DET'
    else:
        det_df.loc[det_df['category'] == 'AZ09_H', 'category'] = 'ILS02_DET'
    if flag_AZ09_PHT >= 2:
        det_df.loc[det_df['category'] == 'AZ09_L', 'category'] = 'ILS03_PHT'
    else:
        det_df.loc[det_df['category'] == 'AZ09_L', 'category'] = 'ILS02_PHT'

    if flag_AZ08 >= 2:
        det_df.loc[det_df['category'] == 'AZ08_N', 'category'] = 'AZ08_T'
    else:
        det_df.loc[det_df['category'] == 'AZ08_N', 'category'] = 'AZ08_O'

    if flag_STR03 >= 5:
        det_df.loc[det_df['category'] == 'STR03', 'category'] = 'AZ15'



    if product_id:
        if 'STR10_PHT' in category_list or 'STR10_DET' in category_list:
            if 'L1' == product_id[:2]:
                det_df.loc[det_df['category'] == 'STR10_PHT', 'category'] = 'STR10'
            elif 'L3' == product_id[:2]:
                det_df.loc[det_df['category'] == 'STR10_DET', 'category'] = 'STR10'




    # prio_check
    if len(det_df) != 0:
        Max_conf = det_df['score'].values.max()
        prio_thr = Max_conf * prio_weight
        filtered_final = det_df[det_df['score'] >= prio_thr]

        final_code = prio_check(prio_lst, list(filtered_final['category']))
        best_score = filtered_final.loc[filtered_final['category'] == final_code, 'score'].max()
        best_idx = filtered_final.loc[filtered_final['category'] == final_code, 'score'].argmax()
        best_bbox = filtered_final.loc[best_idx, 'bbox']
    # elif pad_flag:
    #    final_code = 'PAD'
    #    best_bbox = []
    #    best_score = 1
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






    # if final_code in ['AZ06_PAD', 'AZ10', 'AZ12', 'AZ15', 'AZ99', 'Defocus', 'PAD', 'STR01', 'STR03', 'CVD05','AZ11','ESD']:
    #     final_code = '0'
    # elif final_code in ['AZ08_X', 'AZ08_T', 'AZ08_N', 'AZ08_O']:
    #     final_code = 'AZ08'
    # elif final_code in ['AZ09_T', 'AZ09_H', 'AZ09_L', 'AZ09_O', 'ILS03_DET', 'ILS03_PHT']:
    #     final_code = 'AZ09'
    # elif final_code in ['STR10_DET', 'STR10_PHT']:
    #     final_code = 'STR10'

    #flag = juage_code(final_code, best_score)



    return final_code, best_bbox, best_score, img


def juage_code(final_code, best_score):
    df_code = pd.read_csv(r'D:\final_1A\1A902\prepare\filter_code.csv')
    score = df_code.loc[df_code['code'] == final_code, 'score'].max()
    print(final_code)
    if best_score > score:
        return 1
    else:
        return 0



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
