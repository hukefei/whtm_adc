import cv2
import os
import numpy as np
import glob
from mmdet.apis import inference_detector, init_detector
import pickle
import tqdm
import time
import itertools
import matplotlib.pyplot as plt
import pandas as pd
import pickle

PRIO_ORDER = ['E01', 'F01', 'unknown', 'PO01', 'E02', 'LA03', 'LA02', 'K02', 'PO02', 'PO03', 'PN02', 'E03', 'B02',
              'FALSE']
save_dir = r'/data/sdv1/whtm/mask/test/0525_particle/result/pkl/'


def predict(img, *args, **kwargs):
    assert isinstance(img, str)
    model = kwargs.get('model')
    classes = kwargs.get('classes')

    image_name = img.replace('\\', '/').split('/')[-1]
    # image_gray = cv2.imread(img, 0)
    image_color = cv2.imread(img, 1)

    pkl_file = os.path.join(save_dir, image_name.replace('jpg', 'pkl'))
    if os.path.exists(pkl_file):
        with open(pkl_file, 'rb') as f:
            mmdet_result = pickle.load(f)
    else:
        mmdet_result = inference_detector(model, image_color)
        with open(pkl_file, 'wb') as f:
            pickle.dump(mmdet_result, f)
    # mmdet_result = inference_detector(model, image_color)
    code, bbox, score, image = judge_code(mmdet_result, image_color, image_name, classes)

    return code, bbox, score, image


def init_model(config, checkpoint, *args, **kwargs):
    model = init_detector(config, checkpoint)
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
    image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
    result = np.array(mmdet_result)
    h, w, c = image_color.shape
    _ = draw_bboxes(image_color, image_name, mmdet_result, classes)

    d_ = model_test(mmdet_result, image_name, classes)
    det_df = pd.DataFrame(d_, columns=['name', 'category', 'bbox', 'score', 'bbox_score',
                                       'xmin', 'ymin', 'xmax', 'ymax', 'ctx', 'cty', 'size', 'remark'])

    # filter confidence
    all = classes.copy()
    all.remove('pixel')
    det_df = filter_code(det_df, all, 0.2)
    det_df = filter_code(det_df, 'pixel', 0.3)

    # judge LA series
    det_df = filter_code(det_df, 'JC', 1, 'LA02')
    det_df = filter_code(det_df, 'LS2', 1, 'LA02')
    det_df = filter_code(det_df, 'LS1', 1, 'LA03')

    # judge short(E01/E02)

    # judge E series
    det_df = count_code(det_df, 'DK', 2, 'E01', 'E02')
    det_df = count_code(det_df, 'LK', 2, 'E01', 'E02')
    # det_df = filter_code(det_df, 'SLK', 1, 'E01')

    # judge particle
    if ('KP' in det_df['category'].values) or ('PP' in det_df['category'].values) or (
            'BP' in det_df['category'].values):
        det_df, image_color = judge_particle(det_df, result, image_color, image_gray)

    # filter pixels
    det_df = filter_code(det_df, 'pixel', 1)
    # final judge
    code, bbox, score = final_judge(det_df, PRIO_ORDER, prio_weight=0.8)
    draw_code(image_color, code, score, (0, h))
    # cv2.imwrite(os.path.join('/data/sdv1/whtm/mask/test/result/', image_name), image_draw)

    return code, bbox, score, image_color


def judge_particle(det_df, result, image_color, image_gray):
    # judge particle
    # det_df = filter_code(det_df, 'BP', 1, 'PN02')

    # calculate area size
    pixels = det_df.loc[det_df['category'] == 'pixel', 'bbox_score'].values
    pixels = [list(a) for a in pixels]
    pixels = np.array(pixels)
    if len(pixels) == 0:
        print('particle in the figure but no enough pixels detected!')
        det_df = filter_code(det_df, ['KP', 'PP'], 1, 'unknown')
        return det_df, image_color
    # sorting
    pixels = pixels[pixels[:, -1].argsort()]
    # template pixel size and pixel mask
    template_pixel_cor = pixels[0].astype(int)
    template_pixel = image_gray[template_pixel_cor[1]:template_pixel_cor[3],
                     template_pixel_cor[0]:template_pixel_cor[2]]
    cv2.imwrite('/data/sdv1/whtm/mask/test/temp/template_pixel.jpg', template_pixel)
    thr = template_pixel[int(template_pixel.shape[0] / 2 - 5):int(template_pixel.shape[0] / 2 + 5),
          int(template_pixel.shape[1] / 2 - 5):int(template_pixel.shape[1] / 2 + 5)]
    thr = np.mean(thr)
    # t, otsu = cv2.threshold(template_pixel, thr*0.95, 255, cv2.THRESH_BINARY)
    otsu = cv2.inRange(template_pixel, thr * 0.90, thr * 1.1)
    otsu = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, np.ones([5, 5]))
    cv2.imwrite('/data/sdv1/whtm/mask/test/temp/otsu.jpg', otsu)
    contours, hierarchy = cv2.findContours(otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # bbox_size = template_pixel.size
    try:
        cnt = contours[np.argmax([cv2.contourArea(cnt) for cnt in contours])]
        pixel_size = cv2.contourArea(cnt)
        pixel_mask = np.zeros(otsu.shape)
        pixel_mask = cv2.drawContours(pixel_mask, [cnt], 0, 255, -1)
        cv2.imwrite('/data/sdv1/whtm/mask/test/temp/pixel_mask.jpg', pixel_mask)
    except Exception as e:
        print('no contours detected in the template pixel!')
        det_df = filter_code(det_df, ['KP', 'PP'], 1, 'unknown')
        return det_df, image_color

    bboxes, image_color = generate_pixels(image_color, template_pixel_cor, pixels)
    if bboxes is None:
        print('generate pixels failed!')
        det_df = filter_code(det_df, ['KP', 'PP'], 1, 'unknown')
        return det_df, image_color

    bbox_particle_size = [-1] * len(bboxes)

    for i, bbox in enumerate(bboxes):
        for idx, row in det_df[(det_df['category'] == 'KP') | (det_df['category'] == 'PP') | (det_df['category'] == 'BP')].iterrows():
            det_df.loc[idx, 'remark'].setdefault('covered', [])
            particle_bbox = row['bbox']
            particle_type = row['category']
            if check_contact(bbox, particle_bbox):
                pixel = image_gray[bbox[1]:bbox[3], bbox[0]:bbox[2]]

                if pixel.shape != template_pixel.shape:
                    # print('margin pixel!')
                    bbox_particle_size[i] = 0
                    det_df.loc[idx, 'remark'].setdefault('covered', []).append(0)
                else:
                    # get particle mask
                    overlap = get_overlap(bbox, particle_bbox)
                    iof = bbox_overlap(particle_bbox, bbox, mode='iof')
                    if iof >= 0.98:
                        det_df.loc[idx, 'category'] = 'KP'
                    elif iof <= 0.02:
                        det_df.loc[idx, 'category'] = 'BP'
                    mask_d = [overlap[0] - bbox[0], overlap[1] - bbox[1], overlap[2] - bbox[0],
                              overlap[3] - bbox[1]]
                    mask_d = [int(x) for x in mask_d]
                    a = np.zeros_like(pixel)
                    a = cv2.rectangle(a, (mask_d[0], mask_d[1]), (mask_d[2], mask_d[3]), 255, -1)
                    mask = cv2.bitwise_and(pixel_mask.astype(np.int8), a.astype(np.int8))
                    cv2.imwrite('/data/sdv1/whtm/mask/test/temp/a.jpg', a)

                    # calculate pixel covered size
                    t, otsud = cv2.threshold(pixel, thr * 0.95, 255, cv2.THRESH_BINARY)
                    otsud = cv2.morphologyEx(otsud, cv2.MORPH_OPEN, np.ones([10, 10]))
                    otsud = cv2.bitwise_not(otsud)
                    cv2.imwrite('/data/sdv1/whtm/mask/test/temp/otsud.jpg', otsud)
                    particle_area = cv2.bitwise_and(otsud, otsud, mask=mask.astype(np.uint8))

                    # filter strange particle area
                    mask_d = np.zeros_like(particle_area)
                    s = cv2.erode(particle_area, np.ones([5, 5]))
                    contours, _ = cv2.findContours(s, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    for cnt in contours:
                        x, y, w, h = cv2.boundingRect(cnt)
                        rect_size = w * h
                        d_size = cv2.contourArea(cnt)
                        extend = float(d_size) / rect_size
                        aspect = float(w) / h
                        # print(d_size, extend, aspect)
                        if extend <= 0.2 or aspect <= 0.1 or aspect >= 10:
                            mask_d = cv2.drawContours(mask_d, [cnt], -1, 255, -1)

                    mask_d = cv2.dilate(mask_d, np.ones([5, 5]))
                    mask_d = cv2.bitwise_not(mask_d)

                    d = cv2.bitwise_and(particle_area, particle_area, mask=mask_d)
                    cv2.imwrite('/data/sdv1/whtm/mask/test/temp/d.jpg', d)
                    particle_size = min(np.sum(d) / 255 / pixel_size, 1.0)
                    bbox_particle_size[i] = max(particle_size, bbox_particle_size[i])
                    det_df.loc[idx, 'remark'].setdefault('covered', []).append(particle_size)
        if bbox_particle_size[i] == -1:
            bbox_particle_size[i] = 0
    draw_size(image_color, bboxes, bbox_particle_size)

    total_particle_size = sum(bbox_particle_size)
    if total_particle_size >= 2:
        det_df = filter_code(det_df, 'KP', 1, 'PO01')
        det_df = filter_code(det_df, 'PP', 1, 'PO01')
    elif total_particle_size >= 1 and (check_code_exist(det_df, 'LA03') or check_code_exist(det_df, 'LA02')):
        det_df = filter_code(det_df, 'KP', 1, 'PO01')
        det_df = filter_code(det_df, 'PP', 1, 'PO01')

    for idx, row in det_df[det_df['category'] == 'PP'].iterrows():
        coverd = row['remark']['covered']
        if coverd == []:
            max_size = 0
            total_size = 0
        else:
            max_size = max(coverd)
            total_size = sum(coverd)
        if max_size >= 0.5:
            det_df.loc[idx, 'category'] = 'K02'
        else:
            det_df.loc[idx, 'category'] = 'PO02'

    kp_size_list = []
    for idx, row in det_df[det_df['category'] == 'KP'].iterrows():
        kp_size_list.extend(row['remark']['covered'])
    kp_total_size = sum(kp_size_list)
    if kp_total_size < 1:
        filter_code(det_df, 'KP', 1, 'PO03')
    elif kp_total_size >= 1 and kp_total_size < 2:
        filter_code(det_df, 'KP', 1, 'K02')

    return det_df, image_color


def get_overlap(bbox1, bbox2):
    overlap = [max(bbox1[0], bbox2[0]), max(bbox1[1], bbox2[1]), min(bbox1[2], bbox2[2]),
               min(bbox1[3], bbox2[3])]
    if overlap[2] - overlap[0] <= 0 or overlap[3] - overlap[1] <= 0:
        overlap = []
    return overlap


def bbox_overlap(bbox1, bbox2, mode='iou'):
    assert mode in ('iou', 'iof')
    overlap = get_overlap(bbox1, bbox2)
    if overlap == []:
        overlap_size = 0
    else:
        overlap_size = (overlap[2] - overlap[0]) * (overlap[3] - overlap[1])

    bbox1_size = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_size = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    if mode == 'iof':
        return overlap_size / bbox1_size
    else:
        return overlap_size / (bbox1_size + bbox2_size - overlap_size)


def judge_short_count(det_df, result, image_color, image_gray):
    # TODO: design algorithm to count short pixels number
    pass


def draw_code(img, code, score, position, size=2):
    cv2.putText(img, f'{code}: {str(np.round(score, 2))}', position,
                cv2.FONT_HERSHEY_SIMPLEX, size, (0, 255, 0), 3)


def draw_bboxes(img, img_name, mmdet_results, codes, thr=0):
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


def draw_size(img, bboxes, sizes, font_size=1):
    assert len(bboxes) == len(sizes)
    for i, bbox in enumerate(bboxes):
        size = sizes[i]
        cv2.putText(img, f'size: {str(np.round(size, 2))}', (int(bbox[0]), int(bbox[3])),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 0, 0), 2)


def generate_pixels(img_c, anchor, pixels):
    img = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)
    h, w = img.shape

    obj_h, obj_w = anchor[3] - anchor[1], anchor[2] - anchor[0]

    centroids = pixels[:, :2]
    centroids = centroids.astype(int)

    error = np.mean([obj_h, obj_w]) * 0.1
    num = len(centroids)
    r_ = []
    for i in range(num):
        for j in range(1, num - i):
            c1 = centroids[i, :]
            c2 = centroids[i + j, :]
            d = c2 - c1
            r_.append(d)
    try:
        r_ = np.abs(np.array(r_))
        x_interval = np.min(r_[np.where(r_[:, 1] < error)][:, 0])
        y_interval = np.min(r_[np.where(r_[:, 0] < error)][:, 1])
    except Exception as e:
        # print(e)
        return None, img_c
    error = np.mean([x_interval, y_interval]) * 0.5
    all_c = make_grid(anchor, x_interval, y_interval, h, w)

    while True:
        left_pixels = []
        all_c_check = np.round((np.array(all_c) / error).astype(float))
        for p in pixels[1:]:
            p = p[:2].astype(int)
            p_check = np.round(p / error)
            if p_check not in all_c_check:
                left_pixels.append(p)
        if len(left_pixels) == 0:
            break
        else:
            all_c.extend(make_grid(left_pixels[0], x_interval, y_interval, h, w))
            pixels = left_pixels

    bboxes = []
    for c in all_c:
        # cv2.rectangle(cent_img, (c[0], c[1]), (c[0]+obj_w, c[1]+obj_h), (0, 0, 255), -1)
        bbox = [c[0], c[1], c[0] + obj_w, c[1] + obj_h]
        bbox = np.clip(bbox, 0, max(w, h))
        img_bbox = [0, 0, w, h]
        overlap = [max(bbox[0], img_bbox[0]), max(bbox[1], img_bbox[1]), min(bbox[2], img_bbox[2]),
                   min(bbox[3], img_bbox[3])]
        overlap_size = max((overlap[2] - overlap[0]), 0) * max((overlap[3] - overlap[1]), 0)
        if overlap_size > 0:
            # print(overlap_size, bbox, overlap)
            bboxes.append(bbox)

    bboxes = np.array(bboxes)
    # print(len(bboxes), bboxes)

    # ed = time.time()
    # print('spend time: {:.2f}s'.format(ed - st))

    cent_img = np.zeros_like(img_c)
    for b in bboxes:
        cv2.rectangle(cent_img, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), -1)
    added_img = cv2.addWeighted(img_c, 1, cent_img, 0.5, 0)
    # save_path = os.path.join(save_dir, image_name)
    # cv2.imwrite(save_path, added_img)

    return bboxes, added_img


def make_grid(anchor, x_interval, y_interval, h, w):
    num_rows = int(h / y_interval)
    num_cols = int(w / x_interval)

    all_centroids = []
    #     print(centroids_)
    bias_x = int((anchor[0] - h / 2) / x_interval)
    bias_y = int((anchor[1] - w / 2) / y_interval)
    all_centroids.append(anchor[:2])
    # select a center to generate the grid
    for i, j in itertools.product(range(-num_rows - bias_x, num_rows + 1 - bias_x),
                                  range(-num_cols - bias_y, num_cols + 1 - bias_y)):
        new_x = anchor[0] + i * x_interval
        new_y = anchor[1] + j * y_interval
        if i == 0 and j == 0:
            continue
        all_centroids.append(np.array([new_x, new_y]))
    # print(all_centroids)
    return all_centroids


def check_inside(bbox1, bbox2, sigma=0.05):
    '''
    check whether bbox1 is inside bbox2 (IOF < sigma)
    :param bbox1:
    :param bbox2:
    :return:
    '''
    overlap = [max(bbox1[0], bbox2[0]), max(bbox1[1], bbox2[1]), min(bbox1[2], bbox2[2]),
               min(bbox1[3], bbox2[3])]
    w = max(0, overlap[2] - overlap[0])
    h = max(0, overlap[3] - overlap[1])
    overlap_size = w * h
    bbox1_size = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    if overlap_size / bbox1_size >= (1 - sigma):
        return True
    else:
        return False


def check_contact(bbox1, bbox2, sigma=0.05):
    '''
    check whether bbox1 contacts bbox2 (IOF > sigma)
    :param bbox1:
    :param bbox2:
    :return:
    '''
    overlap = [max(bbox1[0], bbox2[0]), max(bbox1[1], bbox2[1]), min(bbox1[2], bbox2[2]),
               min(bbox1[3], bbox2[3])]
    w = max(0, overlap[2] - overlap[0])
    h = max(0, overlap[3] - overlap[1])
    overlap_size = w * h
    bbox1_size = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    if overlap_size / bbox1_size >= sigma:
        return True
    else:
        return False


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


def count_code(df, code, count, code1, code2=None, thr=0):
    if code2 is None:
        code2 = code
    check_code = df[(df['category'] == code) & (df['score'] >= thr)]
    if len(check_code) >= count:
        df.loc[df['category'] == code, 'category'] = code1
    else:
        df.loc[df['category'] == code, 'category'] = code2
    return df


def check_code_exist(df, code):
    if code in df['category'].values:
        return True
    else:
        return False


def final_judge(det_df, prio_order=None, prio_weight=0.8):
    if len(det_df) != 0:
        Max_conf = det_df['score'].values.max()
        prio_thr = Max_conf * prio_weight
        filtered_final = det_df[det_df['score'] >= prio_thr]

        final_code = prio_check(prio_order, list(filtered_final['category']))
        if final_code == 'NOT INPLEMENTED':
            return 'NOT INPLEMENTED', [], 0
        best_score = filtered_final.loc[filtered_final['category'] == final_code, 'score'].max()
        best_idx = int(filtered_final.loc[filtered_final['category'] == final_code, 'score'].idxmax())
        best_bbox = filtered_final.loc[best_idx, 'bbox']
    else:
        final_code = 'FALSE'
        best_bbox = []
        best_score = 1
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
        return 'NOT INPLEMENTED'
    final_code = prio_lst[min(idx_lst)]
    return final_code


if __name__ == '__main__':
    path = r'/data/sdv1/whtm/mask/model/FMM_color_0519/'
    opt = process(path)
    img_path = r'/data/sdv1/whtm/mask/test/data/'
    progressbar = tqdm.tqdm(glob.glob(os.path.join(img_path, '*.jpg')))
    for img in progressbar:
        code, bbox, score, _ = predict(img, **opt)
    # img = r'/data/sdv1/whtm/mask/test/data/U4549FH2FAHRT109_-380159-2_122-419_before.jpg'
    # code, bbox, score, _ = predict(img, **opt)
