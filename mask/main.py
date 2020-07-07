import cv2
import os
import numpy as np
import glob
from mmdet.apis import inference_detector, init_detector
import tqdm
import itertools
import pandas as pd

PRIO_ORDER = ['CN', 'E01', 'F01', 'PO01', 'E02', 'LA03', 'LA02', 'unknown', 'K02', 'PO02', 'PO03', 'PN02', 'E03', 'B02',
              'FALSE']
PRIO_WEIGHT = 0.5
KILLER = ['CN', 'E01', 'F01', 'PO01']


def predict(img, **kwargs):
    assert isinstance(img, str)
    model = kwargs.get('model')
    classes = kwargs.get('classes')

    image_name = img.replace('\\', '/').split('/')[-1]
    image_color = cv2.imread(img)
    mmdet_result = inference_detector(model, img)
    mmdet_result = compose_nms(mmdet_result, classes)
    code, bbox, score, image = judge_code(mmdet_result, image_color, image_name, classes)

    return code, bbox, score, image

def compose_nms(mmdet_result, classes):
    nonkiller = classes.copy()
    nonkiller.remove('F01')
    nonkiller.remove('DK')
    nonkiller.remove('LK')
    nonkiller.remove('CN')
    nonkiller.remove('pixel')
    pair = [(['B02', 'E03'], 0.5, 'iof-h'), (['E03', 'BP'], 0.5, 'iof-l'), (nonkiller, 0.7, 'iou')]
    for p in pair:
        c_l, thr, mode = p
        index_list = [classes.index(c) for c in c_l]
        m_result = [mmdet_result[i] for i in index_list]
        m_result = all_classes_nms(m_result, thr, mode)
        for idx, i in enumerate(index_list):
            mmdet_result[i] = m_result[idx]
    return mmdet_result


def init_model(config, checkpoint, *args, **kwargs):
    model = init_detector(config, checkpoint, **kwargs)
    return model


def process(path):
    files = os.listdir(path)
    assert 'config_color.py' in files, 'mmdet config file shoule be included'
    assert 'model.pth' in files, 'checkpoint file should be included'
    assert 'classes.txt' in files, 'classes file should be included'
    config_color = os.path.join(path, 'config_color.py')
    checkpoint = os.path.join(path, 'model.pth')
    model = init_model(config_color, checkpoint)
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
    h, w, c = image_color.shape
    _ = draw_bboxes(image_color, mmdet_result[:-1], classes[:-1])

    d_ = model_test(mmdet_result, image_name, classes)
    det_df = pd.DataFrame(d_, columns=['name', 'category', 'bbox', 'score', 'bbox_score',
                                       'xmin', 'ymin', 'xmax', 'ymax', 'ctx', 'cty', 'size', 'remark'])
    det_df = det_df.sort_values(by='score', ascending=False)

    # filter confidence
    all = classes.copy()
    all.remove('pixel')
    det_df = filter_code(det_df, all, 0.1)
    det_df = filter_code(det_df, 'pixel', 0.3)
    det_df = filter_code(det_df, 'CN', 0.4)

    # judge LA series
    det_df = count_code(det_df, 'JC', 2, 'LA02')
    det_df = filter_code(det_df, 'LS2', 1, 'LA02')
    det_df = filter_code(det_df, 'LS1', 1, 'LA03')

    # judge short(E01/E02)
    count = judge_short_count(det_df)
    if count <= 0:
        det_df = filter_code(det_df, 'LK', 1, 'unknown')
    elif count == 1:
        det_df = filter_code(det_df, 'LK', 1, 'E02')
    elif count >= 2:
        det_df = filter_code(det_df, 'LK', 1, 'E01')

    # judge E series
    det_df = count_code(det_df, 'DK', 2, 'E01', 'E02')

    # judge particle
    if ('KP' in det_df['category'].values) or ('PP' in det_df['category'].values) or (
            'BP' in det_df['category'].values):
        det_df, image_color = judge_particle(det_df, image_color, image_gray)

    # filter pixels
    det_df = filter_code(det_df, 'pixel', 1)
    # final judge
    code, bbox, score = final_judge(det_df, PRIO_ORDER, prio_weight=PRIO_WEIGHT)
    draw_code(image_color, code, score, (0, h))

    return code, bbox, score, image_color


def judge_particle(det_df, image_color, image_gray):
    # judge particle
    h, w, c = image_color.shape
    image_size = h * w

    # filter big particle
    det_df.loc[(det_df['category'] == 'PP') & (det_df['size'] >= image_size * 0.25), 'category'] = 'PO01'

    # calculate area size
    pixels = det_df.loc[det_df['category'] == 'pixel', 'bbox_score'].values
    pixels = [list(a) for a in pixels]
    pixels = np.array(pixels)
    if len(pixels) == 0:
        print('particle in the figure but no enough pixels detected!')
        det_df = filter_code(det_df, ['KP', 'PP'], 1, 'unknown')
        det_df = filter_code(det_df, 'BP', 1, 'PN02')
        return det_df, image_color
    # sorting
    pixels = pixels[pixels[:, -1].argsort()[::-1]]
    # template pixel size and pixel mask
    template_pixel_cor = pixels[0].astype(int)
    template_pixel = image_gray[template_pixel_cor[1]:template_pixel_cor[3],
                     template_pixel_cor[0]:template_pixel_cor[2]]
    p_h, p_w = template_pixel.shape
    template_pixel_size = p_h * p_w
    # cv2.imwrite('/data/sdv1/whtm/mask/test/temp/template_pixel.jpg', template_pixel)
    thr = get_threshold(image_gray, pixels)
    # t, otsu = cv2.threshold(template_pixel, thr*0.95, 255, cv2.THRESH_BINARY)
    otsu = cv2.inRange(template_pixel, thr * 0.9, thr * 1.1)
    otsu = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, np.ones([5, 5]))
    # cv2.imwrite('/data/sdv1/whtm/mask/test/temp/otsu.jpg', otsu)
    contours, hierarchy = cv2.findContours(otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # bbox_size = template_pixel.size
    try:
        cnt = contours[np.argmax([cv2.contourArea(cnt) for cnt in contours])]
        pixel_size = cv2.contourArea(cnt)
        pixel_mask = np.zeros(otsu.shape)
        pixel_mask = cv2.drawContours(pixel_mask, [cnt], 0, 255, -1)
    except Exception as e:
        print('no contours detected in the template pixel!')
        det_df = filter_code(det_df, ['KP', 'PP'], 1, 'unknown')
        det_df = filter_code(det_df, 'BP', 1, 'PN02')
        return det_df, image_color

    bboxes = generate_pixels(image_color, template_pixel_cor, pixels)
    if bboxes is None:
        print('generate pixels failed!')
        det_df = filter_code(det_df, ['KP', 'PP'], 1, 'unknown')
        det_df = filter_code(det_df, 'BP', 1, 'PN02')
        return det_df, image_color
    # image_color = draw_pixels(bboxes, image_color)

    bbox_particle_size = [-1] * len(bboxes)

    for i, bbox in enumerate(bboxes):
        for idx, row in det_df[
            (det_df['category'] == 'KP') | (det_df['category'] == 'PP') | (det_df['category'] == 'BP')].iterrows():
            det_df.loc[idx, 'remark'].setdefault('covered', [])
            det_df.loc[idx, 'remark'].setdefault('iof', [])
            particle_bbox = row['bbox']
            particle_type = row['category']
            if check_contact(bbox, particle_bbox):
                pixel = image_gray[bbox[1]:bbox[3], bbox[0]:bbox[2]]

                if pixel.shape != template_pixel.shape:
                    # print('margin pixel!')
                    pixel_h, pixel_w = pixel.shape
                    x = bbox[:2].copy()
                    if x[0] == 0:
                        x[0] = bbox[2] - p_w
                    if x[1] == 0:
                        x[1] = bbox[3] - p_h
                    pixel_ = np.ones((p_h, p_w))
                    pixel_[(bbox[1] - x[1]):(bbox[3] - x[1]), (bbox[0] - x[0]):(bbox[2] - x[0])] = pixel
                    pixel = pixel_.astype(np.uint8)

                # get particle mask
                overlap = get_overlap(bbox, particle_bbox)
                iof = bbox_overlap(particle_bbox, bbox, mode='iof')
                det_df.loc[idx, 'remark']['iof'].append(iof)

                mask_d = [overlap[0] - bbox[0], overlap[1] - bbox[1], overlap[2] - bbox[0],
                          overlap[3] - bbox[1]]
                mask_d = [int(x) for x in mask_d]
                a = np.zeros_like(pixel)
                a = cv2.rectangle(a, (mask_d[0], mask_d[1]), (mask_d[2], mask_d[3]), 255, -1)
                mask = cv2.bitwise_and(pixel_mask.astype(np.int8), a.astype(np.int8))

                # calculate pixel covered size
                otsud = cv2.inRange(pixel, thr * 0.95, thr * 1.05)
                otsud = cv2.bitwise_not(otsud)
                r2 = cv2.morphologyEx(otsud, cv2.MORPH_BLACKHAT, np.ones([5, 5]))
                otsud = cv2.bitwise_or(otsud, r2)
                particle_area = cv2.bitwise_and(otsud, otsud, mask=mask.astype(np.uint8))
                r1 = cv2.morphologyEx(particle_area, cv2.MORPH_TOPHAT, np.ones((10, 10)))
                r1 = cv2.bitwise_not(r1)
                particle_area = cv2.bitwise_and(particle_area, particle_area, mask=r1)
                contours, hierarchy = cv2.findContours(particle_area, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                if contours != []:
                    particle_size = np.max([cv2.contourArea(cnt) for cnt in contours])
                    particle_size = min(particle_size / pixel_size, 1.0)
                else:
                    particle_size = min(np.sum(particle_area) / 255 / pixel_size, 1.0)

                bbox_particle_size[i] = max(particle_size, bbox_particle_size[i])
                det_df.loc[idx, 'remark'].setdefault('covered', []).append(particle_size)

        if bbox_particle_size[i] == -1:
            bbox_particle_size[i] = 0
    draw_size(image_color, bboxes, bbox_particle_size)

    # modify code
    for idx, row in det_df[det_df['category'] == 'KP'].iterrows():
        iof = row['remark']['iof']
        size = row['size']
        if sum(iof) <= 0.6 or size >= 1.1 * template_pixel_size or len(iof) > 1:
            print('KP -> PP')
            det_df.loc[idx, 'category'] = 'PP'

    for idx, row in det_df[det_df['category'] == 'PP'].iterrows():
        covered = row['remark']['covered']
        iof = row['remark']['iof']
        if np.sum(covered) <= 0.01 and sum(iof) <= 0.1:
            print('PP -> BP')
            det_df.loc[idx, 'category'] = 'BP'

    for idx, row in det_df[det_df['category'] == 'BP'].iterrows():
        covered = row['remark']['covered']
        iof = row['remark']['iof']
        if np.sum(covered) >= 0.02:
            print('BP -> PP')
            det_df.loc[idx, 'category'] = 'PP'

    # judge PN02
    det_df = filter_code(det_df, 'BP', 1, 'PN02')

    total_particle_size = sum(bbox_particle_size)
    # judge code PO01
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
        else:
            max_size = max(coverd)
        if max_size >= 0.5:
            det_df.loc[idx, 'category'] = 'K02'
        else:
            det_df.loc[idx, 'category'] = 'PO02'

    if total_particle_size < 1:
        filter_code(det_df, 'KP', 1, 'PO03')
    elif total_particle_size >= 1 and total_particle_size < 2:
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


def judge_short_count(det_df):
    if 'pixel' in det_df['category'].values or 'JC' in det_df['category'].values or 'DK' in det_df['category'].values:
        template_pixel = det_df.loc[(det_df['category'] == 'pixel') | (det_df['category'] == 'JC') | (
                det_df['category'] == 'DK'), 'bbox'].values[0]
        pixel_size = (template_pixel[2] - template_pixel[0]) * (template_pixel[3] - template_pixel[1])
        count = 0
        for idx, row in det_df[det_df['category'] == 'LK'].iterrows():
            bbox = row['bbox']
            lk_size = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if lk_size / pixel_size >= 8:
                count += 2
            else:
                count += 1
        return count
    else:
        return -1


def draw_code(img, code, score, position, size=2):
    cv2.putText(img, f'{code}: {str(np.round(score, 2))}', position,
                cv2.FONT_HERSHEY_SIMPLEX, size, (0, 255, 0), 3)


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


def draw_size(img, bboxes, sizes, font_size=1):
    assert len(bboxes) == len(sizes)
    for i, bbox in enumerate(bboxes):
        size = sizes[i]
        cv2.putText(img, f'size: {str(np.round(size, 2))}', (int(bbox[0]), int(bbox[3])),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 0, 0), 2)
    cv2.putText(img, f'total size: {str(np.round(sum(sizes), 2))}', (0, 50),
                cv2.FONT_HERSHEY_SIMPLEX, font_size*2, (0, 0, 255), 2)


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
        x_interval, y_interval = get_interval(r_, error)
    except Exception as e:
        return None
    error = np.mean([obj_h, obj_w]) * 0.5
    all_c = make_grid(anchor, x_interval, y_interval, h, w)

    while True:
        left_pixels = []
        all_c_check = np.array(all_c)
        # all_c_check = np.round((np.array(all_c) / error).astype(float))
        for p in pixels[1:]:
            p = p[:2].astype(int)
            dist = np.abs(all_c_check - p)
            if not np.any(np.bitwise_and(dist[:, 0] < error, dist[:, 1] < error)):
                left_pixels.append(p)
            # p_check = np.round(p / error)
            # if p_check not in all_c_check:
        if len(left_pixels) == 0:
            break
        else:
            all_c.extend(make_grid(left_pixels[0], x_interval, y_interval, h, w))
            pixels = left_pixels

    bboxes = []
    for c in all_c:
        bbox = [c[0], c[1], c[0] + obj_w, c[1] + obj_h]
        bbox = [np.clip(bbox[0], 0, w), np.clip(bbox[1], 0, h), np.clip(bbox[2], 0, w), np.clip(bbox[3], 0, h)]
        img_bbox = [0, 0, w, h]
        overlap = [max(bbox[0], img_bbox[0]), max(bbox[1], img_bbox[1]), min(bbox[2], img_bbox[2]),
                   min(bbox[3], img_bbox[3])]
        overlap_size = max((overlap[2] - overlap[0]), 0) * max((overlap[3] - overlap[1]), 0)
        if overlap_size > 0:
            bboxes.append(bbox)

    bboxes = np.array(bboxes)

    return bboxes


def get_interval(distances, error):
    x_dists = distances[np.where(distances[:, 1] < error)][:, 0]
    y_dists = distances[np.where(distances[:, 0] < error)][:, 1]
    x_dists = x_dists[x_dists.argsort()]
    y_dists = y_dists[y_dists.argsort()]
    x_base = x_dists[0]
    y_base = y_dists[0]
    x_intervals = []
    y_intervals = []
    for d_ in x_dists:
        bs = round(d_ / x_base)
        mod = np.abs(d_ - bs * x_base)
        if mod <= error * 2:
            x_intervals.append(d_ / bs)
    for d_ in y_dists:
        bs = round(d_ / y_base)
        mod = np.abs(d_ - bs * y_base)
        if mod <= error * 2:
            y_intervals.append(d_ / bs)
    x_interval = int(np.mean(x_intervals))
    y_interval = int(np.mean(y_intervals))
    if max(x_interval, y_interval) / min(x_interval, y_interval) >= 2:
        raise ValueError
    return x_interval, y_interval


def get_threshold(image_gray, pixels):
    thr_lst = []
    for pixel in pixels:
        template_pixel_cor = pixel.astype(int)
        template_pixel = image_gray[template_pixel_cor[1]:template_pixel_cor[3],
                         template_pixel_cor[0]:template_pixel_cor[2]]
        thr = template_pixel[int(template_pixel.shape[0] / 2 - 5):int(template_pixel.shape[0] / 2 + 5),
              int(template_pixel.shape[1] / 2 - 5):int(template_pixel.shape[1] / 2 + 5)]
        thr = np.mean(thr)
        thr_lst.append(thr)
    thr = np.mean(thr_lst)
    return thr

def draw_pixels(bboxes, image_color):
    for b in bboxes:
        cv2.rectangle(image_color, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 3)
    return image_color


def make_grid(anchor, x_interval, y_interval, h, w):
    num_rows = int(h / y_interval) + 1
    num_cols = int(w / x_interval) + 1

    all_centroids = []
    bias_x = int((anchor[0] - h / 2) / x_interval)
    bias_y = int((anchor[1] - w / 2) / y_interval)
    all_centroids.append(anchor[:2])
    for i, j in itertools.product(range(-num_rows - bias_x, num_rows + 1 - bias_x),
                                  range(-num_cols - bias_y, num_cols + 1 - bias_y)):
        new_x = anchor[0] + i * x_interval
        new_y = anchor[1] + j * y_interval
        if i == 0 and j == 0:
            continue
        all_centroids.append(np.array([new_x, new_y]))
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


def check_contact(bbox1, bbox2, sigma=0.01):
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
    check_code = df[(df['category'] == code) & (df['score'] >= thr)]
    if len(check_code) >= count:
        df.loc[df['category'] == code, 'category'] = code1
    elif code2 is None:
        df = filter_code(df, code, 1)
    else:
        df.loc[df['category'] == code, 'category'] = code2
    return df


def count(df, code, thr=0):
    check_code = df[(df['category'] == code) & (df['score'] >= thr)]
    return len(check_code)


def check_code_exist(df, code):
    if code in df['category'].values:
        return True
    else:
        return False


def final_judge(det_df, prio_order=None, prio_weight=0.5):
    if len(det_df) != 0:
        for killer in KILLER:
            if killer in det_df.category.values:
                final_code = killer
                best_score = det_df.loc[det_df['category'] == final_code, 'score'].max()
                best_idx = int(det_df.loc[det_df['category'] == final_code, 'score'].idxmax())
                best_bbox = det_df.loc[best_idx, 'bbox']
                return final_code, best_bbox, best_score

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
        elif mode == 'iof-h':
            over = (w * h) / area[i]
        elif mode == 'iof-l':
            over = (w * h) / area[order[1:]]
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
    for idx, l in enumerate(final_):
        final_[idx] = np.array(l)
    return final_


if __name__ == '__main__':
    path = r'/data/sdv1/whtm/mask/model/FMM_color_0519/'
    opt = process(path)
    img_path = r'/data/sdv1/whtm/mask/test/data/'
    progressbar = tqdm.tqdm(glob.glob(os.path.join(img_path, '*.jpg')))
    for img in progressbar:
        code, bbox, score, _ = predict(img, **opt)
