import cv2
import os
import numpy as np
import glob
from mmdet.apis import inference_detector, init_detector
import tqdm
import itertools
import pandas as pd
from xml.etree import ElementTree as ET
import glob

def main():
    path = r'/data/sdv1/whtm/mask/model/FMM_0611/'
    opt = process(path)
    img_path = r'/data/sdv1/whtm/mask/data/0615/original_images'
    ann_path = r'/data/sdv1/whtm/mask/data/0615/annotations'
    imgs = glob.glob(os.path.join(img_path, '*.jpg'))
    progressbar = tqdm.tqdm(imgs)
    save_dir = r'/data/sdv1/whtm/mask/data/0615/add_pixels/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for img in progressbar:
        image_name = img.split('/')[-1]
        ann_name = image_name.replace('jpg', 'xml')
        ann_ = os.path.join(ann_path, ann_name)
        tree = add_(img, ann_, **opt)
        if tree is not None:
            tree.write(os.path.join(save_dir, ann_name))


def add_(img, ann, **kwargs):
    assert isinstance(img, str)
    model = kwargs.get('model')
    classes = kwargs.get('classes')

    image_name = img.replace('\\', '/').split('/')[-1]
    image_color = cv2.imread(img, 1)

    # pkl_file = os.path.join(save_dir, image_name.replace('jpg', 'pkl'))
    # if os.path.exists(pkl_file):
    #     with open(pkl_file, 'rb') as f:
    #         mmdet_result = pickle.load(f)
    # else:
    #     mmdet_result = inference_detector(model, image_color)
    #     with open(pkl_file, 'wb') as f:
    #         pickle.dump(mmdet_result, f)

    mmdet_result = inference_detector(model, image_color)
    pixels = mmdet_result[-1]
    pixels = np.array(pixels)
    pixels = pixels[pixels[:, -1].argsort()[::-1]]
    pixels = pixels[np.where(pixels[:, 4]>= 0.2)]
    if len(pixels) == 0:
        tree = ET.parse(ann)
        print(img)
        return tree
    # template pixel size and pixel mask
    template_pixel_cor = pixels[0].astype(int)
    obj_w, obj_h = template_pixel_cor[2] - template_pixel_cor[0], template_pixel_cor[3] - template_pixel_cor[1]
    bboxes = generate_pixels(image_color, template_pixel_cor, pixels)
    if bboxes is None:
        print(img)
        tree = ET.parse(ann)
        return tree
    tree = add_pixels(ann, bboxes, [obj_w, obj_h])

    return tree


def init_model(config, checkpoint, *args, **kwargs):
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

def add_pixels(ann, bboxes, pixel_size):
    tree = ET.parse(ann)
    root = tree.getroot()
    w, h = pixel_size
    objs = root.findall('object')
    defect = []
    for obj in objs:
        if obj[0].text != 'pixel':
            bbox = [obj[4][0].text, obj[4][1].text, obj[4][2].text, obj[4][3].text]
            bbox = [int(b) for b in bbox]
            defect.append(bbox)

    for c in bboxes:
        ow, oh = (c[2] - c[0]), (c[3] - c[1])
        if not (ow == w and oh == h):
            continue
        flag = False
        for d in defect:
            overlap = [max(d[0], c[0]), max(d[1], c[1]), min(d[2], c[0] + w), min(d[3], c[1] + h)]
            if (overlap[2] - overlap[0]) > 0 and (overlap[3] - overlap[1]) > 0:
                flag = True
                break
        if flag:
            continue
        obj_element = ET.SubElement(root, 'object')
        name = ET.SubElement(obj_element, 'name')
        name.text = 'pixel'
        pose = ET.SubElement(obj_element, 'pose')
        pose.text = 'Unspecified'
        t = ET.SubElement(obj_element, 'truncated')
        t.text = '0'
        d = ET.SubElement(obj_element, 'difficult')
        d.text = '0'
        bndbox = ET.SubElement(obj_element, 'bndbox')
        xmin = ET.SubElement(bndbox, 'xmin')
        xmin.text = str(c[0])
        ymin = ET.SubElement(bndbox, 'ymin')
        ymin.text = str(c[1])
        xmax = ET.SubElement(bndbox, 'xmax')
        xmax.text = str(c[0] + w)
        ymax = ET.SubElement(bndbox, 'ymax')
        ymax.text = str(c[1] + h)

    return tree


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
        # print(e)
        return None
    error = np.mean([obj_h, obj_w])*0.5
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
        # cv2.rectangle(cent_img, (c[0], c[1]), (c[0]+obj_w, c[1]+obj_h), (0, 0, 255), -1)
        bbox = [c[0], c[1], c[0] + obj_w, c[1] + obj_h]
        bbox = [np.clip(bbox[0], 0, w), np.clip(bbox[1], 0, h), np.clip(bbox[2], 0, w), np.clip(bbox[3], 0, h)]
        img_bbox = [0, 0, w, h]
        overlap = [max(bbox[0], img_bbox[0]), max(bbox[1], img_bbox[1]), min(bbox[2], img_bbox[2]),
                   min(bbox[3], img_bbox[3])]
        overlap_size = max((overlap[2] - overlap[0]), 0) * max((overlap[3] - overlap[1]), 0)
        if overlap_size > 0:
            # print(overlap_size, bbox, overlap)
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
    return x_interval, y_interval


def make_grid(anchor, x_interval, y_interval, h, w):
    num_rows = int(h / y_interval) + 1
    num_cols = int(w / x_interval) + 1

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

if __name__ == '__main__':
    main()