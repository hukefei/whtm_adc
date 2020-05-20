import cv2
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import itertools
import time


def generate_pixels(img_f, pixel_cor, save_dir=None):
    image_name = img_f.replace('\\', '/').split('/')[-1]
    img = cv2.imread(img_f, 0)
    img_c = cv2.imread(img_f, 1)
    h, w = img.shape

    st = time.time()

    obj = img[pixel_cor[0]:pixel_cor[2], pixel_cor[1]:pixel_cor[3]]
    obj_h, obj_w = obj.shape

    res = cv2.matchTemplate(obj, img, method=cv2.TM_CCOEFF_NORMED)
    thr, dst = cv2.threshold(res, 0.7, 255, type=cv2.THRESH_BINARY)
    dst = np.uint8(dst)

    nccomps = cv2.connectedComponentsWithStats(dst)
    _ = nccomps[0]
    labels = nccomps[1]
    status = nccomps[2]
    centroids = nccomps[3]

    centroids = centroids.astype(int)
    centroids = centroids[1:]

    error = np.mean([obj_h, obj_w]) * 0.1
    num = len(centroids)
    r_ = []
    for i in range(num):
        for j in range(1, num - i):
            c1 = centroids[i, :]
            c2 = centroids[i + j, :]
            d = c2 - c1
            r_.append(d)
    r_ = np.abs(np.array(r_))
    x_interval = np.min(r_[np.where(r_[:, 1] < error)][:, 0])
    y_interval = np.min(r_[np.where(r_[:, 0] < error)][:, 1])

    all_c = make_grid(centroids, x_interval, y_interval, h, w)

    bboxes = []
    for c in all_c:
        # cv2.rectangle(cent_img, (c[0], c[1]), (c[0]+obj_w, c[1]+obj_h), (0, 0, 255), -1)
        bbox = [c[0], c[1], c[0] + obj_w, c[1] + obj_h]
        bbox = np.clip(bbox, 0, max(w, h))
        img_bbox = [0, 0, w, h]
        overlap = [max(bbox[0], img_bbox[0]), max(bbox[1], img_bbox[1]), min(bbox[2], img_bbox[2]),
                   min(bbox[3], img_bbox[3])]
        overlap_size = (overlap[2] - overlap[0]) * (overlap[3] - overlap[1])
        if overlap_size > 0:
            # print(overlap_size, bbox, overlap)
            bboxes.append(bbox)

    bboxes = np.array(bboxes)
    # print(len(bboxes), bboxes)

    cent_img = np.zeros_like(img_c)
    for b in bboxes:
        cv2.rectangle(cent_img, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), -1)

    ed = time.time()
    print('spend time: {:.2f}s'.format(ed - st))

    if save_dir is not None:
        added_img = cv2.addWeighted(img_c, 1, cent_img, 0.5, 0)
        save_path = os.path.join(save_dir, image_name)
        cv2.imwrite(save_path, added_img)

    return bboxes, added_img


def make_grid(centroid, x_interval, y_interval, h, w):
    num_rows = int(h / y_interval)
    num_cols = int(w / x_interval)
    error = np.mean([x_interval, y_interval]) * 0.1
    centroids_ = centroid.copy()
    all_centroids = []
    #     print(centroids_)
    while True:
        if len(centroids_) == 0:
            break
        anchor = centroids_[0]
        bias_x = int((anchor[0] - h / 2) / x_interval)
        bias_y = int((anchor[1] - w / 2) / y_interval)
        centroids_ = np.delete(centroids_, 0, 0)
        all_centroids.append(anchor)
        # select a center to generate the grid
        for i, j in itertools.product(range(-num_rows - bias_x, num_rows + 1 - bias_x),
                                      range(-num_cols - bias_y, num_cols + 1 - bias_y)):
            new_x = anchor[0] + i * x_interval
            new_y = anchor[1] + j * y_interval
            if i == 0 and j == 0:
                continue
            new_flag = True
            for idx, t in enumerate(centroids_):
                if (t[0] <= new_x + error) and (t[0] >= new_x - error) and (t[1] <= new_y + error) and (
                        t[1] >= new_y - error):
                    all_centroids.append(centroids_[idx])
                    centroids_ = np.delete(centroids_, idx, 0)
                    new_flag = False
            if new_flag:
                all_centroids.append(np.array([new_x, new_y]))

    all_centroids = np.array(all_centroids)
    # print(all_centroids)
    return all_centroids


if __name__ == '__main__':
    img_f = r'C:\Users\root\Pictures\MASK\U4639FH1FBEBW201_-103437-3_-601-7941_before.jpg'
    save_dir = r'C:\Users\root\Pictures\MASK\save'
    pixel_cor = [184, 406, 370, 586]
    generate_pixels(img_f, pixel_cor, save_dir)
