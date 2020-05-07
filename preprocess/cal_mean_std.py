import numpy as np
import cv2
import os

def cal_mean_std(img):
    means = []
    stds = []
    for i in range(3):
        img_ = img[:, :, i]
        mean = np.average(img_)
        std = np.std(img_)
        means.append(mean)
        stds.append(std)
    return means, stds

def cal_multiple_mean_std(imgs):
    means = {}
    stds = {}
    num = len(imgs)
    for img in imgs:
        m, s = cal_mean_std(img)
        means.setdefault('B', []).append(m[0])
        means.setdefault('G', []).append(m[1])
        means.setdefault('R', []).append(m[2])

        stds.setdefault('B', []).append(s[0])
        stds.setdefault('G', []).append(s[1])
        stds.setdefault('R', []).append(s[2])

    f_m = [np.mean(means[k]) for k in ['B', 'G', 'R']]
    f_s = [np.mean(stds[k]) for k in ['B', 'G', 'R']]
    return f_m, f_s

def cal_all_patterns(img_dir):
    img_b, img_g, img_r, img_w = [], [], [], []
    for img in os.listdir(img_dir):
        pattern = img.split('_')[1]
        if pattern == 'BLUE':
            img_b.append(cv2.imread(os.path.join(img_dir, img)))
        elif pattern == 'GREEN':
            img_g.append(cv2.imread(os.path.join(img_dir, img)))
        elif pattern == 'RED':
            img_r.append(cv2.imread(os.path.join(img_dir, img)))
        elif pattern == 'WHITE':
            img_w.append(cv2.imread(os.path.join(img_dir, img)))
        else:
            continue

    m_result = []
    s_result = []
    for img_set in (img_b, img_g, img_r, img_w):
        rm, rs = cal_multiple_mean_std(img_set)
        m_result += rm
        s_result += rs

    return m_result, s_result

if __name__ == '__main__':
    img_dir = r'E:\diandengji\cal_img_set'
    m, s = cal_all_patterns(img_dir)
    print(m, s)