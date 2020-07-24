#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
from mmdet.apis import inference_detector, init_detector
import pickle
import tqdm
import os
import cv2


def model_test(imgs,
               cfg_file,
               ckpt_file,
               save_file,
               device='cuda:0'):
    assert len(imgs) >= 1, 'There must be at least one image'
    model = init_detector(cfg_file, ckpt_file, device=device)
    final_result = []

    progressbar = tqdm.tqdm(imgs)
    for _, img in enumerate(progressbar):
        image = cv2.imread(img)
        if image is None:
            print('image {} is empty'.format(img))
            final_result.append([])
        result = inference_detector(model, image)
        final_result.append(result)

    if save_file is not None:
        with open(save_file, 'wb') as fp:
            pickle.dump(final_result, fp)

    return final_result



if __name__ == '__main__':
    imgs = glob.glob('/data/sdv1/whtm/56A02/val/total_val/*/*.jpg')
    cfg_file = '/data/sdv1/whtm/56A02/config/56A02_0724.py'
    ckpt_file = '/data/sdv1/whtm/56A02/workdir/0724/epoch_12.pth'
    save_file = r'/data/sdv1/whtm/56A02/result/0724/val_result.pkl'
    if not os.path.exists(os.path.dirname(save_file)):
        os.makedirs(os.path.dirname(save_file))
    results = model_test(imgs, cfg_file, ckpt_file, save_file)
