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
    imgs = glob.glob('/data/sdv1/whtm/mask/data/0512/test/*.jpg')
    cfg_file = '/data/sdv1/whtm/mask/config/mask_rcnn_r50_fpn_1x.py'
    ckpt_file = '/data/sdv1/whtm/mask/workdir/0512-1/epoch_12.pth'
    save_file = r'/data/sdv1/whtm/mask/result/0512/particle.pkl'
    results = model_test(imgs, cfg_file, ckpt_file, save_file)
