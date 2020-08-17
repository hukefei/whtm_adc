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
    imgs = glob.glob('/data/sdv1/whtm/56A02/test/test_split/*/*.jpg')
    cfg_file = '/data/sdv1/whtm/56A02/model/0806/config.py'
    ckpt_file = '/data/sdv1/whtm/56A02/model/0806/model.pth'
    save_file = r'/data/sdv1/whtm/56A02/result/0806/test_split.pkl'
    if not os.path.exists(os.path.dirname(save_file)):
        os.makedirs(os.path.dirname(save_file))
    results = model_test(imgs, cfg_file, ckpt_file, save_file)
