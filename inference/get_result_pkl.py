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

    if len(imgs) == 1:
        final_result = inference_detector(model, imgs[0])
    else:
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
    imgs = glob.glob('/data/sdv1/whtm/ddj/data/0326test/diandengji_0319_add/*.jpg')
    cfg_file = '/data/sdv1/whtm/ddj/config/ovli_0331.py'
    ckpt_file = '/data/sdv1/whtm/ddj/work_dir/ovli_0331_2/epoch_6.pth'
    save_file = r'/data/sdv1/whtm/ddj/result/ovli_0331.pkl'
    results = model_test(imgs, cfg_file, ckpt_file, save_file)
