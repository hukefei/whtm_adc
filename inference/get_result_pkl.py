#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
from mmdet.apis import inference_detector, init_detector
import pickle
from progressbar import ProgressBar


def model_test(imgs,
               cfg_file,
               ckpt_file,
               save_file,
               device='cuda:0'):
    model = init_detector(cfg_file, ckpt_file, device=device)
    final_result = []

    pbar = ProgressBar().start()
    for i, img in enumerate(imgs):
        pbar.update(int(i / (len(imgs) - 1) * 100))
        result = inference_detector(model, img)
        final_result.append(result)
    pbar.finish()

    if save_file is not None:
        with open(save_file, 'wb') as fp:
            pickle.dump(final_result, fp)

    return final_result



if __name__ == '__main__':
    imgs = glob.glob('/data/sdv1/GD/data/guangdong1_round1_testA_20190818/*.jpg')
    cfg_file = '/data/sdv1/GD/config/gd_0820_new.py'
    ckpt_file = '/data/sdv1/GD/model/0820/epoch_22.pth'
    save_file = r'result.pkl'
    results = model_test(imgs, cfg_file, ckpt_file, save_file)
