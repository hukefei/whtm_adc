from rule.rule_56A02 import default_rule
import pandas as pd
import json
import glob
from mmdet.apis import inference_detector, init_detector
import pickle
import os
import cv2
import tqdm
import numpy as np



def inference_cls(result_file,
                  imgs,
                  config,
                  code_file,
                  output,
                  size_file=None,
                  save_img=False,
                  save_all=False):
    # if os.path.exists(result_file):
    #     with open(result_file, 'rb') as f:
    #         results = pickle.load(f)
    # else:
    results = []
    model = init_detector(CONFIG, CKPT)
    progressbar = tqdm.tqdm(imgs)
    for img in progressbar:
        result = inference_detector(model, img)
        results.append(result)
    # with open(result_file, 'wb') as fp:
    #     pickle.dump(results, fp)

    cls_result_lst = []
    for i, result in enumerate(results):
        img = imgs[i]
        if result == []:
            print('skip image {}'.format(img))
            continue
        img_name = img.replace('\\', '/').split('/')[-1]
        gt_code = img.replace('\\', '/').split('/')[-2]
        print('processing {}'.format(img_name))
        if size_file is not None:
            with open(size_file, 'r') as f:
                size_dict = json.load(f)
            if img_name not in size_dict.keys():
                print('{} has no size data!'.format(img_name))
                s = 0
            else:
                s = size_dict[img_name]
                print('{}: {}'.format(img_name, s))
        else:
            s = None
        main_code, bbox, score, img_ = default_rule(result, img, img_name, config, code_file, save_img, size=s)
        print(main_code, bbox, score)

        draw_code(img_, main_code, score)

        if save_all:
            save_path = os.path.join(output, gt_code, main_code)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            # shutil.copy(img, os.path.join(save_path, img_name))
            cv2.imwrite(os.path.join(save_path, img_name), img_)
        elif main_code != gt_code:
            save_path = os.path.join(output, gt_code, main_code)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            # shutil.copy(img, os.path.join(save_path, img_name))
            cv2.imwrite(os.path.join(save_path, img_name), img_)
        cls_result_lst.append({'image name': img_name, 'pred code': main_code, 'defect score': score, 'oic code': gt_code})


    cls_df = pd.DataFrame(cls_result_lst)
    cls_df.to_excel(os.path.join(output, 'result.xlsx'))

def draw_code(img, code, score, position=None, size=2):
    if position is None:
        h, w, c = img.shape
        position = (0, h)
    cv2.putText(img, f'{code}: {str(np.round(score, 2))}', position,
                cv2.FONT_HERSHEY_SIMPLEX, size, (0, 255, 0), 3)

if __name__ == '__main__':
    imgs = glob.glob(r'/data/sdv1/whtm/56A02/test/test_split/*/*.jpg')
    pkl_file = r'/data/sdv1/whtm/56A02/result/0813/test.pkl'
    json_file = None
    output = r'/data/sdv1/whtm/56A02/result/0813/result'
    size_file = r'/data/sdv1/whtm/56A02/test/test_split/size.json'

    code_file = r'/data/sdv1/whtm/56A02/model/0811/classes.txt'
    CONFIG = '/data/sdv1/whtm/56A02/model/0811/config.py'
    CKPT = '/data/sdv1/whtm/56A02/model/0811/model.pth'

    # size_file = None
    # if not os.path.exists(img_save_path):
    #     os.makedirs(img_save_path)
    # open config file
    if json_file is not None:
        with open(json_file) as f:
            config = json.load(f)
    else:
        config = None
    with open(code_file) as f:
        codes = f.read().splitlines()

    if not os.path.exists(output):
        os.makedirs(output)

    inference_cls(pkl_file, imgs, config, codes, output, save_img=True, size_file=size_file)
