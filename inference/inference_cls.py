from rule.rule_1GE02 import default_rule
import cv2
import os, glob
import pandas as pd
import pickle
import json


def inference_cls(result_file,
                  img_path,
                  config_file,
                  code_file,
                  output_file,
                  img_save_path,
                  size_file=None,
                  save_img=False):
    imgs = glob.glob(os.path.join(img_path, '*.jpg'))

    with open(result_file, 'rb') as f:
        results = pickle.load(f)

    cls_result_lst = []
    for i, result in enumerate(results):
        img = imgs[i]
        img_name = img.replace('\\', '/').split('/')[-1]
        print('processing {}'.format(img_name))
        if size_file is not None:
            with open(size_file, 'r') as f:
                size_dict = json.load(f)
            if img_name not in size_dict.keys():
                print('{} has no size data!')
                s = 0
            else:
                s = int(size_dict[img_name])
        else:
            s = None
        main_code, bbox, score, img = default_rule(result, img_path, img_name, config_file, code_file, save_img, size=s)
        print(main_code, bbox, score)
        cls_result_lst.append({'image name': img_name, 'pred code': main_code, 'defect score': score})
        if save_img:
            cv2.imwrite(os.path.join(img_save_path, img_name), img)

    cls_df = pd.DataFrame(cls_result_lst)
    cls_df.to_excel(output_file)


if __name__ == '__main__':
    img_path = r'/data/sdv1/whtm/data/1GE02_v2/1GE02_train_test_data/test'
    pkl_file = r'/data/sdv1/whtm/result/1GE02/1GE02_v3_test.pkl'
    config_file = r'/data/sdv1/whtm/document/1GE02/G6_1GE02-V1.0.json'
    code_file = r'/data/sdv1/whtm/document/1GE02/G6_1GE02-V1.0.txt'
    output = r'/data/sdv1/whtm/result/1GE02/1GE02_v3_test.xlsx'
    img_save_path = r'/data/sdv1/whtm/result/1GE02/1GE02_v3_test'
    size_file = r'/data/sdv1/whtm/document/1GE02/1GE02_img_size.json'
    if not os.path.exists(img_save_path):
        os.makedirs(img_save_path)

    inference_cls(pkl_file, img_path, config_file, code_file, output, img_save_path, save_img=True, size_file=size_file)
