from rule.rule_21101 import default_rule
import cv2
import os, glob
import pandas as pd
import pickle
import json
import shutil



def inference_cls(result_file,
                  imgs,
                  config,
                  code_file,
                  output,
                  img_save_path,
                  size_file=None,
                  save_img=False):
    if not os.path.exists(img_save_path):
        os.makedirs(img_save_path)

    with open(result_file, 'rb') as f:
        results = pickle.load(f)

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
                print('{} has no size data!')
                s = 0
            else:
                s = int(size_dict[img_name])
        else:
            s = None
        main_code, bbox, score, img_ = default_rule(result, img, img_name, config, code_file, save_img)
        print(main_code, bbox, score)

        if save_img:
            cv2.imwrite(os.path.join(img_save_path, img_name), img_)

        if main_code != gt_code:
            save_path = os.path.join(output, gt_code, main_code)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            shutil.copy(img, os.path.join(save_path, img_name))
        cls_result_lst.append({'image name': img_name, 'pred code': main_code, 'defect score': score, 'oic code': gt_code})


    cls_df = pd.DataFrame(cls_result_lst)
    cls_df.to_excel(os.path.join(output, 'result.xlsx'))



if __name__ == '__main__':
    img_path = r'/data/sdv1/whtm/ddj/data/0326test/diandengji_0319_add/'
    pkl_file = r'/data/sdv1/whtm/ddj/result/ovli_0331.pkl'
    json_file = r'/data/sdv1/whtm/ddj/data/0320/rule.json'
    code_file = r'/data/sdv1/whtm/ddj/data/0320/classes.txt'
    output = r'/data/sdv1/whtm/ddj/result/0331/ovli_0331.xlsx'
    img_save_path = r'/data/sdv1/whtm/ddj/result/0331/images'
    # size_file = r'/data/sdv1/whtm/document/1GE02/1GE02_bt1_img_size.json'
    size_file = None
    if not os.path.exists(img_save_path):
        os.makedirs(img_save_path)
    # open config file
    with open(json_file) as f:
        config = json.load(f)
    with open(code_file) as f:
        codes = f.read().splitlines()

    inference_cls(pkl_file, img_path, config, codes, output, img_save_path, save_img=True, size_file=size_file)
