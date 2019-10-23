from rule.rule_21101 import default_rule
import cv2
import os, glob
import pandas as pd
import pickle


def inference_cls(result_file,
                  img_path,
                  config_file,
                  code_file,
                  output_file,
                  img_save_path,
                  save_img=False):
    imgs = glob.glob(os.path.join(img_path, '*.jpg'))

    with open(result_file, 'rb') as f:
        results = pickle.load(f)

    cls_result_lst = []
    for i, result in enumerate(results):
        img = imgs[i]
        img_name = img.replace('\\', '/').split('/')[-1]
        print('processing {}'.format(img_name))
        main_code, bbox, score, img = default_rule(result, img_path, img_name, config_file, code_file, save_img)
        print(main_code)
        cls_result_lst.append({'image name': img_name, 'pred code': main_code, 'defect score': score})
        if save_img:
            cv2.imwrite(os.path.join(img_save_path, img_name), img)

    cls_df = pd.DataFrame(cls_result_lst)
    cls_df.to_excel(output_file)


if __name__ == '__main__':
    img_path = r'/data/sdv1/whtm/data/21101_bt/21101bt_part2_2000'
    pkl_file = r'/data/sdv1/whtm/result/21101/21101_v6_bt2.pkl'
    config_file = r'/data/sdv1/whtm/document/G6_21101-V1.0.json'
    code_file = r'/data/sdv1/whtm/document/G6_21101-V1.0.txt'
    output = r'/data/sdv1/whtm/result/21101/21101_test.xlsx'
    img_save_path = r'/data/sdv1/whtm/result/21101/21101_test'

    inference_cls(pkl_file, img_path, config_file, code_file, output, img_save_path)
