import json
import pandas as pd
import numpy as np
import os

class code_dict():
    def __init__(self, code_file):
        self.code_df = pd.read_excel(code_file)
        assert 'CODE' in self.code_df.columns, 'CODE should be in the code file'
        assert 'ID' in self.code_df.columns, 'ID should be in the code file'

    def code2id(self, code_lst):
        if isinstance(code_lst, str):
            return self.code_df.loc[self.code_df['CODE'] == code_lst, 'ID'].values[0]
        if isinstance(code_lst, list):
            id_lst = []
            for code in code_lst:
                id_lst.append(self.code_df.loc[self.code_df['CODE'] == code, 'ID'].values[0])
            return id_lst

    def id2code(self, id_lst):
        if isinstance(id_lst, int):
            return self.code_df.loc[self.code_df['ID'] == id_lst, 'CODE'].values[0]
        if isinstance(id_lst, list):
            code_lst = []
            for id in id_lst:
                code_lst.append(self.code_df.loc[self.code_df['ID'] == id, 'CODE'].values[0])
            return code_lst

def prio_check(prio_lst, code_lst):
    idx_lst = []
    for code in code_lst:
        assert code in prio_lst, '{} should be in priority file'.format(code)
        idx = prio_lst.index(code)
        idx_lst.append(idx)
    final_code = prio_lst[min(idx_lst)][0]
    return final_code


def gt2cls(json_file,
           test_images,
           prio_file,
           code_dict_obj,
           output='ground_truth_result.xlsx'):
    with open(json_file) as f:
        json_dict = json.load(f)

    prio_df = pd.read_excel(prio_file)
    prio_lst = list(prio_df.values)

    img_lst = []
    img_dict = {}
    for d_ in json_dict['images']:
        file_name = d_['file_name']
        id = d_['id']
        img_lst.append(file_name)
        img_dict[file_name] = id

    annos = pd.DataFrame(json_dict['annotations'])

    gt_result = []
    json_img_lst = []
    for img in img_lst:
        img_id = img_dict[img]
        annos_per_img = annos[annos['image_id'] == img_id]
        assert len(annos_per_img) != 0, 'no gt bbox in image {}'.format(img)

        category_lst = list(annos_per_img['category_id'].values)
        category_lst = code_dict_obj.id2code(category_lst)
        gt_code = prio_check(prio_lst, category_lst)
        gt_result.append({'image name': img, 'true code': gt_code})
        json_img_lst.append(img)

    # add normal test images classification result
    for root, _, files in os.walk(test_images):
        for file in files:
            if (file.endswith('jpg') or file.endswith('JPG')) and (file not in img_lst):
                gt_result.append({'image name': file, 'true code': 'COM99'})

    gt_df = pd.DataFrame(gt_result)
    gt_df.to_excel(output)


if __name__ == '__main__':
    gt_json = r'D:\Project\WHTM\data\21101\train_test_data\test.json'
    prio_file = r'D:\Project\WHTM\data\21101\21101_prio.xlsx'
    test_images = r'D:\Project\WHTM\data\21101\train_test_data\test'
    code_file = r'D:\Project\WHTM\data\21101\21101_code_dict.xlsx'
    code = code_dict(code_file)
    gt2cls(gt_json, test_images, prio_file, code)
