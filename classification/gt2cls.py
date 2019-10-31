import json
import pandas as pd
import os
from utils.Code_dictionary import CodeDictionary


def prio_check(prio_lst, code_lst):
    idx_lst = []
    for code in code_lst:
        assert code in prio_lst, '{} should be in priority file'.format(code)
        idx = prio_lst.index(code)
        idx_lst.append(idx)
    final_code = prio_lst[min(idx_lst)]
    return final_code


def gt2cls(json_file,
           test_images,
           prio_file,
           code_dict_obj,
           size_file=None,
           output='ground_truth_result.xlsx'):
    with open(json_file) as f:
        json_dict = json.load(f)

    with open(prio_file) as f:
        config = json.load(f)
    prio_order = config['prio_order']

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
        # get size
        if size_file is not None:
            with open(size_file) as f:
                size_dict = json.load(f)
            size = int(size_dict.get(img, 0))
        else:
            size = None
        img_id = img_dict[img]
        annos_per_img = annos[annos['image_id'] == img_id]
        assert len(annos_per_img) != 0, 'no gt bbox in image {}'.format(img)

        category_lst = list(annos_per_img['category_id'].values)
        category_lst = code_dict_obj.id2code(category_lst)

        if size is not None:
            category_lst = ['AZ21' if (i == 'PR') and (size >= 40) else i for i in category_lst]
            category_lst = ['0' if (i == 'PR') and (size < 40) else i for i in category_lst]
            print(size, category_lst)
        gt_code = prio_check(prio_order, category_lst)
        print(gt_code)
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
    gt_json = r'/data/sdv1/whtm/data/1GE02_v2/1GE02_train_test_data/test.json'
    prio_file = r'/data/sdv1/whtm/document/1GE02/G6_1GE02-V1.0.json'
    test_images = r'/data/sdv1/whtm/data/1GE02_v2/1GE02_train_test_data/test'
    code_file = r'/data/sdv1/whtm/document/1GE02/G6_1GE02-V1.0.txt'
    size_file = r'/data/sdv1/whtm/document/1GE02/1GE02_img_size.json'

    code = CodeDictionary(code_file)
    gt2cls(gt_json, test_images, prio_file, code, size_file=size_file)
