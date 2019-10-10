#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import shutil
import numpy as np
import xml.etree.ElementTree as ET
import random
from utils.Code_dictionary import CodeDictionary

random.seed(54321)

START_IMAGE_ID = 1
START_BOUNDING_BOX_ID = 1


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def split_dataset(xml_dir, train_percent):
    total_xml = os.listdir(xml_dir)
    num = len(total_xml)
    train = random.sample(range(num), int(num * train_percent))
    ftrain, ftest = [], []
    for i in range(num):
        name = total_xml[i][:-4] + '\n'
        if i in train:
            ftrain.append(name)
        else:
            ftest.append(name)
    return ftrain, ftest


def get(root, name):
    vars = root.findall(name)
    return vars


def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.' % (name, root.tag))
    if length > 0 and len(vars) != length:
        raise NotImplementedError('The size of %s is supposed to be %d, but is %d.' % (name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars


def convert(name_list, xml_dir, img_dir, save_img, save_json, code_dictionary):
    json_dict = {"images": [], "type": "instances", "annotations": [], "categories": []}
    categories = code_dictionary.code_dict
    bnd_id = START_BOUNDING_BOX_ID
    image_id = START_IMAGE_ID

    for name in name_list:
        name = name.strip()
        print("Processing %s" % name)
        xml_f = os.path.join(xml_dir, name + '.xml')
        tree = ET.parse(xml_f)
        root = tree.getroot()

        size = get_and_check(root, 'size', 1)
        width = int(get_and_check(size, 'width', 1).text)
        height = int(get_and_check(size, 'height', 1).text)
        image = {'file_name': name + '.jpg', 'height': height, 'width': width, 'id': image_id}
        json_dict['images'].append(image)

        for obj in get(root, 'object'):
            category = get_and_check(obj, 'name', 1).text
            category_id = categories[category]

            bndbox = get_and_check(obj, 'bndbox', 1)
            xmin = float(get_and_check(bndbox, 'xmin', 1).text)
            ymin = float(get_and_check(bndbox, 'ymin', 1).text)
            xmax = float(get_and_check(bndbox, 'xmax', 1).text)
            ymax = float(get_and_check(bndbox, 'ymax', 1).text)
            assert (xmax > xmin)
            assert (ymax > ymin)
            o_width = abs(xmax - xmin)
            o_height = abs(ymax - ymin)
            ann = {'area': o_width * o_height, 'iscrowd': 0, 'image_id': image_id,
                   'bbox': [xmin, ymin, o_width, o_height],
                   'category_id': category_id, 'id': bnd_id, 'ignore': 0,
                   'segmentation': []}
            json_dict['annotations'].append(ann)
            bnd_id += 1
        shutil.copy(os.path.join(img_dir, name + '.jpg'), os.path.join(save_img, name + '.jpg'))
        image_id += 1
    """
    for cate, cid in categories.items():
        cat = {'supercategory': 'none', 'id': cid, 'name': cate}
        json_dict['categories'].append(cat)
    """
    for cid, cate in categories.items():
        cat = {'supercategory': 'none', 'id': cate, 'name': cid}
        json_dict['categories'].append(cat)

    json_fp = open(save_json, 'w')
    json_str = json.dumps(json_dict, indent=4, cls=MyEncoder)
    json_fp.write(json_str)
    json_fp.close()


if __name__ == '__main__':
    xml_dir = r'D:\Project\WHTM\data\21101\final_dataset\annotations'
    img_dir = r'D:\Project\WHTM\data\21101\final_dataset\images'

    save_dir = r'D:\Project\WHTM\data\21101\train_test_data'
    code_file = r'D:\Project\WHTM\code\21101code.xlsx'
    code = CodeDictionary(code_file)

    train_json = os.path.join(save_dir, 'train.json')
    test_json = os.path.join(save_dir, 'test.json')
    train_img = os.path.join(save_dir, 'train')
    test_img = os.path.join(save_dir, 'test')
    for dir in [train_img, test_img]:
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.makedirs(dir)

    train, test = split_dataset(xml_dir, 0.9)
    convert(train, xml_dir, img_dir, train_img, train_json, code)
    convert(test, xml_dir, img_dir, test_img, test_json, code)
