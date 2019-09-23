#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pip install lxml

import sys
import os
import cv2
import json
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split, RepeatedKFold
import shutil
import argparse
import numpy as np


def joinpath(lst):
    root = lst[0]
    if len(lst) == 1:
        return root
    for item in lst[1:]:
        root = os.path.join(root, item)
    return root


def generate_txt(path):
    filename = os.path.join(path, 'img_list.txt')
    f = open(filename, 'w')

    for root, dirs, files in os.walk(path):
        for file in files:
            if file.split('.')[-1] not in ('JPG', 'jpg'):
                continue
            filepath = os.path.join(root, file)
            last_dir = path.split('\\')[-1]
            lst = filepath.split('\\')
            idx = lst.index(last_dir)
            lst1 = lst[idx + 1:]
            filepath = joinpath(lst1)
            filepath = filepath.replace('\\', '/')
            f.write(filepath)
            f.write('\n')
    f.close()


fww1030_image_id = 10000000
fww1030_bounding_box_id = 10000000


def get_current_image_id():
    global fww1030_image_id
    fww1030_image_id += 1
    return fww1030_image_id


def get_current_annotation_id():
    global fww1030_bounding_box_id
    fww1030_bounding_box_id += 1
    return fww1030_bounding_box_id


def get(root, name):
    vars = root.findall(name)
    return vars


def get_and_check(root, name, length):
    var_lst = root.findall(name)
    if len(var_lst) == 0:
        raise NotImplementedError('Can not find %s in %s.' % (name, root.tag))
    if length > 0 and len(var_lst) != length:
        raise NotImplementedError('The size of %s is supposed to be %d, but is %d.' % (
            name, length, len(var_lst)))
    if length == 1:
        var_lst = var_lst[0]
    return var_lst


def get_labels(xml_list, path, ignored_lst=[], ignored_thr=100):
    categories = dict()
    for line in xml_list:
        try:
            line = os.path.join(path, line)
            tree = ET.parse(line)
            root = tree.getroot()
            for obj in get(root, 'object'):
                category = get_and_check(obj, 'name', 1).text
                if category not in categories:
                    categories[category] = 1
                else:
                    categories[category] += 1
        except:
            print('Error : {}'.format(line))

    filted_labels = []
    for k, v in categories.items():
        print('** {} **  -> {}'.format(k, v))
        if v > ignored_thr and k not in ignored_lst:
            filted_labels.append(k)
    filted_labels = sorted(filted_labels)
    return filted_labels


def filter_image_label_path(jpg_list, path):
    filted_jpg_xml = []
    for jpg in jpg_list:
        xml = os.path.splitext(jpg)[0] + '.xml'
        xml = xml.replace('images', 'labels')
        if os.path.exists(os.path.join(path, jpg)) and os.path.exists(os.path.join(path, xml)):
            filted_jpg_xml.append((jpg, xml))
        else:
            print('NOT FIND -> {}'.format(jpg))
    return filted_jpg_xml


def cp_jpg_xml(root, jpg, xml):
    dst_jpg = os.path.join(root, jpg)
    dst_xml = os.path.join(root, xml)
    if not os.path.exists(os.path.dirname(dst_jpg)):
        os.makedirs(os.path.dirname(dst_jpg))
    if not os.path.exists(os.path.dirname(dst_xml)):
        os.makedirs(os.path.dirname(dst_xml))
    shutil.copyfile(jpg, dst_jpg)
    shutil.copyfile(xml, dst_xml)


def convert(jpg_xml, categories, json_file, path):
    json_dict = {"images": [], "type": "instances", "annotations": [],
                 "categories": []}

    for jpg_file, xml_file in jpg_xml:
        xml_file = os.path.join(path, xml_file)
        # assert(jpg_file[:-4] == xml_file[:-4])

        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            image_id = get_current_image_id()
            size = get_and_check(root, 'size', 1)
            width = int(get_and_check(size, 'width', 1).text)
            height = int(get_and_check(size, 'height', 1).text)

            # 构造image
            image = {'file_name': jpg_file, 'height': height, 'width': width,
                     'id': image_id}
            json_dict['images'].append(image)

            for obj in get(root, 'object'):
                category = get_and_check(obj, 'name', 1).text

                if category == 'w':
                    print(jpg_file)
                    cp_jpg_xml('w_error', jpg_file, xml_file)
                    break

                if category not in categories:
                    print('skip annotation {}'.format(category))
                    continue

                category_id = categories.index(category) + 1

                bndbox = get_and_check(obj, 'bndbox', 1)
                xmin = float(get_and_check(bndbox, 'xmin', 1).text)
                ymin = float(get_and_check(bndbox, 'ymin', 1).text)
                xmax = float(get_and_check(bndbox, 'xmax', 1).text)
                ymax = float(get_and_check(bndbox, 'ymax', 1).text)
                if (xmax <= xmin) or (ymax <= ymin):
                    print('{} error'.format(xml_file))
                    continue

                o_width = (xmax - xmin)
                o_height = (ymax - ymin)
                ann = {'area': o_width * o_height, 'iscrowd': 0, 'image_id':
                    image_id, 'bbox': [xmin, ymin, o_width, o_height],
                       'category_id': category_id, 'id': get_current_annotation_id(), 'ignore': 0,
                       'segmentation': []}
                json_dict['annotations'].append(ann)
        except NotImplementedError:
            # print('xml {} file error!'.format(jpg_file))
            cp_jpg_xml('format_error', jpg_file, xml_file)

    for cid, cate in enumerate(categories):
        cat = {'supercategory': 'FWW', 'id': cid + 1, 'name': cate}
        json_dict['categories'].append(cat)

    json_file = os.path.join(path, json_file)
    json_fp = open(json_file, 'w')
    json.dump(json_dict, json_fp, indent=4)
    json_fp.close()


def write_categories_2_file(categories, filename):
    if not os.path.exists(filename):
        with open(filename, 'w') as fp:
            fp.writelines([item + '\n' for item in categories])


def convert_oks_list(oks_list, categories, json_file):
    json_dict = {"images": [], "type": "instances", "annotations": [],
                 "categories": []}

    roi = cv2.imread('./template.png')

    for jpg_file in oks_list:
        current = cv2.imread(jpg_file)
        image_id = get_current_image_id()
        width = current.shape[1]
        height = current.shape[0]

        # 构造image
        image = {'file_name': jpg_file, 'height': height, 'width': width,
                 'id': image_id}
        json_dict['images'].append(image)

        for _ in range(1):
            category = 'STR05'

            if category not in categories:
                print('skip annotation {}'.format(category))
                continue

            category_id = categories.index(category) + 1

            xmin = 100
            ymin = 100
            xmax = xmin + roi.shape[1]
            ymax = ymin + roi.shape[0]
            if (xmax <= xmin) or (ymax <= ymin):
                print('{} error'.format(jpg_file))

            current[ymin: ymax, xmin: xmax] = roi

            o_width = (xmax - xmin)
            o_height = (ymax - ymin)
            ann = {'area': o_width * o_height, 'iscrowd': 0, 'image_id':
                image_id, 'bbox': [xmin, ymin, o_width, o_height],
                   'category_id': category_id, 'id': get_current_annotation_id(), 'ignore': 0,
                   'segmentation': []}
            json_dict['annotations'].append(ann)
        cv2.imwrite(jpg_file, current)

    for cid, cate in enumerate(categories):
        cat = {'supercategory': 'FWW', 'id': cid + 1, 'name': cate}
        json_dict['categories'].append(cat)

    json_fp = open(json_file, 'w')
    json.dump(json_dict, json_fp, indent=4)
    json_fp.close()


def parse_args():
    parser = argparse.ArgumentParser(description='inference and evaluate test datasets')
    parser.add_argument('path', help='image path')
    parser.add_argument('--ignored', help='ignored categories, seperated by ,')
    parser.add_argument('--test_size', default=0.1, type=float, help='test set size')
    parser.add_argument('--kfold', type=int, help='KFold Numbers')
    parser.add_argument('--repeats',default=3, type=int, help='repeat times for kfold')
    parser.add_argument('--ignored_thr', default=100, type=int, help='threshold for ignoring annotations')
    args = parser.parse_args()

    return args


def main(random_state=888):
    args = parse_args()
    if args.ignored is not None:
        ignored = args.ignored.split(',')
    else:
        ignored = []
    test_size = args.test_size
    path = args.path
    kfold = args.kfold
    repeats = args.repeats
    ignored_thr = args.ignored_thr

    generate_txt(path)
    list_fp = open(os.path.join(path, 'img_list.txt'), 'r')
    jpg_list = [item.strip() for item in list_fp]
    jpg_xml = filter_image_label_path(jpg_list, path)
    print('\nnum samples -> ** {} **\n'.format(len(jpg_xml)))

    categories = get_labels([item[1] for item in jpg_xml], path, ignored, ignored_thr)
    print(categories)

    X = [item[0] for item in jpg_xml]
    Y = [item[1] for item in jpg_xml]
    labels = [item.split('/')[0] for item in Y]
    if kfold is None:
        train_x, test_x, train_y, test_y = train_test_split(
            X, Y, test_size=test_size, random_state=random_state, stratify=labels)
        train_samples = [(jpg, xml) for jpg, xml in zip(train_x, train_y)]
        test_samples = [(jpg, xml) for jpg, xml in zip(test_x, test_y)]

        convert(train_samples, categories, 'train.json', path)
        convert(test_samples, categories, 'test.json', path)
    else:
        kf = RepeatedKFold(n_repeats=repeats, n_splits=kfold, random_state=random_state)
        count = 0
        for train_idx, test_idx in kf.split(X):
            count += 1
            print(count)
            train_x, train_y = np.array(X)[train_idx], np.array(Y)[train_idx]
            test_x, test_y = np.array(X)[test_idx], np.array(Y)[test_idx]
            train_samples = [(jpg, xml) for jpg, xml in zip(train_x, train_y)]
            test_samples = [(jpg, xml) for jpg, xml in zip(test_x, test_y)]

            convert(train_samples, categories, 'train_{}.json'.format(count), path)
            convert(test_samples, categories, 'test_{}.json'.format(count), path)

    # write_categories_2_file(categories, 'labels.txt')

    list_fp.close()


if __name__ == '__main__':
    # path = 'D:\XMTM\data\11203\11203label'
    main()
