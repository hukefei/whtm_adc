import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import glob
import xml.etree.ElementTree as ET
from utils.Code_dictionary import CodeDictionary
import json
import cv2
from collections import OrderedDict


class Explorer():
    def __init__(self, json_file, img_dir, classes):
        self.code_dict = CodeDictionary(classes)
        self.codes = self.code_dict.code_id_lst
        self.img_dir = img_dir
        print(self.codes)
        self.ann_df = self.get_annotations(json_file)
        print(self.ann_df)

    def get_annotations(self, json_file):
        with open(json_file) as fp:
            json_dict = json.load(fp)
        result_lst = []
        img2id_dict = {}
        id2img_dict = {}
        img_lst = json_dict['images']
        for img in img_lst:
            image_name = img.get('file_name')
            height = img.get('height')
            width = img.get('width')
            id = img.get('id')
            img2id_dict[image_name] = {'id': id, 'width': width, 'height': height}
            id2img_dict[id] = {'image_name': image_name, 'width': width, 'height': height}

        ann_lst = json_dict['annotations']
        for ann in ann_lst:
            image_id = ann['image_id']
            image_name = id2img_dict[image_id]['image_name']
            width = id2img_dict[image_id]['width']
            height = id2img_dict[image_id]['height']
            score = ann.get('score', 1)
            if (width is None) or (height is None):
                image_path = os.path.join(self.img_dir, image_name)
                img_fp = cv2.imread(image_path)
                height, width, _ = img_fp.shape

            xmin = ann['bbox'][0]
            ymin = ann['bbox'][1]
            w = ann['bbox'][2]
            h = ann['bbox'][3]
            xmax = xmin + w
            ymax = ymin + h
            category_id = str(int(ann['category_id']))
            center_x = (xmin + xmax) / 2
            center_y = (ymin + ymax) / 2
            size = w * h
            result_lst.append(
                {'image': image_id, 'category_id': category_id, 'xmin': xmin,
                 'ymin': ymin, 'xmax': xmax, 'ymax': ymax, 'width': width, 'height': height, 'w': w, 'h': h,
                 'size': size, 'center_x': center_x, 'center_y': center_y, 'score': score})

        # for img in img_lst:
        #     img_name = img.replace('\\', '/').split('/')[-1]
        #     xml_name = img_name[:-3] + 'xml'
        #     xmL_path = os.path.join(ann_dir, xml_name)
        #     if xmL_path not in xml_lst:
        #         print('image {} has no corresponding annotation file'.format(img_name))
        #         continue
        #     try:
        #         tree = ET.parse(xmL_path)
        #     except Exception as e:
        #         print(xml_name, e)
        #         print('{} file error!'.format(xml_name))
        #         continue
        #     root = tree.getroot()
        #     size = root[4]
        #     width = int(size[0].text)
        #     height = int(size[1].text)
        #     depth = int(size[2].text)
        #
        #     objects = root.findall('object')
        #     defect_count = len(objects)
        #     code_id = 0
        #     for obj in objects:
        #         category = obj[0].text
        #         category_id = self.code_dict.code2id(category)
        #         xmin = float(obj[4][0].text)
        #         ymin = float(obj[4][1].text)
        #         xmax = float(obj[4][2].text)
        #         ymax = float(obj[4][3].text)
        #         w = xmax - xmin
        #         h = ymax - ymin
        #         result_lst.append(
        #             {'image': img_name, 'id': code_id, 'category': category, 'category_id': category_id, 'xmin': xmin,
        #              'ymin': ymin, 'ymin': ymin, 'ymax': ymax, 'width': width, 'height': height, 'w': w, 'h': h,
        #              'size': w * h, 'center_x': (xmin + xmax) / 2, 'center_y': (ymin + ymax) / 2,
        #              'defect count': defect_count})
        #         code_id += 1

        df = pd.DataFrame(result_lst)
        # df.set_index(['image', 'id'], inplace=True)

        return df

    def plot_center(self, mode='original', with_size=False, default_size=8, size_meaning='bbox'):
        assert mode in ['original', 'normalize']
        assert size_meaning in ['bbox', 'score']

        MIN_SCORE_SIZE = 3
        MAX_SCORE_SIZE = 33

        if mode == 'normalize':
            x_dict = {}
            y_dict = {}
            size_dict = {}
            for _, row in self.ann_df.iterrows():
                center_x = row['center_x']
                center_y = row['center_y']
                category_id = str(row['category_id'])
                width = row['width']
                height = row['height']
                center_x = center_x / width
                center_y = center_y / height
                score = row['score']
                if size_meaning == 'bbox':
                    size = row['size'] / 5000
                elif size_meaning == 'score':
                    size = MIN_SCORE_SIZE + (MAX_SCORE_SIZE - MIN_SCORE_SIZE) * score

                if not with_size:
                    size = default_size

                x_dict.setdefault(category_id, []).append(center_x)
                y_dict.setdefault(category_id, []).append(center_y)
                size_dict.setdefault(category_id, []).append(size)

            fig = plt.figure()
            for id in self.codes:
                plt.scatter(x_dict[id], y_dict[id], size_dict[id], cmap=id, label=id)
            legend = plt.legend(loc='upper left', shadow=False)
            plt.show()

        elif mode == 'original':
            x_dict = {}
            y_dict = {}
            size_dict = {}
            img_size_lst = []

            for _, row in self.ann_df.iterrows():
                center_x = row['center_x']
                center_y = row['center_y']
                category_id = str(row['category_id'])
                width = row['width']
                height = row['height']
                img_size = str(width) + '_' + str(height)
                img_size_lst.append(img_size)
                score = row['score']
                if size_meaning == 'bbox':
                    size = row['size'] / 5000
                elif size_meaning == 'score':
                    size = MIN_SCORE_SIZE + (MAX_SCORE_SIZE - MIN_SCORE_SIZE) * score
                if not with_size:
                    size = default_size

                x_dict.setdefault(img_size, {})
                x_dict[img_size].setdefault(category_id, []).append(center_x)

                y_dict.setdefault(img_size, {})
                y_dict[img_size].setdefault(category_id, []).append(center_y)

                size_dict.setdefault(img_size, {})
                size_dict[img_size].setdefault(category_id, []).append(size)

            img_size_lst = list(set(img_size_lst))
            for img_size in img_size_lst:
                fig = plt.figure()
                for id in self.codes:
                    x_lst = x_dict[img_size].get(id)
                    y_lst = y_dict[img_size].get(id)
                    size_lst = size_dict[img_size].get(id)
                    if x_lst is not None:
                        plt.scatter(x_lst, y_lst, size_lst, cmap=id, label=id)
                legend = plt.legend(loc='upper left', shadow=False)
                plt.show()

    def plot_category_count_bar(self):
        category_count = OrderedDict()
        for idx, row in self.ann_df.iterrows():
            category_id = str(row['category_id'])
            category_count.setdefault(category_id, 0)
            category_count[category_id] += 1
        print(category_count)

        category_lst = []
        count_lst = []
        for k in sorted(category_count.keys(), key=lambda x:int(x)):
            v = category_count[k]
            category_lst.append(k)
            count_lst.append(v)

        fig = plt.figure()
        plt.bar(x=category_lst, height=count_lst, label='category count')
        for x, y in enumerate(count_lst):
            plt.text(x, y, '{}'.format(y), ha='center', va='bottom')
        plt.show()

    def plot_bbox(self):
        bbox_dict = {}
        for idx, row in self.ann_df.iterrows():
            category_id = str(row['category_id'])
            xmin = row['xmin']
            ymin = row['ymin']
            xmax = row['xmax']
            ymax = row['ymax']
            w = row['w']
            h = row['h']
            width = row['width']
            height = row['height']
            score = row.get('score', 1)

            bbox_dict.setdefault(category_id, []).append({'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax,
                                                          'w': w, 'h': h, 'score': score, 'width': width,
                                                          'height': height})
        for k, v in bbox_dict.items():
            fig = plt.figure()
            plt.title(k)
            for bbox in v:
                rect = plt.Rectangle((bbox['xmin'] / bbox['width'], bbox['ymin'] / bbox['height']),
                                     bbox['w'] / bbox['width'], bbox['h'] / bbox['height'], fill=False, linewidth=1,
                                     edgecolor='r')
                plt.gca().add_patch(rect)
            plt.show()


if __name__ == '__main__':
    img_dir = r'E:\szl\MASK\demo_dataset\trainval\val'
    # ann_dir = r'D:\Project\chongqing_contest\data\chongqing1_round1_train1_20191223\defect_xmls'
    json_file = r'E:\szl\MASK\demo_dataset\trainval\val.json'
    classes_file = r'E:\szl\MASK\demo_dataset\trainval\classes.txt'
    # id_file = r'D:\Project\chongqing_contest\data\chongqing1_round1_train1_20191223\train_test_dataset\bottom\id.txt'
    # id_file = r'D:\Project\chongqing_contest\data\chongqing1_round1_train1_20191223\top\id.txt'
    explorer = Explorer(json_file, img_dir, classes_file)
    explorer.plot_center(mode='original', with_size=True, size_meaning='score')
    explorer.plot_category_count_bar()
    # explorer.plot_bbox()
