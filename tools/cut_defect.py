# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 16:02:56 2019

@author: hkf
"""

import xml.etree.ElementTree as ET
import os
import cv2





def cut_defect(images, annotations, output_dir):
    for root_dir, _, files in os.walk(images):
        for file in files:
            img_path = os.path.join(root_dir, file)
            xml = file.replace('jpg', 'xml')
            xml_path = os.path.join(annotations, xml)
            tree = ET.parse(xml_path)
            root = tree.getroot()
            objs = root.findall('object')
            img = cv2.imread(img_path)
            start_id = 0
            for obj in objs:
                category = obj[0].text
                bbox = [int(obj[4][i].text) for i in range(4)]
                cut = img[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
                save_dir_code = os.path.join(output_dir, category)
                if not os.path.exists(save_dir_code):
                    os.makedirs(save_dir_code)
                cut_name = file[:-4] + category + str(start_id) + '.jpg'
                start_id += 1
                cut_path = os.path.join(save_dir_code, cut_name)
                cv2.imwrite(cut_path, cut)

if __name__ == '__main__':
    defect_dir = r'D:\Project\WHTM\data\21101\label_data'
    xml_dir = r'D:\Project\WHTM\data\21101\annotations'
    save_dir = r'D:\Project\WHTM\data\21101\cut_out_defect'

    cut_defect(defect_dir, xml_dir, save_dir)

    IMG_WIDTH = 2448
    IMG_HEIGHT = 2056


