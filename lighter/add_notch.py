import cv2
import numpy as np
import pandas as pd
import os
from xml.etree import ElementTree as ET
from tools.pascal_voc_io import PascalVocWriter


def add_notch(img_dir, img_name, xml_dir, xml_save_dir):
    img_file = os.path.join(img_dir, img_name)
    img = cv2.imread(img_file)
    h, w, d = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.imwrite("gray.jpg", gray)
    ret, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

    #cv2.imwrite("binary.jpg", binary)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
    #cv2.putText(img, "{:.3f}".format(len(contours)), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 1)

    best = 0
    max_len = 0
    for idx, c in enumerate(contours):
        if len(c) > max_len:
            max_len = len(c)
            best = c
    contours_x = []
    contours_y = []
    for p in best:
        x = p[0][0]
        y = p[0][1]
        if x > 30 and x < w - 30:
            contours_x.append(x)
        if y > 30 and y < h - 30:
            contours_y.append(y)

    notch_center = (max(contours_x), (min(contours_y) + max(contours_y)) / 2)

    notch_x_min = int(notch_center[0] - 50)
    notch_x_max = int(notch_center[0])
    notch_y_min = int(notch_center[1] - 50)
    notch_y_max = int(notch_center[1] + 50)
    print(notch_x_min, notch_y_min, notch_x_max, notch_y_max)

    # read xml file
    xml_name = img_name[:-3] + 'xml'
    xml_file = os.path.join(xml_dir, xml_name)
    xml_writer = PascalVocWriter('', img_name, [h, w, d])
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        objs = root.findall('object')
        for obj in objs:
            name = obj[0].text
            xmin = obj[4][0].text
            ymin = obj[4][1].text
            xmax = obj[4][2].text
            ymax = obj[4][3].text
            xml_writer.addBndBox(xmin, ymin, xmax, ymax, name)
            if name == 'notch':
                print('{} already has notch label'.format(img_name))
                return 0
    except Exception as e:
        pass
    xml_writer.addBndBox(notch_x_min, notch_y_min, notch_x_max, notch_y_max, 'notch')
    xml_writer.save(os.path.join(xml_save_dir, xml_name))


if __name__ == '__main__':
    img_dir = r'E:\diandengji\final dataset\images'
    #img_name = r'W93VP2827B0512_WHITE_20190730.jpg'
    xml_dir = r'E:\diandengji\final dataset\annotations'
    xml_save = r'E:\diandengji\final dataset\add_notch_save'

    for img in os.listdir(img_dir):
        if 'WHITE' not in img:
            continue
        try:
            add_notch(img_dir, img, xml_dir, xml_save)
        except Exception as e:
            print(img, e)
