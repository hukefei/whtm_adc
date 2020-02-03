from __future__ import division
import cv2
import math
import os
from PIL import Image
import xml.dom.minidom
import xml.etree.ElementTree as ET
import numpy as np
import copy
from tools.pascal_voc_io import PascalVocWriter

ImgPath = r'C:\Users\huker\Desktop\Desktop'
AnnoPath = r'C:\Users\huker\Desktop\Desktop'
ProcessedPath = r'C:\Users\huker\Desktop\hehe\save'

def check_in(bbox1, bbox2):
    '''
    check whether bbox2 in bbox1
    '''
    if bbox2[0]>=bbox1[0] and bbox2[1]>=bbox1[1] and bbox2[2]<=bbox1[2] and bbox2[3]<=bbox1[3]:
        return True
    else:
        return False

imagelist = []

CROP_WIDTH = 1200
CROP_HEIGHT = 1000

for path, d, filelist in os.walk(ImgPath):
    for filename in filelist:
        if filename.endswith('JPG') or filename.endswith('jpg'):
            imagelist.append(os.path.join(path, filename))

print(imagelist)
# img = cv2.imread(imagelist)
# print(img.shape)
# size_x ,size_y ,z = img.shape

for image in imagelist:

    img = cv2.imread(image)
    # print(img.shape)
    size_y, size_x, depth = img.shape

    image_pre, ext = os.path.splitext(image)

    # print(image_pre)

    imgfile = image

    x1s, x2s, y1s, y2s, n = 0, 0, 0, 0, 0

    # if not os.path.exists(AnnoPath + image_pre + '.xml'): continue
    # print(image_pre)
    xmlfile = image_pre + '.xml'
    # print(xmlfile)
    #DomTree = xml.dom.minidom.parse(xmlfile)
    tree = ET.parse(xmlfile)
    root = tree.getroot()

    folder = root[0].text
    file = root[1].text
    path = root[2].text
    print(file)

    objs = root.findall('object')

    k = 0

    obj_lst = []
    crop_lst = []
    for obj in objs:
        code = obj[0].text
        diff = obj[3].text
        bbox = obj[4]
        xmin = int(bbox[0].text)
        ymin = int(bbox[1].text)
        xmax = int(bbox[2].text)
        ymax = int(bbox[3].text)
        obj_lst.append((xmin, ymin, xmax, ymax, code, diff))



        x_c = int((xmin + xmax)/2)
        y_c = int((ymin + ymax)/2)

        crop_xmin = int(max(min(x_c - CROP_WIDTH/2, size_x - CROP_WIDTH), 0))
        crop_ymin = int(max(min(y_c - CROP_HEIGHT/2, size_y - CROP_HEIGHT), 0))
        crop_xmax = crop_xmin + CROP_WIDTH
        crop_ymax = crop_ymin + CROP_HEIGHT

        crop_lst.append((crop_xmin, crop_ymin, crop_xmax, crop_ymax))

        print(code, crop_xmin, crop_ymin, crop_xmax, crop_ymax)

    copy_obj = copy.deepcopy(obj_lst)
    for idx, crop in enumerate(crop_lst):
        crop_file_name = file[:-4] + '_{}.jpg'.format(idx)
        crop_xml_name = crop_file_name.replace('jpg', 'xml')
        XMLWriter = PascalVocWriter(folder, crop_file_name, [CROP_WIDTH, CROP_HEIGHT, depth], path)
        flag = 0
        crop_bbox = crop
        for i, obj in enumerate(copy_obj):
            obj_bbox = obj[:4]
            obj_code = obj[4]
            obj_diff = obj[5]

            savepath = os.path.join(ProcessedPath, obj_code)

            if not os.path.exists(savepath):
                os.makedirs(savepath)

            if check_in(crop_bbox, obj_bbox):
                XMLWriter.addBndBox(obj[0] - crop_bbox[0], obj[1] - crop_bbox[1], obj[2] - crop_bbox[0], obj[3] - crop_bbox[1], obj_code, difficult=int(obj_diff))
                copy_obj.pop(i)
                flag = 1
        if flag:
            crop_img = img[crop_bbox[1]:crop_bbox[3], crop_bbox[0]:crop_bbox[2], :]
            cv2.imwrite(os.path.join(savepath, crop_file_name), crop_img)
            XMLWriter.save(os.path.join(savepath, crop_xml_name))



    print('-'*50)



