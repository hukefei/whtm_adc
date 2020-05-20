import os
import cv2
from xml.etree import ElementTree as ET
import glob
import matplotlib.pyplot as plt
import numpy as np


def add_pixels(img, ann):
    img_ = cv2.imread(img)
    tree = ET.parse(ann)
    root = tree.getroot()
    objs = root.findall('object')
    bboxes = []
    defect = []
    for obj in objs:
        # print(obj[0].text)
        if obj[0].text == 'pixel':
            bbox = [obj[4][0].text, obj[4][1].text, obj[4][2].text, obj[4][3].text]
            bbox = [int(b) for b in bbox]
            bboxes.append(bbox)
    if len(bboxes) == 2:
        print('2:', img)
        return None
    if len(bboxes) == 0:
        print('0:', img)
        return tree

    for obj in objs:
        # print(obj[0].text)
        if obj[0].text != 'pixel':
            bbox = [obj[4][0].text, obj[4][1].text, obj[4][2].text, obj[4][3].text]
            bbox = [int(b) for b in bbox]
            defect.append(bbox)
    # print(defect)
    bbox = bboxes[0]
    obj_ = img_[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    lt = [bbox[0], bbox[1]]
    # print('lt:', lt)
    w, h, c = obj_.shape

    res = cv2.matchTemplate(obj_, img_, method=cv2.TM_CCOEFF_NORMED)
    thr, dst = cv2.threshold(res, 0.8, 255, type=cv2.THRESH_BINARY)
    dst = np.uint8(dst)

    nccomps = cv2.connectedComponentsWithStats(dst)
    _ = nccomps[0]
    labels = nccomps[1]
    status = nccomps[2]
    centroids = nccomps[3]

    centroids = centroids.astype(int)

    error = 10
    for c in centroids:
        # print(c)
        if c[0] >= lt[0] - error and c[0] <= lt[0] + error and c[1] >= lt[1] - error and c[1] <= lt[1] + error:
            # print(c)
            continue
        flag = False
        for d in defect:
            overlap = [max(d[0], c[0]), max(d[1], c[1]), min(d[2], c[0] + h), min(d[3], c[1] + w)]
            if (overlap[2] - overlap[0]) > 0 and (overlap[3] - overlap[1]) > 0:
                # print(c)
                flag = True
                continue
        if flag:
            continue

        obj_element = ET.SubElement(root, 'object')
        name = ET.SubElement(obj_element, 'name')
        name.text = 'pixel'
        pose = ET.SubElement(obj_element, 'pose')
        pose.text = 'Unspecified'
        t = ET.SubElement(obj_element, 'truncated')
        t.text = '0'
        d = ET.SubElement(obj_element, 'difficult')
        d.text = '0'
        bndbox = ET.SubElement(obj_element, 'bndbox')
        xmin = ET.SubElement(bndbox, 'xmin')
        xmin.text = str(c[0])
        ymin = ET.SubElement(bndbox, 'ymin')
        ymin.text = str(c[1])
        xmax = ET.SubElement(bndbox, 'xmax')
        xmax.text = str(c[0] + h)
        ymax = ET.SubElement(bndbox, 'ymax')
        ymax.text = str(c[1] + w)

    return tree


img_dir = r'F:\WHTM\MASK\data\0514\0516\val'
ann_dir = r'F:\WHTM\MASK\data\0514\0516\val'
save_dir = r'F:\WHTM\MASK\data\0514\add_pixels\val'

images = glob.glob(os.path.join(img_dir, '*.jpg'))
annotations = glob.glob(os.path.join(ann_dir, '*.xml'))

for img, ann in zip(*(images, annotations)):
    tree = add_pixels(img, ann)
    ann_name = ann.replace('\\', '/').split('/')[-1]
    if tree is not None:
        tree.write(os.path.join(save_dir, ann_name))
