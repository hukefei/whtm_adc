#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, json, shutil
from tools.pascal_voc_io import PascalVocWriter

def load_json(filepath):
    f = open(filepath, 'r')
    data = json.load(f)
    img_names, total_d = [], []
    for d in data:
        d_ = {}
        for k,v in d.items():
            d_.setdefault(k, []).append(v)
        d.update(d_)

        filename = d['name'][0]
        if filename not in img_names:
            img_names.append(filename)
            total_d.append(d)
        else:
            inx = img_names.index(filename)
            total_d[inx]['defect_name'].extend(d['defect_name'])
            total_d[inx]['bbox'].extend(d['bbox'])
    f.close()
    return img_names, total_d


if __name__ == '__main__':
    jsonfile_1 = r"F:\guangdong\data\guangdong1_round1_train1_20190818\guangdong1_round1_train1_20190818\Annotations\anno_train.json"
    img_names_1, total_d_1 = load_json(jsonfile_1)

    img_names = img_names_1
    total_d = total_d_1

    foldername = r'F:\guangdong\data\guangdong1_round1_train1_20190818\guangdong1_round1_train1_20190818\defect_Images'
    save_dir = r'F:\guangdong\data\guangdong1_round1_train1_20190818\guangdong1_round1_train1_20190818\xmls'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        shutil.rmtree(save_dir)
        os.makedirs(save_dir)

    width = 2446
    height = 1000
    depth = 3
    imgSize = [height, width, depth]
    for i in range(len(img_names)):
        filename = img_names[i]
        print(i, filename)
        localImgPath = os.path.join(foldername, filename)
        XMLWriter = PascalVocWriter(foldername, filename, imgSize, localImgPath)
        cls = total_d[i]['defect_name']
        box = total_d[i]['bbox']
        categories = {
            '稀密档': 'ximidang', '粗经': 'cujing', '松经': 'songjing', '跳花': 'tiaohua', '浪纹档': 'langwendang',
            '三丝': 'sansi', '结头': 'jietou', '花板跳': 'huabantiao', '油渍': 'youzi', '粗维': 'cuwei', '星跳': 'xingtiao',
            '毛粒': 'maoli', '断氨纶': 'duananlun', '浆斑': 'jiangban', '断经': 'duanjing', '破洞': 'podong', '轧痕': 'yahen',
            '纬缩': 'weisuo', '百脚': 'baijiao', '吊经': 'diaojing', '污渍': 'wuzi', '修痕': 'xiuhen', '筘路': 'koulu',
            '磨痕': 'mohen', '死皱': 'sizhou', '烧毛痕': 'shaomaohen', '水渍': 'shuizi', '双纬': 'shuangwei',
            '云织': 'yunzhi', '双经': 'shuangjing', '整经结': 'zhengjingjie', '纬纱不良': 'weishabuliang',
            '色差档': 'sechadang', '跳纱': 'tiaosha'
        }
        for j in range(len(box)):
            # XMLWriter.addBndBox(box[j][0], box[j][1], box[j][2], box[j][3], cls[j])
            XMLWriter.addBndBox(int(round(box[j][0],0)), int(round(box[j][1],0)), int(round(box[j][2],0)), int(round(box[j][3],0)), categories[cls[j]])
        XMLWriter.save(os.path.join(save_dir, filename[:-4]+'.xml'))