import os
import json
import numpy as np
import xml.etree.ElementTree as ET
from utils.Code_dictionary import CodeDictionary
import random



START_IMAGE_ID = 0
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

def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.' % (name, root.tag))
    if length > 0 and len(vars) != length:
        raise NotImplementedError('The size of %s is supposed to be %d, but is %d.' % (name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars

def get(root, name):
    vars = root.findall(name)
    return vars

def convert(name_list, xml_dir, save_json, code_dictionary, ratio=1):
    json_dict = {"images": [], "type": "instances", "annotations": [], "categories": []}
    categories = code_dictionary.code_dict
    bnd_id = START_BOUNDING_BOX_ID
    image_id = START_IMAGE_ID
    if ratio != 1:
        name_list = random.sample(name_list, int(ratio * len(name_list)))

    for name in name_list:
        name = name.strip()
        image_id += 1
        checked = False
        print("Processing {}".format(name))
        for p in ('WHITE', 'RED', 'GREEN', 'BLUE'):
            name_p = name.replace('WHITE', p)
            xml_f = os.path.join(xml_dir, name_p + '.xml')
            if not os.path.exists(os.path.join(xml_dir, xml_f)):
                continue
            tree = ET.parse(xml_f)
            root = tree.getroot()

            if not checked:
                size = get_and_check(root, 'size', 1)
                width = int(get_and_check(size, 'width', 1).text)
                height = int(get_and_check(size, 'height', 1).text)
                image = {'file_name': name + '.jpg', 'height': height, 'width': width, 'id': image_id}
                json_dict['images'].append(image)
                checked = True


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

    """
    for cate, cid in categories.items():
        cat = {'supercategory': 'none', 'id': cid, 'name': cate}
        json_dict['categories'].append(cat)
    """
    for cid, cate in categories.items():
        cat = {'supercategory': 'none', 'id': cate, 'name': cid}
        json_dict['categories'].append(cat)

    with open(save_json, 'w') as json_fp:
        json_str = json.dumps(json_dict, indent=4, cls=MyEncoder)
        json_fp.write(json_str)


if __name__ == '__main__':
    img_dir = r'E:\diandengji\final dataset\choose_images'
    ann_dir = r'E:\diandengji\final dataset\annotations'
    name_lst = []
    for root, _, files in os.walk(img_dir):
        for file in files:
            if file.endswith('jpg') or file.endswith('JPG'):
                if 'WHITE' not in file:
                    continue
                name_lst.append(file[:-4])

    json_file = r'E:\diandengji\final dataset\val.json'
    code_file = r'E:\diandengji\final dataset\documents\classes.txt'
    code = CodeDictionary(code_file)

    convert(name_lst, ann_dir, json_file, code, ratio=0.2)