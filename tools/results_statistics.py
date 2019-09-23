import json
import matplotlib.pyplot as plt
import numpy as np


def json_load(file):
    with open(file) as f:
        json_dict = json.load(f)
    return json_dict

def defect_analyse(json_dict):
    img_lst = []
    score_lst = []
    category_lst = []
    bbox_lst = []
    for d_ in json_dict:
        img_name = d_['name']
        category = d_['category']
        bbox = d_['bbox']
        score = d_['score']

        img_lst.append(img_name)
        score_lst.append(score)
        category_lst.append(category)
        bbox_lst.append(bbox)

    print(score_lst)
    image_data_analyse(img_lst)
    score_analyse(score_lst)
    bbox_analyse(bbox_lst)

def image_data_analyse(img_lst):
    result = {}
    for img in set(img_lst):
        result[img] = img_lst.count(img)
    img_num = len(result)
    print(img_num)
    defect_count = list(result.values())

    bin = np.linspace(0,30,9)
    n, bins, _ = plt.hist(defect_count)
    print(n, bins)
    plt.xlabel('defect_count')
    plt.ylabel('count')
    plt.title('defect number count per image')
    plt.show()


def score_analyse(score_lst):
    bin = np.linspace(0,1,41)
    n, bins, _ = plt.hist(score_lst, bins= bin)
    print(n, bins)
    plt.xlabel('score')
    plt.ylabel('count')
    plt.title('score distribution')
    plt.show()

def bbox_analyse(bbox_lst):
    w_lst = []
    h_lst = []
    print('{} bboxes'.format(len(bbox_lst)))
    for bbox in bbox_lst:
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        w_lst.append(w)
        h_lst.append(h)

    plt.scatter(w_lst, h_lst)
    plt.xlabel('width')
    plt.ylabel('height')
    plt.title('bbox size scatters')
    plt.show()




if __name__ == '__main__':
    json_file = r'F:\guangdong\result\results_gd_0820_1.json'
    json_dict = json_load(json_file)
    defect_analyse(json_dict)