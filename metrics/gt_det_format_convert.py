import os
import json
import shutil




def make_save_dirs(save_dir):
    gt_save_dir = os.path.join(save_dir, 'groundtruths')
    det_save_dir = os.path.join(save_dir, 'detections')
    if not os.path.exists(gt_save_dir):
        os.makedirs(gt_save_dir)
    if not os.path.exists(det_save_dir):
        os.makedirs(det_save_dir)

    return gt_save_dir, det_save_dir


def gt_convert(test_gt_json, gt_save_dir):
    with open(test_gt_json) as f:
        gt_json = json.load(f)

    imgs = gt_json['images']
    anns = gt_json['annotations']

    for img in imgs:
        filename = img['file_name'].replace('jpg', 'txt')
        img_id = img['id']
        bbox_lst = []
        for ann in anns:
            if ann['image_id'] == img_id:
                bbox = ann['bbox']
                category = ann['category_id']
                xmin = bbox[0]
                ymin = bbox[1]
                xmax = xmin + bbox[2]
                ymax = ymin + bbox[3]
                bbox_lst.append([str(category), str(xmin), str(ymin) ,str(xmax), str(ymax)])
        with open(os.path.join(gt_save_dir, filename),'w') as f:
            for bbox in bbox_lst:
                f.write(' '.join(bbox) + '\n')

def det_convert(test_result_json, det_save_dir):
    with open(test_result_json) as f:
        det_lst = json.load(f)

    for det in det_lst:
        img_name = det['name']
        category = det['category']
        bbox = det['bbox']
        score = det['score']
        xmin, ymin, xmax, ymax = bbox
        lst = [str(category), str(score), str(xmin), str(ymin), str(xmax), str(ymax)]
        det_file = os.path.join(det_save_dir, img_name.replace('jpg', 'txt'))
        with open(det_file, 'a') as f:
            f.write(' '.join(lst) + '\n')

if __name__ == '__main__':
    test_gt_json = r'F:\guangdong\data\guangdong1_round1_train1_20190818\guangdong1_round1_train1_20190818\test_datasets\test_datasets\test.json'
    test_result_json = r'F:\guangdong\result\results_0820.json'
    save_dir = r'F:\guangdong\metrics\test'

    gt_save_dir, det_save_dir = make_save_dirs(save_dir)
    gt_convert(test_gt_json, gt_save_dir)
    det_convert(test_result_json, det_save_dir)


