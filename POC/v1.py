import cv2
import os, glob
import pandas as pd
import pickle
import json
import shutil
import numpy as np
import tqdm
from mmdet.apis import inference_detector, init_detector


def inference_cls(result_file,
                  imgs,
                  code_file,
                  output,
                  img_save_path,
                  save_img=False):
    if not os.path.exists(img_save_path):
        os.makedirs(img_save_path)

    with open(result_file, 'rb') as f:
        results = pickle.load(f)

    cls_result_lst = []
    for i, result in enumerate(results):
        img = imgs[i]
        if result == []:
            print('skip image {}'.format(img))
            continue
        img_name = img.replace('\\', '/').split('/')[-1]
        gt_code = img.replace('\\', '/').split('/')[-2]
        print('processing {}'.format(img_name))

        main_code, bbox, score, img_ = default_rule(result, img, img_name, code_file, save_img)
        print(main_code, bbox, score)

        if save_img:
            cv2.imwrite(os.path.join(img_save_path, img_name), img_)

        if main_code != gt_code:
            save_path = os.path.join(output, gt_code, main_code)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            shutil.copy(img, os.path.join(save_path, img_name))
        cls_result_lst.append(
            {'image name': img_name, 'pred code': main_code, 'defect score': score, 'oic code': gt_code})

    cls_df = pd.DataFrame(cls_result_lst)
    cls_df.to_excel(os.path.join(output, 'result.xlsx'))

def get_result(imgs,
               cfg_file,
               ckpt_file,
               save_file,
               device='cuda:0'):
    assert len(imgs) >= 1, 'There must be at least one image'
    model = init_detector(cfg_file, ckpt_file, device=device)
    final_result = []

    progressbar = tqdm.tqdm(imgs)
    for _, img in enumerate(progressbar):
        image = cv2.imread(img)
        if image is None:
            print('image {} is empty'.format(img))
            final_result.append([])
        result = inference_detector(model, image)
        final_result.append(result)

    if save_file is not None:
        with open(save_file, 'wb') as fp:
            pickle.dump(final_result, fp)

    return final_result


def model_test(result,
               img_name,
               codes,
               score_thr=0.05):
    output_bboxes = []
    json_dict = []

    total_bbox = []
    for id, boxes in enumerate(result):  # loop for categories
        category_id = id + 1
        if len(boxes) != 0:
            for box in boxes:  # loop for bbox
                conf = box[4]
                if conf > score_thr:
                    total_bbox.append(list(box) + [category_id])

    bboxes = np.array(total_bbox)
    best_bboxes = bboxes
    output_bboxes.append(best_bboxes)
    for bbox in best_bboxes:
        coord = [round(i, 2) for i in bbox[:4]]
        conf, category = bbox[4], codes[int(bbox[5]) - 1]
        json_dict.append({'name': img_name, 'category': category, 'bbox': coord, 'score': conf, 'bbox_score': bbox[:5],
                          'xmin': coord[0], 'ymin': coord[1], 'xmax': coord[2], 'ymax': coord[3],
                          'ctx': (coord[0] + coord[2]) / 2, 'cty': (coord[1] + coord[3]) / 2,
                          'size': (coord[2] - coord[0]) * (coord[3] - coord[1])})

    return json_dict


def show_and_save_images(img_path, img_name, bboxes, codes, out_dir=None):
    img = cv2.imread(os.path.join(img_path, img_name))
    for i, bbox in enumerate(bboxes):
        bbox = np.array(bbox)
        bbox_int = bbox[:4].astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        code = codes[i]
        label_txt = code + ': ' + str(round(bbox[4], 2))
        cv2.rectangle(img, left_top, right_bottom, (0, 0, 255), 1)
        cv2.putText(img, label_txt, (bbox_int[0], max(bbox_int[1] - 2, 0)),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    if out_dir is not None:
        cv2.imwrite(os.path.join(out_dir, img_name), img)

    return img

def default_rule(det_lst, img_path, img_name, codes, draw_img=True):
    # PRIO CHECK
    det_df = model_test(det_lst, img_name, codes)
    det_df = pd.DataFrame(det_df)
    if len(det_df) == 0:
        code = 'FALSE'
        score = 0,
        bbox = []
    else:
        code = det_df.loc[det_df['score'].argmax(), 'category']
        bbox = det_df.loc[det_df['score'].argmax(), 'bbox']
        score = det_df['score'].max()

    # draw images
    if draw_img:
        img = show_and_save_images(img_path,
                                   img_name,
                                   det_df.bbox_score.values,
                                   det_df.category.values)
    else:
        img = None

    return code, bbox, score, img


if __name__ == '__main__':
    img_dir = r'/data/sdv1/dali/V1 DDJ/data/0528/images/'
    imgs = glob.glob(os.path.join(img_dir, '*/*.jpg'))
    class_file = r'/data/sdv1/v1/data/classes.txt'
    with open(class_file, 'r') as f:
        codes = f.read().splitlines()
    result = get_result(imgs, r'/data/sdv1/v1/config/faster_rcnn_r50_fpn_1x.py',
                        r'/data/sdv1/v1/workdir/0527/epoch_24.pth',
                        r'/data/sdv1/v1/result/0529/result.pkl')
    inference_cls(r'/data/sdv1/v1/result/0529/result.pkl',
                  imgs,
                  codes,
                  r'/data/sdv1/v1/result/0529/',
                  r'/data/sdv1/v1/result/0529/images'
                  )
