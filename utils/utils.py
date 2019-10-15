import numpy as np
import cv2
import os
import itertools
import pandas as pd


def bboxes_iou(boxes1, boxes2):
    """
    boxes: [xmin, ymin, xmax, ymax, score, class] format coordinates.
    """
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:4], boxes2[..., 2:4])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    ious = np.maximum(1.0 * inter_area / union_area, 0.0)

    return ious


def nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
    """
    Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
          https://github.com/bharatsingh430/soft-nms
    """
    best_bboxes = []

    while len(bboxes) > 0:
        max_ind = np.argmax(bboxes[:, 4])
        best_bbox = bboxes[max_ind]
        best_bboxes.append(list(best_bbox))

        bboxes = np.concatenate([bboxes[: max_ind], bboxes[max_ind + 1:]])
        iou = bboxes_iou(best_bbox[np.newaxis, :4], bboxes[:, :4])
        weight = np.ones((len(iou),), dtype=np.float32)

        assert method in ['nms', 'soft-nms']
        if method == 'nms':
            iou_mask = iou > iou_threshold
            weight[iou_mask] = 0.0
        if method == 'soft-nms':
            weight = np.exp(-(1.0 * iou ** 2 / sigma))

        bboxes[:, 4] = bboxes[:, 4] * weight
        score_mask = bboxes[:, 4] > 0.
        bboxes = bboxes[score_mask]

    return best_bboxes


def nms_df(df, iou_threshold):
    """
        Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
              https://github.com/bharatsingh430/soft-nms
        """
    best_bboxes = []
    bboxes = df['bbox_score'].values

    while len(bboxes) > 0:
        max_ind = np.argmax(bboxes[:, 4])
        best_bbox = bboxes[max_ind]
        best_bboxes.append(list(best_bbox))

        bboxes = np.concatenate([bboxes[: max_ind], bboxes[max_ind + 1:]])
        iou = bboxes_iou(best_bbox[np.newaxis, :4], bboxes[:, :4])
        weight = np.ones((len(iou),), dtype=np.float32)

        iou_mask = iou > iou_threshold
        weight[iou_mask] = 0.0

        bboxes[:, 4] = bboxes[:, 4] * weight
        score_mask = bboxes[:, 4] > 0.
        bboxes = bboxes[score_mask]

    return best_bboxes


def check_in(boxes1, boxes2, thr=0.9):
    """
    boxes: [xmin, ymin, xmax, ymax, score, class] format coordinates.
    check if boxes2 in boxes1 using threshold thr.
    """
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:4], boxes2[..., 2:4])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    if inter_area / boxes2_area >= thr:
        return 2
    elif inter_area / boxes1_area >= thr:
        return 1
    else:
        return 0


def check_in_filter(original_df, df, thr=0.9):
    """
    filter result df using check in function.
    :param df:
    :param thr: threshold
    :return: filtered df
    """
    delect_bbox = []
    for i in itertools.combinations(df.index, 2):
        bbox1 = df.loc[i[0], 'bbox']
        bbox2 = df.loc[i[1], 'bbox']
        check = check_in(bbox1, bbox2, thr)
        if check:
            delect_bbox.append(i[check - 1])
    delect_bbox = set(delect_bbox)
    for bbox_idx in delect_bbox:
        original_df.drop(index=bbox_idx, inplace=True)

    return original_df


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


if __name__ == '__main__':
    bbox1 = [10, 15, 25, 30, 0.5]
    bbox2 = [12, 16, 22, 35, 0.6]
    print(bboxes_iou(bbox1, bbox2))
    print(nms(np.array([bbox1, bbox2]), 0.6))
    print(check_in(bbox1, bbox2, 0.99))
    df = pd.DataFrame([[[11,11,18,18],1],[[11,11,19,19],1]],columns=['bbox','a'])
    print(df)
    df = check_in_filter(df)
    print(df)

