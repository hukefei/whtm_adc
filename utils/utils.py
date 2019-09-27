

def IOU(bbox1, bbox2):
    xmin1, ymin1, xmax1, ymax1 = bbox1
    xmin2, ymin2, xmax2, ymax2 = bbox2

    bbox_size1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    bbox_size2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    x_inner_min = max(xmin1, xmin2)
    y_inner_min = max(ymin1, ymin2)

    x_inner_max = min(xmax1, ymax1)
    y_inner_max = min(ymax1, ymax2)

    if x_inner_min < x_inner_max and y_inner_min < y_inner_max:
        inner_size = (x_inner_max - x_inner_min) * (y_inner_max - y_inner_min)
    else:
        inner_size = 0

    IOU = inner_size / (bbox_size1 + bbox_size2 - inner_size)

    return IOU