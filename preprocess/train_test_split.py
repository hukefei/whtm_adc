import shutil
import os
import random

def train_test_split(img_dir, xml_dir, train_size, output):
    imgs = os.listdir(img_dir)
    train_set = random.sample(imgs, int(len(imgs)*train_size))
    train_dir = os.path.join(output, 'train')
    test_dir = os.path.join(output, 'val')
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    for img in imgs:
        xml = img[:-4] + '.xml'
        if img in train_set:
            shutil.copy(os.path.join(img_dir, img), os.path.join(train_dir, img))
            shutil.copy(os.path.join(xml_dir, xml), os.path.join(train_dir, xml))
        else:
            shutil.copy(os.path.join(img_dir, img), os.path.join(test_dir, img))
            shutil.copy(os.path.join(xml_dir, xml), os.path.join(test_dir, xml))

if __name__ == '__main__':
    img_dir = r'F:\WHTM\CMM\data\trainval_OD_0616\images'
    anns_dir = r'F:\WHTM\CMM\data\trainval_OD_0616\annotations'
    TRAIN_SIZE = 0.8
    output = r'F:\WHTM\CMM\data\trainval_OD_0616'
    train_test_split(img_dir, anns_dir, TRAIN_SIZE, output)
