import os
import shutil

ann_dir = r'E:\szl\MASK\demo_dataset\0514_XML'
img_dir = r'E:\szl\MASK\demo_dataset\AMI100'
save_dir = r'E:\szl\MASK\demo_dataset\labeled'

for root, _, files in os.walk(ann_dir):
    for file in files:
        if file.endswith('xml'):
            img_name = file.replace('xml', 'jpg')
            if not os.path.exists(os.path.join(img_dir, img_name)):
                continue
            shutil.copy(os.path.join(img_dir, img_name), os.path.join(save_dir, img_name))