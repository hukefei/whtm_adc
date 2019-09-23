import cv2
import os
import numpy as np
from PIL import Image

#img_path = r'E:\work\data\14303\14303labeled_NotAZ10\AZ11\uni_defect'
img_path = r'F:\XMTM\station\14103\data\14103_0815\PSL05'

for root, dirs, files in os.walk(img_path):
    for file in files:
        if file[-3:] in ('JPG', 'jpg'):
            try:
                imagepath = os.path.join(root, file)
                img = Image.open(imagepath)
                a = imagepath.encode('ascii')
            except Exception as e:
                print(imagepath, e)

