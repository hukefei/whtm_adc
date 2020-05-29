from mask.main import *
import glob
import shutil
import os
import pandas as pd
from CMM.main import *

path = r'/data/sdv1/whtm/CMM/model/0529/'
opt = process(path)

img_path = r'/data/sdv1/whtm/CMM/data/trainval_0529/val/'
imgs = glob.glob(os.path.join(img_path, 'FALSE/*.jpg'))
# imgs = [os.path.join(img_path, 'PO03/u4639fh1fbebw207_000068997_-00617449_before.jpg')]
progressbar = tqdm.tqdm(imgs)
save_dir = r'/data/sdv1/whtm/mask/test/0528_5'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for img in progressbar:
    code, _, score, _ = predict(img, **opt)