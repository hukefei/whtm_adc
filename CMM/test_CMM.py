from mask.main import *
import glob
import shutil
import os
import pandas as pd
from CMM.main import *

path = r'/data/sdv1/whtm/CMM/model/0605/'
opt = process(path)

img_path = r'/data/sdv1/whtm/CMM/test/CMM_gray_test/'
imgs = glob.glob(os.path.join(img_path, '*/*.jpg'))
# imgs = [os.path.join(img_path, 'PO03/u4639fh1fbebw207_000068997_-00617449_before.jpg')]
progressbar = tqdm.tqdm(imgs)
save_dir = r'/data/sdv1/whtm/CMM/result/CMM_gray_test_0605_d'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

result = []
for img in progressbar:
    oic = img.split('/')[-2]
    image_name = img.split('/')[-1]
    code, _, score, img = predict(img, **opt)
    save_path = os.path.join(save_dir, oic, code)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if oic != code:
        cv2.imwrite(os.path.join(save_path, image_name), img)
    result.append({'oic_code': oic, 'adc_code': code, 'score': score, 'image_name': image_name})
    # print(oic, code, score)
df = pd.DataFrame(result, columns=['oic_code', 'adc_code', 'score', 'image_name'])
df.to_excel(os.path.join(save_dir, 'result.xlsx'))