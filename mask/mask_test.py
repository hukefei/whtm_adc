from mask.main import *
import glob
import shutil
import os
import pandas as pd

path = r'/data/sdv1/whtm/mask/model/FMM_0525/'
opt = process(path)
img_path = r'/data/sdv1/whtm/mask/test/Review_AMI200/'
imgs = glob.glob(os.path.join(img_path, '*/*/*.jpg'))
# imgs = [os.path.join(img_path, 'T6647FH1FAERT501_814-6_-687-6796_before.jpg')]
progressbar = tqdm.tqdm(imgs)
save_dir = r'/data/sdv1/whtm/mask/test/0525_AMI200/result'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

result = []
for img in progressbar:
    oic_code = img.split('/')[-3]
    product = img.split('/')[-2]
    image_name = img.split('/')[-1]
    code, bbox, score, image = predict(img, **opt)
    save_path = os.path.join(save_dir, oic_code, code)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # shutil.copy(img, os.path.join(save_path, image_name))
    cv2.imwrite(os.path.join(save_path, image_name), image)
    result.append({'oic_code': oic_code, 'adc_code': code, 'score': score, 'bbox': bbox})

df = pd.DataFrame(result, columns=['oic_code', 'adc_code', 'score', 'bbox'])
df.to_excel(os.path.join(save_dir, 'result.xlsx'))
# img = r'/data/sdv1/whtm/mask/test/data/U4549FH2FAHRT109_-380159-2_122-419_before.jpg'
# code, bbox, score, _ = predict(img, **opt)