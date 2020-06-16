from CMM_OD.main import *
import glob
import shutil
import os
import pandas as pd

path = r'/data/sdv1/whtm/CMM/model/OD_0616_faster/'
opt = process(path)
img_path = r'/data/sdv1/whtm/CMM/test/CMM_TEST_ADD_0609/'
imgs = glob.glob(os.path.join(img_path, '*/*.jpg'))
# imgs = [os.path.join(img_path, '*/*/SC655FH1HAERT202_-285424_-656-5246_before.jpg')]
progressbar = tqdm.tqdm(imgs)
save_dir = r'/data/sdv1/whtm/CMM/result/0616/CMM_TEST_ADD_0609_3'
save_correct_image = False
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

result = []
for img in progressbar:
    oic_code = img.split('/')[-2]
    image_name = img.split('/')[-1]
    code, bbox, score, image = predict(img, **opt)
    save_path = os.path.join(save_dir, oic_code, code)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if not save_correct_image:
        if oic_code != code:
            cv2.imwrite(os.path.join(save_path, image_name), image)
    else:
        cv2.imwrite(os.path.join(save_path, image_name), image)
    result.append({'oic_code': oic_code, 'adc_code': code, 'score': score, 'bbox': bbox, 'image_name': image_name})

df = pd.DataFrame(result, columns=['oic_code', 'adc_code', 'score', 'bbox', 'image_name'])
df.to_excel(os.path.join(save_dir, 'result.xlsx'))
# img = r'/data/sdv1/whtm/mask/test/data/U4549FH2FAHRT109_-380159-2_122-419_before.jpg'
# code, bbox, score, _ = predict(img, **opt)