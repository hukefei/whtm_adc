from mask.main import *
import glob
import shutil
import os
import pandas as pd
import datetime


path = r'/data/sdv1/whtm/mask/model/FMM_color_0730/'
opt = process(path)
img_path = r'/data/sdv1/whtm/mask/test/0723/FMM_REVIEW/'
imgs = glob.glob(os.path.join(img_path, '*/PO01_2.jpg'))
# imgs = [os.path.join(img_path, '*/*/SC655FH1HAERT202_-285424_-656-5246_before.jpg')]
progressbar = tqdm.tqdm(imgs)
dateTime_p = datetime.datetime.now()
str_p = datetime.datetime.strftime(dateTime_p, '%m%d%H%M%S')
save_dir = r'/data/sdv1/whtm/mask/result/0730_val_{}'.format(str_p)
save_correct_image = False
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

total_ = len(progressbar)
correct = 0

result = []
for img in progressbar:
    oic_code = img.split('/')[-2]
    image_name = img.split('/')[-1]
    code, bbox, score, image = predict(img, **opt)
    save_path = os.path.join(save_dir, oic_code, code)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # shutil.copy(img, os.path.join(save_path, image_name))
    print(oic_code, code)
    if oic_code == code:
        correct += 1
    if not save_correct_image:
        if oic_code != code:
            cv2.imwrite(os.path.join(save_path, image_name), image)
    else:
        cv2.imwrite(os.path.join(save_path, image_name), image)
    result.append({'oic_code': oic_code, 'adc_code': code, 'score': score, 'bbox': bbox, 'image_name': image_name})

print(correct / total_)

df = pd.DataFrame(result, columns=['oic_code', 'adc_code', 'score', 'bbox', 'image_name'])
df.to_excel(os.path.join(save_dir, 'result.xlsx'))