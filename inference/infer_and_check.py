from inference.get_result_pkl import model_test
from inference.inference_cls import inference_cls
import glob
import os
import json
import pickle


class Config(object):
    # img_path = r'/data/sdv1/whtm/24000/data/test/24000_val_3/'
    img_path = r'/data/sdv1/whtm/24000/data/0509/24000_655_0508'

    # cfg_file = '/data/sdv1/whtm/24000/config/0427.py'
    # ckpt_file = '/data/sdv1/whtm/24000/workdir/24000_0427/epoch_6.pth'
    # json_file = r'/data/sdv1/whtm/24000/model/0426/rule.json'
    # code_file = r'/data/sdv1/whtm/24000/model/0426/classes.txt'

    cfg_file = r'/data/sdv1/whtm/24000/model/0427/0427.py'
    ckpt_file = r'/data/sdv1/whtm/24000/model/0427/model-2026ed0b.pth'
    json_file = r'/data/sdv1/whtm/24000/model/0427/rule.json'
    code_file = r'/data/sdv1/whtm/24000/model/0427/classes.txt'

    output = r'/data/sdv1/whtm/24000/result/0509_2/'
    save_file = r'24000_0509_2.pkl'
    img_save_path = r'images'
    save_file = os.path.join(output, save_file)
    img_save_path = os.path.join(output, img_save_path)

    size_file = None

    imgs = glob.glob(os.path.join(img_path, '*/*.jpg'))

    # open config file
    with open(json_file) as f:
        config = json.load(f)
    with open(code_file) as f:
        codes = f.read().splitlines()


opt = Config()
if not os.path.exists(opt.output):
    os.makedirs(opt.output)

if os.path.exists(opt.save_file):
    print('pkl file already exists!')
    with open(opt.save_file, 'rb') as f:
        result = pickle.load(f)
else:
    result = model_test(opt.imgs, opt.cfg_file, opt.ckpt_file, opt.save_file)
inference_cls(opt.save_file, opt.imgs, opt.config, opt.codes, opt.output, opt.img_save_path, save_img=False,
              size_file=opt.size_file)
