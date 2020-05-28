from inference.get_result_pkl import model_test
from inference.inference_cls import inference_cls
import glob
import os
import json
import pickle


class Config(object):
    # img_path = r'/data/sdv1/whtm/24000/data/test/24000_val_3/'
    img_path = r'/data/sdv2/wjq/21101/data/0511_rep01/REP01'

    # cfg_file = '/data/sdv1/whtm/24000/config/0427.py'
    # ckpt_file = '/data/sdv1/whtm/24000/workdir/24000_0427/epoch_6.pth'
    # json_file = r'/data/sdv1/whtm/24000/model/0426/rule.json'
    # code_file = r'/data/sdv1/whtm/24000/model/0426/classes.txt'

    cfg_file = r'/data/sdv2/wjq/21101/config/test_faster.py'
    ckpt_file = r'/data/sdv2/wjq/21101/work_dir/test_rep01/epoch_20.pth'
    json_file = r'/data/sdv2/wjq/21101_0415/rule.json'
    code_file = r'/data/sdv2/wjq/21101_0415/test_class.txt'

    output = r'/data/sdv2/wjq/output_0528_train/test'
    save_file = r'w01le1129b0311_40737_-71438_before.pkl'
    img_save_path = r'images'
    save_file = os.path.join(output, save_file)
    img_save_path = os.path.join(output, img_save_path)

    size_file = None
    use_pkl = False

    imgs = glob.glob(os.path.join(img_path, 'w02se0501a0501_294755_643933_before.jpg'))
    imgs = [os.path.join(img_path, 'w02se0501a0501_294755_643933_before.jpg')]

    # open config file
    with open(json_file) as f:
        config = json.load(f)
    with open(code_file) as f:
        codes = f.read().splitlines()


opt = Config()
if not os.path.exists(opt.output):
    os.makedirs(opt.output)

if os.path.exists(opt.save_file) and opt.use_pkl:
    print('pkl file already exists!')
    with open(opt.save_file, 'rb') as f:
        result = pickle.load(f)
else:
    result = model_test(opt.imgs, opt.cfg_file, opt.ckpt_file, opt.save_file)
inference_cls(opt.save_file, opt.imgs, opt.config, opt.codes, opt.output, opt.img_save_path, save_img=False,
              size_file=opt.size_file)
