import json
import pandas as pd
import numpy as np

JSON_FILE = r''

def maxconf(det_df):
    if len(det_df) == 0:
        return 'C0M99'
    else:
        max_idx = det_df['score'].idxmax()
        code = det_df.loc[max_idx, 'category']
        return code

def priority(det_df, priority_file):
    pass

def maxconf_priority(det_df, priority_file):
    pass

def det2cls(json_file, false_thr=0.05, rule = 0):
    rule_dict = {0: 'Max Confidence',
                 1: 'Priority Only',
                 2: 'Max Confidence and Priority'}
    rule_func = {0: maxconf,
                 1: priority,
                 2: maxconf_priority}
    chosen_rule_func = rule_func[rule]


    assert rule in rule_dict.keys(), 'wrong rule parameter set'

    print('Now using converting rule: {}'.format(rule_dict[rule]))

    with open(json_file) as f:
        json_dict = json.load(f)

    det_df = pd.DataFrame(json_dict)
    img_lst = list(det_df['name'].unique())

    filtered_df = det_df[det_df['score'] > false_thr]

    cls_result = []
    for img_name in img_lst:
        img_det_df = filtered_df[filtered_df['name'] == img_name]
        cls_result = chosen_rule_func(img_det_df)
        cls_result.append({'image name':img_name, 'code':cls_result})