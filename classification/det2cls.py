import json
import pandas as pd
from utils.Code_dictionary import CodeDictionary
import os


def prio_check(prio_lst, code_lst):
    idx_lst = []
    for code in code_lst:
        assert code in prio_lst, '{} should be in priority file'.format(code)
        idx = prio_lst.index(code)
        idx_lst.append(idx)
    final_code = prio_lst[min(idx_lst)][0]
    return final_code


def filter_code(df, code, thr, replace=None):
    check_code = df[(df['category'] == code) & (df['score'] < thr)]
    if replace is None:
        df = df.drop(index=check_code.index)
    else:
        df.loc[check_code.index, 'category'] = replace
    return df


def maxconf(det_df, **kwargs):
    if len(det_df) == 0:
        return kwargs['false_name']
    else:
        filtered = det_df[det_df['score'] >= kwargs['other_thr']]
        if len(filtered) == 0:
            return kwargs['other_name']

        max_idx = det_df['score'].idxmax()
        code = det_df.loc[max_idx, 'category']
        return code


def priority(det_df, **kwargs):
    assert 'prio_file' in kwargs.keys(), 'Must input a priority file'
    priority_file = kwargs['prio_file']
    if len(det_df) == 0:
        return kwargs['false_name']
    else:
        filtered = det_df[det_df['score'] >= kwargs['other_thr']]
        if len(filtered) == 0:
            return kwargs['other_name']

        prio = pd.read_excel(priority_file)
        prio_lst = list(prio.values)
        final_code = prio_check(prio_lst, list(filtered['category'].values))
        return final_code


def maxconf_priority(det_df, **kwargs):
    assert 'prio_weight' in kwargs.keys(), 'Must input priority weight'
    assert 'prio_file' in kwargs.keys(), 'Must input priority file'

    prio_weight = kwargs['prio_weight']
    prio_file = kwargs['prio_file']

    if len(det_df) == 0:
        return kwargs['false_name']
    else:
        filtered = det_df[det_df['score'] >= kwargs['other_thr']]
        if len(filtered) == 0:
            return kwargs['other_name']

        Max_conf = max(filtered['score'].values)
        prio_thr = Max_conf * prio_weight
        filtered = filtered[filtered['score'] >= prio_thr]

        prio = pd.read_excel(prio_file)
        prio_lst = list(prio.values)
        final_code = prio_check(prio_lst, list(filtered['category'].values))

        return final_code


'''
def default_rule(det_df, **kwargs):
    assert 'prio_weight' in kwargs.keys(), 'Must input priority weight'
    assert 'prio_file' in kwargs.keys(), 'Must input priority file'

    prio_weight = kwargs['prio_weight']
    prio_file = kwargs['prio_file']

    if len(det_df) == 0:
        return kwargs['false_name']
    else:
        filtered = det_df[det_df['score'] >= kwargs['other_thr']]
        if len(filtered) == 0:
            return kwargs['other_name']

        df_res04 = filtered[filtered['category'] == 'RES04']
        df_res05 = filtered[filtered['category'] == 'RES05']
        for idx, row in df_res04.iterrows():
            bbox1 = row['bbox']
            count = 0
            for idx2, row2 in df_res05.iterrows():
                bbox2 = row2['bbox']
                if check_in(bbox1, bbox2):
                    count += 1
            if count == 2:
                filtered.drop(idx, inplace=True)
                print('Drop 1 RES04 bbox')

        Max_conf = max(filtered['score'].values)
        prio_thr = Max_conf * prio_weight
        filtered = filtered[filtered['score'] >= prio_thr]

        prio = pd.read_excel(prio_file)
        prio_lst = list(prio.values)
        final_code = prio_check(prio_lst, list(filtered['category'].values))

        return final_code
'''


def default_rule(det_df, **kwargs):
    assert 'prio_weight' in kwargs.keys(), 'Must input priority weight'
    assert 'prio_file' in kwargs.keys(), 'Must input priority file'

    prio_weight = kwargs['prio_weight']
    prio_file = kwargs['prio_file']

    if len(det_df) == 0:
        return kwargs['false_name'], 1
    else:
        filtered = det_df[det_df['score'] >= kwargs['other_thr']]
        if len(filtered) == 0:
            return kwargs['other_name']

        df_res04 = filtered[filtered['category'] == 'QS']
        if len(df_res04) >= 3:
            filtered.loc[filtered['category'] == 'QS', 'category'] = 'RES04'
        else:
            filtered.loc[filtered['category'] == 'QS', 'category'] = 'RES05'

        # delect redundant RES06 code
        # delect redundant RES03 code
        filtered = filter_code(filtered, 'RES06', 0.7)
        #filtered = filter_code(filtered, 'RES03', 0.3)
        filtered = filter_code(filtered, 'RES03', 0.5, 'RES05')
        filtered = filter_code(filtered, 'AZ08', 0.6)
        filtered = filter_code(filtered, 'STR02', 0.6)
        filtered = filter_code(filtered, 'STR04', 0.8, 'COM01')
        filtered = filter_code(filtered, 'COM03', 0.9)
        filtered = filter_code(filtered, 'COM01', 0.3)
        filtered = filter_code(filtered, 'PLN01', 0.4)
        filtered = filter_code(filtered, 'REP01', 0.9)

        if len(filtered) == 0:
            return kwargs['false_name'], 1

        Max_conf = max(filtered['score'].values)
        prio_thr = Max_conf * prio_weight
        filtered = filtered[filtered['score'] >= prio_thr]

        prio = pd.read_excel(prio_file)
        prio_lst = list(prio.values)
        final_code = prio_check(prio_lst, list(filtered['category'].values))
        defect_score = max(filtered.loc[filtered['category'] == final_code, 'score'].values)
        return final_code, defect_score


def check_in(bbox1, bbox2, thr=0.8):
    bbox2_size = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    inner = (max(bbox1[0], bbox2[0]), max(bbox1[1], bbox2[1]), min(bbox1[2], bbox2[2]), min(bbox1[3], bbox2[3]))
    inner_w = inner[2] - inner[0]
    inner_h = inner[3] - inner[1]
    if inner_w <= 0 or inner_h <= 0:
        return False
    else:
        inner_size = inner_h * inner_w
        if inner_size / bbox2_size > thr:
            return True
        else:
            return False


def det2cls(json_file,
            test_images_dir,
            code_dict,
            false_thr=0.05,
            other_thr=0.5,
            other_name='other',
            rule=0,
            prio_file=None,
            prio_weight=0.9,
            false_name='COM99',
            output='classification_result.xlsx'):
    rule_dict = {0: 'Max Confidence',
                 1: 'Priority Only',
                 2: 'Max Confidence and Priority',
                 3: 'Default'}
    rule_func = {0: maxconf,
                 1: priority,
                 2: maxconf_priority,
                 3: default_rule}
    chosen_rule_func = rule_func[rule]
    print('Bbox score less than {} is judged as {}'.format(other_thr, other_name))
    print('Bbox score less than {} is judged as COM99'.format(false_thr))
    print('-' * 50)

    assert rule in rule_dict.keys(), 'wrong rule parameter set'

    print('Now using converting rule: {}'.format(rule_dict[rule]))

    with open(json_file) as f:
        json_dict = json.load(f)

    det_df = pd.DataFrame(json_dict)
    det_df['category'] = code_dict.id2code(list(det_df['category'].values))
    # img_lst = list(det_df['name'].unique())
    img_lst = os.listdir(test_images_dir)

    filtered_df = det_df[det_df['score'] > false_thr]

    cls_result_lst = []
    for img_name in img_lst:
        img_det_df = filtered_df[filtered_df['name'] == img_name]
        cls_result, defect_score = chosen_rule_func(img_det_df,
                                                    prio_file=prio_file,
                                                    prio_weight=prio_weight,
                                                    other_thr=other_thr,
                                                    other_name=other_name,
                                                    false_name=false_name)

        cls_result_lst.append({'image name': img_name, 'pred code': cls_result, 'defect score': defect_score})
    cls_df = pd.DataFrame(cls_result_lst)
    cls_df.to_excel(output)


if __name__ == '__main__':
    result_json = r'D:\Project\WHTM\result\21101\bt2\21101_bt2_result_v2.json'
    prio_file = r'D:\Project\WHTM\code\21101_prio.xlsx'
    test_images = r'E:\whtm\21101bt_part2_2000'
    code_file = r'D:\Project\WHTM\code\21101code.xlsx'
    code = CodeDictionary(code_file)

    det2cls(result_json, test_images, code, false_thr=0.05, other_thr=0, rule=3, prio_file=prio_file, prio_weight=0.1)
