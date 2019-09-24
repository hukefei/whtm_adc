import json
import pandas as pd
import numpy as np

def prio_check(prio_lst, code_lst):
    idx_lst = []
    for code in code_lst:
        assert code in prio_lst, '{} should be in priority file'.format(code)
        idx = prio_lst.index(code)
        idx_lst.append(idx)
    final_code = prio_lst[min(idx_lst)][0]
    return final_code


def maxconf(det_df, **kwargs):
    if len(det_df) == 0:
        return 'C0M99'
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
        return 'COM99'
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
        return 'COM99'
    else:
        filtered = det_df[det_df['score'] >= kwargs['other_thr']]
        if len(filtered) == 0:
            return kwargs['other_name']

        Max_conf = max(det_df['score'].values)
        prio_thr = Max_conf * prio_weight
        filtered = filtered[filtered['score'] >= prio_thr]

        prio = pd.read_excel(prio_file)
        prio_lst = list(prio.values)
        final_code = prio_check(prio_lst, list(filtered['category'].values))

        return final_code


def det2cls(json_file,
            false_thr=0.05,
            other_thr=0.5,
            other_name='other',
            rule=0,
            prio_file=None,
            prio_weight=0.9,
            output='classification_result.xlsx'):
    rule_dict = {0: 'Max Confidence',
                 1: 'Priority Only',
                 2: 'Max Confidence and Priority'}
    rule_func = {0: maxconf,
                 1: priority,
                 2: maxconf_priority}
    chosen_rule_func = rule_func[rule]
    print('Bbox score less than {} is judged as {}'.format(other_thr, other_name))
    print('Bbox score less than {} is judged as COM99'.format(false_thr))
    print('-' * 50)

    assert rule in rule_dict.keys(), 'wrong rule parameter set'

    print('Now using converting rule: {}'.format(rule_dict[rule]))

    with open(json_file) as f:
        json_dict = json.load(f)

    det_df = pd.DataFrame(json_dict)
    img_lst = list(det_df['name'].unique())

    filtered_df = det_df[det_df['score'] > false_thr]

    cls_result_lst = []
    for img_name in img_lst:
        img_det_df = filtered_df[filtered_df['name'] == img_name]
        cls_result = chosen_rule_func(img_det_df,
                                      prio_file=prio_file,
                                      prio_weight=prio_weight,
                                      other_thr=other_thr,
                                      other_name=other_name)
        cls_result_lst.append({'image name': img_name, 'pred code': cls_result})
    cls_df = pd.DataFrame(cls_result_lst)
    cls_df.to_excel(output)


if __name__ == '__main__':
    result_json = r'C:\Users\huker\Desktop\project\submit\test\result.json'
    prio_file = r'C:\Users\huker\Desktop\project\submit\test\cloth_defect_prio.xlsx'

    det2cls(result_json, false_thr=0.4, rule=1, prio_file=prio_file)
