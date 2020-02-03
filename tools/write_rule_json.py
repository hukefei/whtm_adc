import json
import pandas as pd

prio_file = r'D:\Project\WHTM\model\21101\1231\优先级.xlsx'
false_name = 'COM99'
other_name = ''
prio_weight = 0.7
other_thr = 0
save_file = r'D:\Project\WHTM\model\21101\1231\rule.json'

prio_df = pd.read_excel(prio_file)
print(prio_df)
prio_lst = list(prio_df['code'].values)
print(prio_lst)

json_dict = {'false_name': false_name, 'other_name': other_name, 'prio_weight': prio_weight, 'other_thr': other_thr,
             'prio_order': prio_lst}

with open(save_file, 'w') as f:
    json.dump(json_dict, f, indent=4)

