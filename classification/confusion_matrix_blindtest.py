import pandas as pd
from sklearn.metrics import confusion_matrix
from utils.Code_dictionary import CodeDictionary
import copy
import numpy as np


def wrap_confusion_matrix(cm_df):
    for i, col in enumerate(cm_df.columns):
        cm_df.loc['precision', col] = cm_df.iloc[i, i] / cm_df.iloc[:, i].sum()
    for i, idx in enumerate(cm_df.index):
        if idx == 'precision':
            continue
        cm_df.loc[idx, 'recall'] = cm_df.iloc[i, i] / cm_df.iloc[i, :].sum()
    return cm_df


def generate_confusion_matrix(det_result_file,
                              gt_result_file,
                              labels,
                              output='confusion_matrix.xlsx',
                              code_weight=None):
    det_df = pd.read_excel(det_result_file)
    gt_df = pd.read_excel(gt_result_file)
    merged_df = pd.merge(det_df, gt_df, on='image name')

    merged_df.loc[pd.isnull(merged_df['true code']), 'true code'] = merged_df['pred code']
    merged_df.drop(merged_df[merged_df['true code'] == 1].index, axis=0, inplace=True)
    print('{} images merged \n{} images det \n{} images gt'.format(len(merged_df), len(det_df), len(gt_df)))
    y_pred = list(merged_df['pred code'].values.astype(str))
    y_true = list(merged_df['true code'].values.astype(str))

    cm = confusion_matrix(y_true, y_pred, labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    if code_weight is not None:
        code_weight = np.array(code_weight) * 1000
        print('output balanced confusion matrix')
        assert len(code_weight) == len(labels)
        cm_df_balanced = copy.deepcopy(cm_df)
        for i in range(len(code_weight)):
            sum = cm_df_balanced.iloc[i, :].sum()
            row_weight = code_weight[i] / sum
            cm_df_balanced.iloc[i, :] *= row_weight
        cm_df_balanced = wrap_confusion_matrix(cm_df_balanced)
        cm_df_balanced.to_excel(output.replace('.xlsx', '_balanced.xlsx'))

    cm_df = wrap_confusion_matrix(cm_df)
    print(cm_df)

    cm_df.to_excel(output)


if __name__ == '__main__':
    det_result = r'/data/sdv1/whtm/result/21101/v4_bt/classification_result.xlsx'
    gt_result = r'/data/sdv1/whtm/result/21101/v4_bt/true_code.xlsx'
    code_file = r'/data/sdv1/whtm/document/21101code.xlsx'

    code = CodeDictionary(code_file)
    labels = code.code_list
    labels.remove('QS')
    labels.append('RES05')
    labels.append('RES04')
    generate_confusion_matrix(det_result,
                              gt_result,
                              labels,
                              output=r'/data/sdv1/whtm/result/21101/v4_bt/confusion_matrix.xlsx')
