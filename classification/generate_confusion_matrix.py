import pandas as pd
from sklearn.metrics import confusion_matrix

def wrap_confusion_matrix(cm_df):
    for i, col in enumerate(cm_df.columns):
        cm_df.loc['precision', col] = cm_df.iloc[i, i] / cm_df.iloc[:, i].sum()
    for i, idx in enumerate(cm_df.index):
        if idx == 'precision':
            continue
        cm_df.loc[idx, 'recall'] = cm_df.iloc[i, i] / cm_df.iloc[i, :].sum()
    return cm_df

def generate_confusion_matrix(det_result_file, gt_result_file, output = 'confusion_matrix.xlsx'):
    det_df = pd.read_excel(det_result_file)
    gt_df = pd.read_excel(gt_result_file)
    merged_df = pd.merge(det_df, gt_df, on='image name')

    merged_df.dropna(inplace=True)
    print('{} images merged \n{} images det \n{} images det'.format(len(merged_df), len(det_df), len(gt_df)))
    y_pred = list(merged_df['pred code'].values.astype(str))
    y_true = list(merged_df['true code'].values.astype(str))
    labels = list(set(y_pred) | set(y_true))

    cm = confusion_matrix(y_true, y_pred, labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df = wrap_confusion_matrix(cm_df)

    print(cm_df)

    cm_df.to_excel(output)




if __name__ == '__main__':
    det_result = r'C:\Users\huker\PycharmProjects\whtm_adc\classification\classification_result.xlsx'
    gt_result = r'C:\Users\huker\PycharmProjects\whtm_adc\classification\ground_truth_result.xlsx'
    generate_confusion_matrix(det_result, gt_result)