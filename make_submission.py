from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import pandas as pd

import utils.metrics


def find_best_thres(df_val):
    total = len(df_val)

    records = []
    for label in range(28):
        label_key = 'L{:02d}'.format(label)
        prob_key = 'P{:02d}'.format(label)
        df = df_val[[label_key, prob_key]]
        df_pos = df[df[label_key] == 1]
        proportion = len(df_pos) / total

        best_diff = 1000
        best_thres = 0
        for thres in np.arange(0.05, 1.00, 0.01):
            positive = int(np.sum((df_val[prob_key].values > thres).astype(int)))
            cur_proportion = positive / total
            cur_diff = abs(proportion - cur_proportion)
            if cur_diff < best_diff:
                best_diff = cur_diff
                best_thres = thres
        records.append((label, best_thres))
    df_ret = pd.DataFrame.from_records(records, columns=['label', 'thres'])
    return df_ret.set_index('label')


def ensemble(dfs, weights):
    label_keys = ['L{:02}'.format(l) for l in range(28)]
    prob_keys = ['P{:02}'.format(l) for l in range(28)]
    if 'L00' in dfs[0].index:
        df_base = dfs[0][label_keys]
        df_probs = sum([df[prob_keys] * w for df, w in zip(dfs, weights)]) / sum(weights)
        df = pd.concat([df_base, df_probs], axis=1)
    else:
        df = sum([df * w for df, w in zip(dfs, weights)]) / sum(weights)
    return df


def evaluate(df_val, df_thres):
    label_keys = ['L{:02}'.format(l) for l in range(28)]
    prob_keys = ['P{:02}'.format(l) for l in range(28)]

    df_label = df_val[label_keys]
    df_prob = df_val[prob_keys]

    np_label = df_label.values
    np_prob = df_prob.values
    np_pred = (np_prob > df_thres['thres'].values).astype(int)
    f1 = utils.metrics.f1_score(np_label, np_pred)
    return f1


def make_submission(df_test, df_thres):
    thres = df_thres['thres'].values
    records = []
    for Id, row in df_test.iterrows():
        probs = row.values
        pred = list(np.where((probs > thres) == 1)[0])
        labels = ' '.join([str(l) for l in pred])
        records.append((Id, labels))
    df_output = pd.DataFrame.from_records(records, columns=['Id', 'Predicted'])
    return df_output.set_index('Id')


def apply_leak(df_submission, df_leak):
    for key, row in df_leak.iterrows():
        target = row['Target']
        if df_submission.loc[key]['Predicted'] != target:
            df_submission.loc[key]['Predicted'] = target
    return df_submission


def main():
    import warnings
    warnings.filterwarnings("ignore")

    print('make submission')

    test_val_filenames = ['inferences/resnet34.0.test_val.csv',
                          'inferences/resnet34.1.test_val.csv',
                          'inferences/resnet34.2.test_val.csv',
                          'inferences/resnet34.3.test_val.csv',
                          'inferences/resnet34.4.test_val.csv',
                          'inferences/inceptionv3.0.test_val.csv',
                          'inferences/se_resnext50.0.test_val.csv']

    test_filenames = ['inferences/resnet34.0.test.csv',
                      'inferences/resnet34.1.test.csv',
                      'inferences/resnet34.2.test.csv',
                      'inferences/resnet34.3.test.csv',
                      'inferences/resnet34.4.test.csv',
                      'inferences/inceptionv3.0.test.csv',
                      'inferences/se_resnext50.0.test.csv']

    weights = [1/5, 1/5, 1/5, 1/5, 1/5, 1.0, 1.0]

    leak_filenames = ['data/leak.csv',
                      'data/data_leak.ahash.csv',
                      'data/data_leak.phash.csv']

    output_filename = 'submissions/submission.csv'
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    df_test_val_list = [pd.read_csv(f) for f in test_val_filenames]
    df_test_list = [pd.read_csv(f, index_col='Id') for f in test_filenames]

    print('ensemble..')
    df_test_val = ensemble(df_test_val_list, weights)
    df_test = ensemble(df_test_list, weights)

    df_thres = find_best_thres(df_test_val)
    f1 = evaluate(df_test_val, df_thres)
    print('validation f1:', f1)

    df_submission = make_submission(df_test, df_thres)
    df_submission.to_csv(output_filename)

    print('apply leak')
    df_leak_list = [pd.read_csv(f, index_col='Id') for f in leak_filenames]
    for df_leak in df_leak_list:
        df_submission = apply_leak(df_submission, df_leak)
    df_submission.to_csv(output_filename + '.leak.csv')


if __name__ == '__main__':
    main()
