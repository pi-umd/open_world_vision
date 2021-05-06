import argparse
import json
import os

import numpy as np
import pandas as pd
import eval

# Defining column names corresponding to different novelties. These columns should be boolean
CLASS_COUNT = 414  # this is k+1 when class numbers starting from 1
GT_CLASS_COL = 'class_id'
NOVELTY_IND_COLS = [
    'instance_novelty',
    'cls_novelty',
    'attr_novelty',
    'rep_novelty',
]
METRIC_LIST = {
    'accuracy': {
        'func': eval.get_accuracy,
        'known': ['instance_novelty', 'attr_novelty', 'rep_novelty'],
        'unknown': ['cls_novelty'],
        'binary': False,
        'col_list': list(map(str, range(1, CLASS_COUNT + 1))),
        'gt_shift': 0,
    },
    'auroc_base': {
        'func': eval.roc,
        'known': ['instance_novelty'],
        'unknown': ['cls_novelty'],
        'binary': True,
        'gt_shift': 0,
    },
    'auroc_attr': {
        'func': eval.roc,
        'known': ['attr_novelty'],
        'unknown': ['cls_novelty'],
        'binary': True,
        'gt_shift': 0,
    },
    'auroc_rep': {
        'func': eval.roc,
        'known': ['rep_novelty'],
        'unknown': ['cls_novelty'],
        'binary': True,
        'gt_shift': 0,
    },
    'accuracy_base': {
        'func': eval.get_accuracy,
        'known': ['instance_novelty'],
        'unknown': [],
        'binary': False,
        'col_list': list(map(str, range(2, CLASS_COUNT + 1))),
        'gt_shift': 1,
    },
    'accuracy_attr': {
        'func': eval.get_accuracy,
        'known': ['attr_novelty'],
        'unknown': [],
        'binary': False,
        'col_list': list(map(str, range(2, CLASS_COUNT + 1))),
        'gt_shift': 1,
    },
    'accuracy_rep': {
        'func': eval.get_accuracy,
        'known': ['rep_novelty'],
        'unknown': [],
        'binary': False,
        'col_list': list(map(str, range(2, CLASS_COUNT + 1))),
        'gt_shift': 1,
    },
    'auprc_base': {
        'func': eval.pr,
        'known': ['instance_novelty'],
        'unknown': ['cls_novelty'],
        'binary': True,
        'gt_shift': 0,
    },
    'auprc_attr': {
        'func': eval.pr,
        'known': ['attr_novelty'],
        'unknown': ['cls_novelty'],
        'binary': True,
        'gt_shift': 0,
    },
    'auprc_rep': {
        'func': eval.pr,
        'known': ['rep_novelty'],
        'unknown': ['cls_novelty'],
        'binary': True,
        'gt_shift': 0,
    },
    'macro_f1_base': {
        'func': eval.macro_F1,
        'known': ['instance_novelty'],
        'unknown': ['cls_novelty'],
        'binary': False,
        'col_list': list(map(str, range(1, CLASS_COUNT + 1))),
        'gt_shift': 0,
    },
    'macro_f1_attr': {
        'func': eval.macro_F1,
        'known': ['attr_novelty'],
        'unknown': ['cls_novelty'],
        'binary': False,
        'col_list': list(map(str, range(1, CLASS_COUNT + 1))),
        'gt_shift': 0,
    },
    'macro_f1_rep': {
        'func': eval.macro_F1,
        'known': ['rep_novelty'],
        'unknown': ['cls_novelty'],
        'binary': False,
        'col_list': list(map(str, range(1, CLASS_COUNT + 1))),
        'gt_shift': 0,
    },
    'ccr_base': {
        'func': eval.ccr,
        'known': ['instance_novelty'],
        'unknown': ['cls_novelty'],
        'binary': False,
        'col_list': list(map(str, range(1, CLASS_COUNT + 1))),
        'gt_shift': 0,
    },
    'ccr_attr': {
        'func': eval.ccr,
        'known': ['attr_novelty'],
        'unknown': ['cls_novelty'],
        'binary': False,
        'col_list': list(map(str, range(1, CLASS_COUNT + 1))),
        'gt_shift': 0,
    },
    'ccr_rep': {
        'func': eval.ccr,
        'known': ['rep_novelty'],
        'unknown': ['cls_novelty'],
        'binary': False,
        'col_list': list(map(str, range(1, CLASS_COUNT + 1))),
        'gt_shift': 0,
    },
}


def get_parser():
    """"Defines the command line arguments"""
    parser = argparse.ArgumentParser(description='Open World Vision')
    parser.add_argument('--pred_file', required=True, type=str,
                        help='path to csv file containing predicted labels. First column should contain image name and'
                             'rest of the columns should contain predicted probabilities for each of the class_id')
    parser.add_argument('--gt_file', required=True, type=str,
                        help='path to csv file containing ground-truth labels. First column should contain image name'
                             ' and second column should contain the ground truth class_id')
    parser.add_argument('--out_dir', required=True, type=str,
                        help='path of directory for storing results')
    parser.add_argument('--model_name', required=True, type=str,
                        help='unique name for the model')
    return parser


def filter_data(data, novelties=None):
    """filters the dataframe based on given novelty columns. Basically it will include rows which have
    value=1 in the specified novelty columns and value!=1 in rest of NOVELTY_IND_COLUMNS

    Args:
        data (pd.DataFrame): input data frame
        novelties (list of str): list of novelty columns which should specify the novelty type

    Returns:
        filtered_data (pd.DataFrame): filtered data based on above logic
    """
    assert novelties is not None and isinstance(novelties, list), 'novelties variable should be a list'
    filter_query = '('  # building query for dynamic evaluation on a dataframe
    for novelty in novelties:
        assert novelty in NOVELTY_IND_COLS, f'Unsupported novelty column. Available options are ' \
                                            f'{",".join(NOVELTY_IND_COLS)}'
        filter_query += f'{novelty} == 1 | '
    if len(filter_query) > 1:
        filter_query = filter_query[:filter_query.rfind('|')] + ') &'
    else:
        filter_query = ''
    if len(novelties) > 0:
        for novelty in list(set(NOVELTY_IND_COLS) - set(novelties)):
            filter_query += f'{novelty} != 1 & '
    else:
        for novelty in NOVELTY_IND_COLS:
            filter_query += f'{novelty} != 1 & '
    if len(filter_query) > 0:
        filter_query = filter_query[:filter_query.rfind('&')]
    filtered_data = data.query(filter_query)
    return filtered_data


def evaluate(pred_file, gt_file):
    """Main function that calculates all the metrics on server side

    Args:
        pred_file (str): path to csv file containing pred labels
            Column 0 - file name
            Column 1..n - probability of data point being class 'n'. The last column will be considers as unknown class
        gt_file (str): path to csv file containing gt labels for each data point.
            Column 0 - file name
            Other columns will be accessed by name
    """
    assert os.path.isfile(pred_file), f'File not found: {pred_file}'
    assert os.path.isfile(gt_file), f'File not found: {gt_file}'

    gt_df = pd.read_csv(gt_file)
    # TODO Implement checks on format of pred file.
    pred_df = pd.read_csv(pred_file,
                          header=None,
                          index_col=None,
                          names=['path'] + list(map(str, range(1, CLASS_COUNT + 1))))

    results = dict()
    for metric in METRIC_LIST:
        unknown_data = filter_data(gt_df, METRIC_LIST[metric]['unknown'])
        unknown_data['binary_class'] = 1  # 1 - Unknown, 0 - Known
        unknown_data[GT_CLASS_COL] = 0  # We will always assign 0 to unknown class in multi class case
        known_data = filter_data(gt_df, METRIC_LIST[metric]['known'])
        known_data['binary_class'] = 0
        gt_data = pd.concat([unknown_data, known_data])
        data = gt_data.join(pred_df, how='inner', lsuffix='gt_', rsuffix='pred_')
        if METRIC_LIST[metric]['binary']:
            labels = np.asarray(data['binary_class'])
            preds = np.asarray(data[str(1)])  # Unknown probabilities should be the first column
        else:
            labels = np.asarray(data[GT_CLASS_COL])
            preds = np.asarray(data[METRIC_LIST[metric]['col_list']])
        results[metric] = METRIC_LIST[metric]['func'](preds, labels, gt_shift=METRIC_LIST[metric]['gt_shift'])
    return results


def main():
    parser = get_parser()
    args = parser.parse_args()
    res = evaluate(args.pred_file, args.gt_file)
    json.dump(res, open(f'{args.out_dir}/{args.model_name}.json', 'w'))


if __name__ == '__main__':
    main()
