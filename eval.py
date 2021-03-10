import argparse
import os

import numpy as np
import pandas as pd


def get_parser():
    """"Defines the command line arguments"""
    parser = argparse.ArgumentParser(description='Open World Vision')
    parser.add_argument('--pred_file', required=True, type=str,
                        help='path to csv file containing predicted labels. First column should contain image name and'
                             'rest of the columns should contain predicted probabilities for each of the class_id')
    parser.add_argument('--gt_file', required=True, type=str,
                        help='path to csv file containing ground-truth labels. First column should contain image name'
                             ' and second column should contain the ground truth class_id')
    parser.add_argument('--k_vals', default=1, nargs='+', type=int,
                        help='space separated list of k(s) in top-k evaluation')
    return parser


def check_class_validity(p_class, gt_class):
    """
    Check the validity of the inputs for image classification. Raises an exception if invalid.

    Args:
        p_class (np.array): Nx(K+1) matrix with each row corresponding to K+1 class probabilities for each sample
        gt_class (np.array) : Nx1 vector with ground-truth class for each sample
    """
    if p_class.shape[0] != gt_class.shape[0]:
        raise Exception(
            'Number of predicted samples not equal to number of ground-truth samples!')
    if np.any(p_class < 0) or np.any(p_class > 1):
        raise Exception(
            'Predicted class probabilities should be between 0 and 1!')
    if p_class.ndim != 2:
        raise Exception(
            'Predicted probabilities must be a 2D matrix but is an array of dimension {}!'.format(p_class.ndim))
    if np.max(gt_class) >= p_class.shape[1] or np.min(gt_class < 0):
        raise Exception(
            'Ground-truth class labels must lie in the range [0-{}]!'.format(p_class.shape[1]))
    return


def top_k_accuracy(p_class, gt_class, k, mode='all'):
    """
    The method computes top-K accuracy.

    Args:
        p_class: Nx(K+1) matrix with each row corresponding to K+1 class probabilities for each sample
        gt_class: Nx1 compute vector with ground-truth class for each sample
        k: 'k' used in top-K accuracy
    Returns:
        top-K accuracy
    """
    def calc_accuracy(pred, gt, k):
        pred = np.argsort(-pred)[:, :k]
        gt = gt[:, np.newaxis]
        check_zero = pred - gt
        correct = np.sum(np.any(check_zero == 0, axis=1).astype(int))
        return round(float(correct)/pred.shape[0], 5)

    check_class_validity(p_class, gt_class)
    accuracy = dict()
    if mode == 'all':
        accuracy['all'] = calc_accuracy(p_class, gt_class, k)
    else:
        class_list = sorted(np.unique(gt_class))
        for class_id in class_list:
            class_inds = np.where(gt_class == class_id)
            class_gt = gt_class[class_inds]
            class_p = p_class[class_inds]
            accuracy[class_id] = calc_accuracy(class_p, class_gt, k)
    return accuracy


def evaluate(pred_file, gt_file, k=(1, 5,)):
    """Evaluates the performance as top-k accuracy

    Args:
        pred_file (str): path to csv file containing pred labels
            Column 0 - file name
            Column 1..n - probability of data point being class 'n'
        gt_file (str): path to csv file containing gt labels for each data point.
            Column 0 - file name
            Column 1 - class_id
        k (tuple of int): 'k' in 'top-k' accuracy. 'k' top probabilities will be used for evaluation
    """
    assert os.path.isfile(pred_file), f'File not found: {pred_file}'
    assert os.path.isfile(gt_file), f'File not found: {gt_file}'

    gt_df = pd.read_csv(gt_file)
    pred_df = pd.read_csv(pred_file, header=None, index_col=None)
    # Comparing the full file paths as of now, we might want to revisit this based on how we name data points later
    assert len(gt_df) == len(pred_df), 'GT and Prediction files have different number of records'
    assert gt_df.iloc[:, 0].tolist() == pred_df.iloc[:, 0].tolist(), 'GT and Prediction files do not have the data' \
                                                                     ' points in same order'
    gt_class = np.asarray(gt_df.iloc[:, 1].tolist())
    pred_df.drop(pred_df.columns[0], inplace=True, axis=1)
    pred_class = pred_df.to_numpy()
    for k_val in k:
        accuracy = top_k_accuracy(pred_class, gt_class, k_val)
        print(f'The top-{k_val} average accuracy is {accuracy}')
        class_accuracy = top_k_accuracy(pred_class, gt_class, k_val, mode='class')
        for class_id in sorted(class_accuracy.keys()):
            print(f'\t\t The top-{k_val} accuracy for class {class_id} is {class_accuracy[class_id]}')


def main():
    parser = get_parser()
    args = parser.parse_args()
    evaluate(args.pred_file, args.gt_file, tuple(map(int, args.k_vals)))


if __name__ == '__main__':
    main()
