import pickle
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def eval(feature_file, fig_name):
    with open(feature_file, 'rb') as fh:
        predicted_features_gallery, binary_features_gallery, gt_labels_gallery, \
        predicted_features_query, binary_features_query, gt_labels_query, \
        scores, binary_scores = pickle.load(fh)

    scores = -scores
    matrix_query = np.expand_dims(gt_labels_query,1).repeat(len(gt_labels_gallery), axis=1)
    matrix_gallery = np.expand_dims(gt_labels_gallery,0).repeat(len(gt_labels_query), axis=0)
    pair_mask = matrix_query == matrix_gallery
    scores_flatten = scores.flatten()[::10000]
    pair_scores_flatten = scores[pair_mask].flatten()[::1000]
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.histplot(scores_flatten, ax=ax, color='b')
    sns.histplot(pair_scores_flatten, ax=ax, color='g')
    plt.legend(labels=["scores unpaired","scores paired"])
    plt.savefig('{}.png'.format(fig_name), dpi=200)

def main():
    parser = argparse.ArgumentParser(description="Evaluate the feature of models")
    parser.add_argument('--feature_file', '-f')
    parser.add_argument('--fig_name', '-n')
    parser.add_argument('--all', action="store_true")
    args = parser.parse_args()

    if os.path.isfile(args.feature_file):
        eval(args.feature_file, args.fig_name)

if __name__ == "__main__":
    main()