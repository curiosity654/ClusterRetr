import os
import pickle
import argparse
from scipy.spatial.distance import cdist

parser = argparse.ArgumentParser(description='Prepocessing')
parser.add_argument('--feature_file', '-f', default='./data/features_kld.pickle')
args = parser.parse_args()

with open(args.feature_file, 'rb') as fh:
    predicted_features_gallery, gt_labels_gallery, \
    predicted_features_query, gt_labels_query = pickle.load(fh)

scores = cdist(predicted_features_query, predicted_features_gallery)

with open('./data/features_processed.pickle', 'wb') as fh:
        pickle.dump([predicted_features_gallery, gt_labels_gallery, \
                     predicted_features_query, gt_labels_query, scores], fh)