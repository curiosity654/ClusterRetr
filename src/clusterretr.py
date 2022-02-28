import argparse
import pickle
import os
import numpy as np
from utils import nanopq
from utils.tools import eval_precision, eval_AP_inner, compressITQ
from scipy.spatial.distance import cdist
from tqdm import tqdm
# import wandb

def expriment(X, gt_labels_gallery, predicted_features_query, gt_labels_query, M=1, Ks=256):
    pq = nanopq.PQ(M=M, verbose=False, Ks=Ks)

    pq.fit(X)

    X_code = pq.encode(X)
    binary_scores = []
    for query in predicted_features_query:
        dists = pq.dtable(query).adist(X_code)
        binary_scores.append(dists)

    binary_scores = np.stack(binary_scores)

    mAP_ls_binary = [[] for _ in range(len(np.unique(gt_labels_query)))]
    for fi in range(predicted_features_query.shape[0]):
        mapi_binary = eval_AP_inner(gt_labels_query[fi], binary_scores[fi], gt_labels_gallery)
        mAP_ls_binary[gt_labels_query[fi]].append(mapi_binary)
    
    # for mAPi,mAPs in enumerate(mAP_ls):
    #     print(str(mAPi)+' '+str(np.nanmean(mAPs))+' '+str(np.nanstd(mAPs)))
    mAP_binary = np.array([np.nanmean(maps) for maps in mAP_ls_binary]).mean()
    # print('mAP - hash: {:.4f}'.format(mAP_binary))
    # wandb.log({"test/sketchy/mAP/all_binary": mAP_binary})

    prec_ls_binary = [[] for _ in range(len(np.unique(gt_labels_query)))]
    for fi in range(predicted_features_query.shape[0]):
        prec_binary = eval_precision(gt_labels_query[fi], binary_scores[fi], gt_labels_gallery)
        prec_ls_binary[gt_labels_query[fi]].append(prec_binary)

    prec_binary = np.array([np.nanmean(pre) for pre in prec_ls_binary]).mean()
    # print('Precision - hash: {:.4f}'.format(prec_binary))
    # wandb.log({"test/sketchy/precision/all_binary": prec_binary})
    print("{}, {}, {}, {},".format(M, Ks, mAP_binary, prec_binary))
    return mAP_binary, prec_binary

def itq_expriment(X, gt_labels_gallery, predicted_features_query, gt_labels_query, M=1, Ks=256):
    pq = nanopq.PQ(M=M, verbose=False, Ks=Ks)

    pq_X = pq.fit(X)
    binary_features_query, binary_features_gallery = compressITQ(predicted_features_query, pq_X)
    binary_scores = cdist(binary_features_query, binary_features_gallery)

    mAP_ls_binary = [[] for _ in range(len(np.unique(gt_labels_query)))]
    for fi in range(predicted_features_query.shape[0]):
        mapi_binary = eval_AP_inner(gt_labels_query[fi], binary_scores[fi], gt_labels_gallery)
        mAP_ls_binary[gt_labels_query[fi]].append(mapi_binary)
    
    mAP_binary = np.array([np.nanmean(maps) for maps in mAP_ls_binary]).mean()
    # print('mAP - hash: {:.4f}'.format(mAP_binary))

    prec_ls_binary = [[] for _ in range(len(np.unique(gt_labels_query)))]
    for fi in range(predicted_features_query.shape[0]):
        prec_binary = eval_precision(gt_labels_query[fi], binary_scores[fi], gt_labels_gallery)
        prec_ls_binary[gt_labels_query[fi]].append(prec_binary)

    prec_binary = np.array([np.nanmean(pre) for pre in prec_ls_binary]).mean()
    # print('Precision - hash: {:.4f}'.format(prec_binary))
    print("{}, {}, {}, {},".format(M, Ks, mAP_binary, prec_binary))
    return mAP_binary, prec_binary

def reranking_expriment(X, gt_labels_gallery, predicted_features_query, gt_labels_query, scores, M=1, Ks=256):
    pq = nanopq.PQ(M=M, verbose=False, Ks=Ks)

    pq.fit(X)

    X_code = pq.encode(X)
    binary_scores = []
    sort_idxs = []
    for query, score in zip(predicted_features_query, scores):
        # Results: create a distance table online, and compute Asymmetric Distance to each PQ-code 
        dists = pq.dtable(query).adist(X_code)  # (10000, ) 
        uni, indices = np.unique(dists, return_inverse=True)
        offset = 0
        sort_idx = np.zeros_like(score, dtype=int)
        for i, dist in enumerate(uni):
            subid = np.argsort(-score[indices == i])
            subrank = np.zeros_like(subid)
            for s, id in enumerate(subid):
                subrank[id] = s
            subrank += offset
            offset += len(subrank)
            sort_idx[indices == i] = subrank
        sort_idx = np.argsort(sort_idx)
        sort_idxs.append(sort_idx)
        binary_scores.append(dists)

    binary_scores = np.stack(binary_scores)
    
    mAP_ls_binary = [[] for _ in range(len(np.unique(gt_labels_query)))]
    for fi in range(predicted_features_query.shape[0]):
        mapi_binary = eval_AP_inner(gt_labels_query[fi], binary_scores[fi], gt_labels_gallery, sort_idx=sort_idxs[fi])
        mAP_ls_binary[gt_labels_query[fi]].append(mapi_binary)
    
    # for mAPi,mAPs in enumerate(mAP_ls):
    #     print(str(mAPi)+' '+str(np.nanmean(mAPs))+' '+str(np.nanstd(mAPs)))
    mAP_binary = np.array([np.nanmean(maps) for maps in mAP_ls_binary]).mean()
    # print('mAP - hash: {:.4f}'.format(mAP_binary))
    # wandb.log({"test/sketchy/mAP/all_binary": mAP_binary})

    prec_ls_binary = [[] for _ in range(len(np.unique(gt_labels_query)))]
    for fi in range(predicted_features_query.shape[0]):
        prec_binary = eval_precision(gt_labels_query[fi], binary_scores[fi], gt_labels_gallery, sort_idx=sort_idxs[fi])
        prec_ls_binary[gt_labels_query[fi]].append(prec_binary)

    prec_binary = np.array([np.nanmean(pre) for pre in prec_ls_binary]).mean()
    # print('Precision - hash: {:.4f}'.format(prec_binary))
    # wandb.log({"test/sketchy/precision/all_binary": prec_binary})
    print("{}, {}, {}, {},".format(M, Ks, mAP_binary, prec_binary))
    return mAP_binary, prec_binary

def run_experiments(expriment, predicted_features_gallery, gt_labels_gallery, predicted_features_query, gt_labels_query, Ms, Ks):
    from multiprocessing import Process
    processes = []
    for M in Ms:
        for K in Ks:
            exp = Process(target=expriment, args=(predicted_features_gallery, gt_labels_gallery, predicted_features_query, gt_labels_query, M, K,))
            processes.append(exp)

    [p.start() for p in processes]
    [p.join() for p in processes]

if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='Clustering Retrieval')
    parser.add_argument('--feature_file', '-s',  metavar='DIR', default='checkpoints/features_baseline.pickle')
    parser.add_argument('--itq', action='store_true', default=False)
    parser.add_argument('--rerank', action='store_true', default=False)

    args = parser.parse_args()
    print(args)
    with open(args.feature_file, 'rb') as fh:
        predicted_features_gallery, binary_features_gallery, gt_labels_gallery, \
        predicted_features_query, binary_features_query, gt_labels_query, \
        scores, _ = pickle.load(fh)

    Ms = [1, 2, 4, 8, 16]
    Ks = [8, 16, 25, 32, 64, 128, 256]
    print("M, K, mAP, Prec,")
    if args.itq:
        run_experiments(itq_expriment, predicted_features_gallery, gt_labels_gallery, predicted_features_query, gt_labels_query, Ms, Ks)
    elif args.rerank:
        run_experiments(reranking_expriment, predicted_features_gallery, gt_labels_gallery, predicted_features_query, gt_labels_query, Ms, Ks)
    else:
        run_experiments(expriment, predicted_features_gallery, gt_labels_gallery, predicted_features_query, gt_labels_query, Ms, Ks)
    # expriment(predicted_features_gallery, gt_labels_gallery, predicted_features_query, gt_labels_query, 1, 256)