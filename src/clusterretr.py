import argparse
import pickle
import numpy as np
from utils import nanopq
from utils.eval_tools import eval_precision, eval_AP_inner, compressITQ
from scipy.spatial.distance import cdist
from time import time

def expriment(X, gt_labels_gallery, predicted_features_query, gt_labels_query, _, M=1, Ks=256, **args):
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
    
    mAP_binary = np.array([np.nanmean(maps) for maps in mAP_ls_binary]).mean()
    
    prec_ls_binary = [[] for _ in range(len(np.unique(gt_labels_query)))]
    for fi in range(predicted_features_query.shape[0]):
        prec_binary = eval_precision(gt_labels_query[fi], binary_scores[fi], gt_labels_gallery)
        prec_ls_binary[gt_labels_query[fi]].append(prec_binary)

    prec_binary = np.array([np.nanmean(pre) for pre in prec_ls_binary]).mean()
    print("{}, {}, {}, {},".format(M, Ks, mAP_binary, prec_binary))
    return mAP_binary, prec_binary

def fuse_expriment(X, gt_labels_gallery, predicted_features_query, gt_labels_query, _, M=1, Ks=256, lam=0.2):
    pq = nanopq.PQ(M=M, verbose=False, Ks=Ks)

    pq.fit(X)
    scores = cdist(predicted_features_query, X)
    X_code = pq.encode(X)
    binary_scores = []
    for query in predicted_features_query:
        dists = pq.dtable(query).adist(X_code)
        binary_scores.append(dists)

    binary_scores = np.stack(binary_scores)
    binary_scores = (1-lam) * binary_scores + lam * scores

    mAP_ls_binary = [[] for _ in range(len(np.unique(gt_labels_query)))]
    for fi in range(predicted_features_query.shape[0]):
        mapi_binary = eval_AP_inner(gt_labels_query[fi], binary_scores[fi], gt_labels_gallery)
        mAP_ls_binary[gt_labels_query[fi]].append(mapi_binary)
    
    mAP_binary = np.array([np.nanmean(maps) for maps in mAP_ls_binary]).mean()
    
    prec_ls_binary = [[] for _ in range(len(np.unique(gt_labels_query)))]
    for fi in range(predicted_features_query.shape[0]):
        prec_binary = eval_precision(gt_labels_query[fi], binary_scores[fi], gt_labels_gallery)
        prec_ls_binary[gt_labels_query[fi]].append(prec_binary)

    prec_binary = np.array([np.nanmean(pre) for pre in prec_ls_binary]).mean()
    print("{}, {}, {}, {},".format(M, Ks, mAP_binary, prec_binary))
    return mAP_binary, prec_binary

def itq_fuse_expriment(X, gt_labels_gallery, predicted_features_query, gt_labels_query, _, M=1, Ks=256, lam=0.2):
    pq = nanopq.PQ(M=M, verbose=False, Ks=Ks)

    pq_X = pq.fit(X)
    binary_features_query, binary_features_gallery = compressITQ(predicted_features_query, pq_X)
    binary_scores = cdist(binary_features_query, binary_features_gallery)
    scores = cdist(predicted_features_query, X)

    binary_scores = np.stack(binary_scores)
    # lam = args["lam"]
    binary_scores = (1-lam) * binary_scores + lam * scores

    mAP_ls_binary = [[] for _ in range(len(np.unique(gt_labels_query)))]
    for fi in range(predicted_features_query.shape[0]):
        mapi_binary = eval_AP_inner(gt_labels_query[fi], binary_scores[fi], gt_labels_gallery)
        mAP_ls_binary[gt_labels_query[fi]].append(mapi_binary)
    
    mAP_binary = np.array([np.nanmean(maps) for maps in mAP_ls_binary]).mean()
    
    prec_ls_binary = [[] for _ in range(len(np.unique(gt_labels_query)))]
    for fi in range(predicted_features_query.shape[0]):
        prec_binary = eval_precision(gt_labels_query[fi], binary_scores[fi], gt_labels_gallery)
        prec_ls_binary[gt_labels_query[fi]].append(prec_binary)

    prec_binary = np.array([np.nanmean(pre) for pre in prec_ls_binary]).mean()
    print("{}, {}, {}, {},".format(M, Ks, mAP_binary, prec_binary))
    return mAP_binary, prec_binary

def itq_expriment(X, gt_labels_gallery, predicted_features_query, gt_labels_query, _, M=1, Ks=256, **args):
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

def reranking_expriment(X, gt_labels_gallery, predicted_features_query, gt_labels_query, scores, M=1, Ks=256, **args):
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
    
    mAP_binary = np.array([np.nanmean(maps) for maps in mAP_ls_binary]).mean()
    
    prec_ls_binary = [[] for _ in range(len(np.unique(gt_labels_query)))]
    for fi in range(predicted_features_query.shape[0]):
        prec_binary = eval_precision(gt_labels_query[fi], binary_scores[fi], gt_labels_gallery, sort_idx=sort_idxs[fi])
        prec_ls_binary[gt_labels_query[fi]].append(prec_binary)

    prec_binary = np.array([np.nanmean(pre) for pre in prec_ls_binary]).mean()
    print("{}, {}, {}, {},".format(M, Ks, mAP_binary, prec_binary))
    return mAP_binary, prec_binary

def run_experiments(expriment, predicted_features_gallery, gt_labels_gallery, predicted_features_query, gt_labels_query, scores, Ms, Ks, lam=None, Ds=None):
    from multiprocessing import Process
    processes = []
    for M in Ms:
        for K in Ks:
            if lam is not None:
                exp = Process(target=expriment, args=(predicted_features_gallery, gt_labels_gallery, predicted_features_query, gt_labels_query, scores, M, K, lam))
                processes.append(exp)
            elif Ds is not None:
                for D in Ds:
                    exp = Process(target=expriment, args=(predicted_features_gallery, gt_labels_gallery, predicted_features_query, gt_labels_query, scores, M, K, D))
                    processes.append(exp)
            else:
                exp = Process(target=expriment, args=(predicted_features_gallery, gt_labels_gallery, predicted_features_query, gt_labels_query, scores, M, K))
                processes.append(exp)

    [p.start() for p in processes]
    [p.join() for p in processes]

def time_real_computation(predicted_features_gallery, predicted_features_query):
    start = time()
    dist = cdist(predicted_features_query, predicted_features_gallery)
    end = time()

    print("computing dist for shape {}, {}, used: {}s".format(predicted_features_query.shape, predicted_features_gallery.shape, end-start))
    return

def time_pq_computation(predicted_features_gallery, predicted_features_query, M, K):
    pq = nanopq.PQ(M=M, verbose=False, Ks=K)
    pq.fit(predicted_features_gallery)
    X_code = pq.encode(predicted_features_gallery)
    binary_scores = []

    start = time()
    for query in predicted_features_query:
        dists = pq.dtable(query).adist(X_code)
        binary_scores.append(dists)
    end = time()

    print("{}, {}, {},".format(M, K, end-start))
    return

if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='Clustering Retrieval')
    parser.add_argument('--feature_file', '-s',  metavar='DIR', default='checkpoints/features_baseline.pickle')
    parser.add_argument('--itq', action='store_true', default=False)
    parser.add_argument('--rerank', action='store_true', default=False)
    parser.add_argument('--real', action='store_true', default=False)
    parser.add_argument('--fuse', type=float, default=-1)
    parser.add_argument('--extend', type=int, default=0)
    parser.add_argument('--time', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)

    args = parser.parse_args()
    with open(args.feature_file, 'rb') as fh:
        predicted_features_gallery, gt_labels_gallery, \
        predicted_features_query, gt_labels_query, \
        scores, = pickle.load(fh)
        
    Ks = [8, 16, 25, 32, 64]
    Ms = [1, 2, 4, 8, 16]
    # Ms = [2]
    # Ks = [32]
    # Ks = [8, 16, 32]
    if args.time:
        if args.real:
            time_real_computation(predicted_features_gallery, predicted_features_query)
        else:
            print("M, K, Time,")
            for M in Ms:
                for K in Ks:
                    time_pq_computation(predicted_features_gallery, predicted_features_query, M, K)
    else:
        print("M, K, mAP, Prec,")

    if args.extend >= 0:
        seed = 123
        np.random.seed(seed)
        org_channels = np.arange(predicted_features_gallery.shape[1])
        shuffle_channels = np.random.permutation(org_channels)[:args.extend]
        predicted_features_gallery = predicted_features_gallery[:, np.concatenate([org_channels, shuffle_channels])]
        predicted_features_query = predicted_features_query[:, np.concatenate([org_channels, shuffle_channels])]
    
    if args.itq and args.fuse > 0:
        run_experiments(itq_fuse_expriment, predicted_features_gallery, gt_labels_gallery, predicted_features_query, gt_labels_query, scores, Ms, Ks, lam=args.fuse)
    elif args.itq:
        run_experiments(itq_expriment, predicted_features_gallery, gt_labels_gallery, predicted_features_query, gt_labels_query, scores, Ms, Ks)
    elif args.rerank:
        run_experiments(reranking_expriment, predicted_features_gallery, gt_labels_gallery, predicted_features_query, gt_labels_query, scores, Ms, Ks)
    elif args.fuse > 0:
        run_experiments(fuse_expriment, predicted_features_gallery, gt_labels_gallery, predicted_features_query, gt_labels_query, scores, Ms, Ks, lam=args.fuse)
    elif args.debug:
        expriment(predicted_features_gallery, gt_labels_gallery, predicted_features_query, gt_labels_query, 1, 256)
    else:
        run_experiments(expriment, predicted_features_gallery, gt_labels_gallery, predicted_features_query, gt_labels_query, scores, Ms, Ks)