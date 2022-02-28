import pickle
import os
import numpy as np
import sys
sys.path.append("./utils")
import nanopq
from tqdm import tqdm
import wandb

def eval_AP_inner(inst_id, scores, gt_labels, top=None):
    pos_flag = gt_labels == inst_id
    tot = scores.shape[0]
    tot_pos = np.sum(pos_flag)
    
    sort_idx = np.argsort(-scores)
    tp = pos_flag[sort_idx]
    fp = np.logical_not(tp)
    
    if top is not None:
        top = min(top, tot)
        tp = tp[:top]
        fp = fp[:top]
        tot_pos = min(top, tot_pos)
    
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    try:
        rec = tp / tot_pos
        prec = tp / (tp + fp)
    except:
        print(inst_id, tot_pos)
        return np.nan

    ap = VOCap(rec, prec)
    return ap

def VOCap(rec, prec):
    mrec = np.append(0, rec)
    mrec = np.append(mrec, 1)
    
    mpre = np.append(0, prec)
    mpre = np.append(mpre, 0)
    
    for ii in range(len(mpre)-2,-1,-1):
        mpre[ii] = max(mpre[ii], mpre[ii+1])
        
    msk = [i!=j for i,j in zip(mrec[1:], mrec[0:-1])]
    ap = np.sum((mrec[1:][msk]-mrec[0:-1][msk])*mpre[1:][msk])
    return ap


def eval_precision(inst_id, scores, gt_labels, top=100):
    pos_flag = gt_labels == inst_id
    tot = scores.shape[0]
    
    top = min(top, tot)
    
    sort_idx = np.argsort(-scores)
    return np.sum(pos_flag[sort_idx][:top])/top

def expriment(X, gt_labels_gallery, predicted_features_query, gt_labels_query, M=1, Ks=256):
    pq = nanopq.PQ(M=M, verbose=False, Ks=Ks)

    # Train codewords
    pq.fit(X)

    # Encode to PQ-codes
    X_code = pq.encode(X)
    binary_scores = []
    for query in predicted_features_query:
        # Results: create a distance table online, and compute Asymmetric Distance to each PQ-code 
        dists = pq.dtable(query).adist(X_code)  # (10000, ) 
        binary_scores.append(dists)

    binary_scores = - np.stack(binary_scores)

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
    print("M={}, K={}, mAP={}, Prec={}".format(M, Ks, mAP_binary, prec_binary))
    return mAP_binary, prec_binary

savedir = "./quantization"
# savedir = "/root/code/SAKE/checkpoints/SAKE_kld/sketchy"
feature_file = os.path.join(savedir, 'features_kld.pickle')

with open(feature_file, 'rb') as fh:
    predicted_features_gallery, binary_features_gallery, gt_labels_gallery, \
    predicted_features_query, binary_features_query, gt_labels_query, \
    scores, _ = pickle.load(fh)

# savedir = "./checkpoints/SAKE/quickdraw/random0"
# feature_file = os.path.join(savedir, 'features_zero.pickle')
# predicted_features_gallery_file = os.path.join(savedir, 'predicted_features_gallery.pickle')
# gt_labels_gallery_file = os.path.join(savedir, 'gt_labels_gallery.pickle')
# predicted_features_query_file = os.path.join(savedir, 'predicted_features_query.pickle')
# gt_labels_query_file = os.path.join(savedir, 'gt_labels_query.pickle')

# with open(predicted_features_gallery_file, "rb") as fh:
#     predicted_features_gallery = pickle.load(fh)

# with open(gt_labels_gallery_file, "rb") as fh:
#     gt_labels_gallery = pickle.load(fh)

# with open(predicted_features_query_file, "rb") as fh:
#     predicted_features_query = pickle.load(fh)

# with open(gt_labels_query_file, "rb") as fh:
#     gt_labels_query = pickle.load(fh)

binary_scores = []

X = predicted_features_gallery

Ms = [1, 2, 4, 8, 16]
Ks = [8, 16, 25, 32, 64, 128, 256]
# Ms = [1, 2, 4, 16]
# Ks = [32]
# tb_mAP = PrettyTable()
# tb_Prec = PrettyTable()

# tb_mAP.field_names = ["mAP"] + Ks
# tb_Prec.field_names = ["Prec"] + Ks

# for M in tqdm(Ms, position=1):
#     row_mAP = []
#     row_Prec = []
#     for K in tqdm(Ks, position=0):
#         mAP, Prec = expriment(X, gt_labels_gallery, predicted_features_query, gt_labels_query, M, K)
#         print("M={}, K={}, mAP={}, Prec={}".format(M, K, mAP, Prec))
    #     row_mAP.append(mAP)
    #     row_Prec.append(Prec)
    # tb_mAP.add_row([str(M)] + row_mAP)
    # tb_Prec.add_row([str(M)] + row_Prec)

# print(tb_mAP)
# print(tb_Prec)

def run_experiments():
    from multiprocessing import Process
    processes = []
    for M in Ms:
        for K in Ks:
            exp = Process(target=expriment, args=(X, gt_labels_gallery, predicted_features_query, gt_labels_query, M, K,))
            processes.append(exp)

    [p.start() for p in processes]
    [p.join() for p in processes]

if __name__ =='__main__':
    run_experiments()