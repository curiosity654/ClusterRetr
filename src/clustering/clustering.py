import pickle
import os
import numpy as np
import nanopq
from tqdm import tqdm
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from prettytable import PrettyTable

def expriment(X, gt_labels_gallery, predicted_features_query, gt_labels_query, M=1, Ks=256):
    pq = nanopq.PQ(M=M, verbose=False, Ks=Ks)

    # Train codewords
    pq.fit(X)
    labels = pq.labels.squeeze()
    NMI = normalized_mutual_info_score(gt_labels_gallery, labels)
    ARI = adjusted_rand_score(gt_labels_gallery, labels)

    return NMI, ARI

# savedir = "/root/code/SAKE/checkpoints/SAKE/sketchy/"
# savedir = "/root/code/SAKE/checkpoints/SAKE_kld/sketchy"
#/root/code/SAKE/checkpoints/SAKE_kld0.1/tuberlin"
savedir = "./quantization"
feature_file = os.path.join(savedir, 'features_zero.pickle')

with open(feature_file, 'rb') as fh:
    predicted_features_gallery, binary_features_gallery, gt_labels_gallery, \
    predicted_features_query, binary_features_query, gt_labels_query, \
    scores, _ = pickle.load(fh)

X = predicted_features_gallery

# Ms = [1, 2, 4, 8, 16]
Ms = [1]
Ks = [8, 16, 25, 32, 64, 128, 256]
tb_NMI = PrettyTable()
tb_ARI = PrettyTable()

tb_NMI.field_names = ["NMI"] + Ks
tb_ARI.field_names = ["ARI"] + Ks

for M in tqdm(Ms, position=1):
    row_NMI = []
    row_ARI = []
    for K in tqdm(Ks, position=0):
        NMI, ARI = expriment(X, gt_labels_gallery, predicted_features_query, gt_labels_query, M, K)
        # print("M={}, K={}, mAP={}, Prec={}".format(M, K, NMI, ARI))
        row_NMI.append(NMI)
        row_ARI.append(ARI)
    tb_NMI.add_row([str(M)] + row_NMI)
    tb_ARI.add_row([str(M)] + row_ARI)

print(tb_NMI)
print(tb_ARI)

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