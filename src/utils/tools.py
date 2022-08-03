import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import cdist
from torch.nn import Parameter
import math

def binarize(T, nb_classes):
    T = T.cpu().numpy()
    import sklearn.preprocessing
    T = sklearn.preprocessing.label_binarize(
        T, classes = range(0, nb_classes)
    )
    T = torch.FloatTensor(T).cuda()
    return T

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output

class Proxy_Anchor(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, mrg = 0.1, alpha = 32):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed).cuda())
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha
        
    def forward(self, X, T):
        P = self.proxies

        cos = F.linear(l2_norm(X), l2_norm(P))  # Calcluate cosine similarity
        P_one_hot = binarize(T = T, nb_classes = self.nb_classes)
        N_one_hot = 1 - P_one_hot
    
        pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
        neg_exp = torch.exp(self.alpha * (cos + self.mrg))

        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim = 0) != 0).squeeze(dim = 1)   # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)   # The number of positive proxies
        
        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0) 
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)
        
        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        loss = pos_term + neg_term     
        
        return loss

def proxy_synthesis(input_l2, proxy_l2, target, ps_alpha, ps_mu):
    '''
    input_l2: [batch_size, dims] l2-normalized embedding features
    proxy_l2: [n_classes, dims] l2-normalized proxy parameters
    target: [batch_size] Note that adjacent labels should be different (e.g., [0,1,2,3,4,5,...])
    ps_alpha: alpha for beta distribution
    ps_mu: generation ratio (# of synthetics / batch_size)
    '''

    input_list = [input_l2]
    proxy_list = [proxy_l2]
    target_list = [target]

    ps_rate = np.random.beta(ps_alpha, ps_alpha)

    input_aug = ps_rate * input_l2 + (1.0 - ps_rate) * torch.roll(input_l2, 1, dims=0)
    proxy_aug = ps_rate * proxy_l2[target,:] + (1.0 - ps_rate) * torch.roll(proxy_l2[target,:], 1, dims=0)
    input_list.append(input_aug)
    proxy_list.append(proxy_aug)
    
    n_classes = proxy_l2.shape[0]
    pseudo_target = torch.arange(n_classes, n_classes + input_l2.shape[0]).cuda()
    target_list.append(pseudo_target)

    embed_size = int(input_l2.shape[0] * (1.0 + ps_mu))
    proxy_size = int(n_classes + input_l2.shape[0] * ps_mu)
    input_large = torch.cat(input_list, dim=0)[:embed_size,:]
    proxy_large = torch.cat(proxy_list, dim=0)[:proxy_size,:]
    target = torch.cat(target_list, dim=0)[:embed_size]
    
    input_l2 = F.normalize(input_large, p=2, dim=1)
    proxy_l2 = F.normalize(proxy_large, p=2, dim=1)

    return input_l2, proxy_l2, target

class Norm_SoftMax(nn.Module):
    def __init__(self, input_dim, n_classes, scale=23.0, ps_mu=0.0, ps_alpha=0.0, proxy_init=None):
        super(Norm_SoftMax, self).__init__()
        self.scale = scale
        self.n_classes = n_classes
        self.ps_mu = ps_mu
        self.ps_alpha = ps_alpha
        self.proxy = Parameter(torch.Tensor(n_classes, input_dim))
        
        if proxy_init != None:
            self.proxy = Parameter(proxy_init.float())
            print("clip init")
        else:
            nn.init.kaiming_uniform_(self.proxy, a=math.sqrt(5))
        

    def forward(self, input, target):
        batch_size = input.size(0)
        input_l2 = F.normalize(input, p=2, dim=1)
        proxy_l2 = F.normalize(self.proxy, p=2, dim=1)
        
        if self.ps_mu > 0.0:
            input_l2, proxy_l2, target = proxy_synthesis(input_l2, proxy_l2, target,
                                                         self.ps_alpha, self.ps_mu)

        sim_mat = input_l2.matmul(proxy_l2.t())
        
        logits = self.scale * sim_mat
        
        loss = F.cross_entropy(logits, target)
        
        return logits[:batch_size, :self.n_classes], loss

class NormSoftmaxLoss(nn.Module):
    """
    L2 normalize weights and apply temperature scaling on logits.
    """
    def __init__(self,
                 dim,
                 num_classes,
                 temperature=0.05):
        super(NormSoftmaxLoss, self).__init__()

        self.weight = Parameter(torch.Tensor(num_classes, dim))
        # Initialization from nn.Linear (https://github.com/pytorch/pytorch/blob/v1.0.0/torch/nn/modules/linear.py#L129)
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

        self.temperature = temperature
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, embeddings, instance_targets):
        norm_weight = nn.functional.normalize(self.weight, dim=1)

        prediction_logits = nn.functional.linear(embeddings, norm_weight)

        loss = self.loss_fn(prediction_logits / self.temperature, instance_targets)
        return prediction_logits, loss

def ITQ(V, n_iter):
    # Main function for  ITQ which finds a rotation of the PCA embedded data
    # Input:
    #     V: nxc PCA embedded data, n is the number of images and c is the code length
    #     n_iter: max number of iterations, 50 is usually enough
    # Output:
    #     B: nxc binary matrix
    #     R: the ccc rotation matrix found by ITQ
    # Publications:
    #     Yunchao Gong and Svetlana Lazebnik. Iterative Quantization: A
    #     Procrustes Approach to Learning Binary Codes. In CVPR 2011.
    # Initialize with a orthogonal random rotation initialize with a orthogonal random rotation

    bit = V.shape[1]
    np.random.seed(n_iter)
    R = np.random.randn(bit, bit)
    U11, S2, V2 = np.linalg.svd(R)
    R = U11[:, :bit]

    # ITQ to find optimal rotation
    for iter in range(n_iter):
        Z = np.matmul(V, R)
        UX = np.ones((Z.shape[0], Z.shape[1])) * -1
        UX[Z >= 0] = 1
        C = np.matmul(np.transpose(UX), V)
        UB, sigma, UA = np.linalg.svd(C)
        R = np.matmul(UA, np.transpose(UB))

    # Make B binary
    B = UX
    B[B < 0] = 0
    return B, R

def compressITQ(Xtrain, Xtest, n_iter=50):
    # compressITQ runs ITQ
    # Center the data, VERY IMPORTANT
    Xtrain = Xtrain - np.mean(Xtrain, axis=0, keepdims=True)
    Xtest = Xtest - np.mean(Xtest, axis=0, keepdims=True)

    # PCA
    C = np.cov(Xtrain, rowvar=False)
    l, pc = np.linalg.eigh(C, 'U')
    idx = l.argsort()[::-1]
    pc = pc[:, idx]
    XXtrain = np.matmul(Xtrain, pc)
    XXtest = np.matmul(Xtest, pc)

    # ITQ
    _, R = ITQ(XXtrain, n_iter)

    Ctrain = np.matmul(XXtrain, R)
    Ctest = np.matmul(XXtest, R)

    Ctrain = Ctrain > 0
    Ctest = Ctest > 0

    return Ctrain, Ctest

def eval_AP_inner(inst_id, scores, gt_labels, top=None, sort_idx=None):
    pos_flag = gt_labels == inst_id
    tot = scores.shape[0]
    tot_pos = np.sum(pos_flag)
    
    if sort_idx is None:
        sort_idx = np.argsort(scores)
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

    for ii in range(len(mpre) - 2, -1, -1):
        mpre[ii] = max(mpre[ii], mpre[ii + 1])

    msk = [i != j for i, j in zip(mrec[1:], mrec[0:-1])]
    ap = np.sum((mrec[1:][msk] - mrec[0:-1][msk]) * mpre[1:][msk])
    return ap

def eval_precision(inst_id, scores, gt_labels, top=100, sort_idx=None):
    pos_flag = gt_labels == inst_id
    tot = scores.shape[0]
    
    top = min(top, tot)
    if sort_idx is None:
        sort_idx = np.argsort(scores)
    return np.sum(pos_flag[sort_idx][:top])/top

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class SoftCrossEntropy(nn.Module):
    def __init__(self):
        super(SoftCrossEntropy, self).__init__()
        return

    def forward(self, input_logits, target_logits, mask=None, mask_pos=None):
        """
        :param input_logits: prediction logits
        :param target_logits: target logits
        :return: loss
        """
        log_likelihood = - F.log_softmax(input_logits, dim=1)
        
        if mask_pos is not None:
            target_logits = target_logits + mask_pos
        
        if mask is None:
            sample_num, class_num = target_logits.shape
            loss = torch.sum(torch.mul(log_likelihood, F.softmax(target_logits, dim=1)))/sample_num
        else:
            sample_num = torch.sum(mask)
            loss = torch.sum(torch.mul(torch.mul(log_likelihood, F.softmax(target_logits, dim=1)), mask))/sample_num

        return loss

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True