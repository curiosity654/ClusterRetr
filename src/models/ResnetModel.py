from numpy import int32, int8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import pretrainedmodels
from models.senet import cse_resnet50, cse_resnet50_hashing
from models.resnet import resnet50_hashing
import torch.nn.functional as F

class ResnetModel(nn.Module):
    def __init__(self, arch, num_classes, pretrained=True, freeze_features=False):
        super(ResnetModel, self).__init__()
        
        self.num_classes = num_classes
        self.modelName = arch
        
        original_model = models.__dict__[arch](pretrained=pretrained)
        self.features = nn.Sequential(*list(original_model.children())[:-3])
        self.last_block = nn.Sequential(*list(original_model.children())[-3:-1])
        self.linear = nn.Linear(in_features=2048, out_features=num_classes)
        
        # Freeze the resnet layers
        if freeze_features:
            for ff in self.features:
                for pp in ff.parameters():
                    pp.requires_grad = False
                    
    def forward(self, x):
        out = self.features(x)
        out = self.last_block(out)
        out = out.view(out.size()[0],-1)
        out = self.linear(out)
        return out
    
    
class ResnetModel_KDHashing(nn.Module):
    def __init__(self, arch, hashing_dim, num_classes, pretrained=True, freeze_features=False, ems=False):
        super(ResnetModel_KDHashing, self).__init__()
        self.hashing_dim = hashing_dim
        self.num_classes = num_classes
        self.modelName = arch
        
        if pretrained:
            self.original_model = resnet50_hashing(self.hashing_dim)
        else:
            self.original_model = resnet50_hashing(self.hashing_dim, pretrained=False)
        
        self.ems = ems
        if self.ems:
            print('Error, no ems implementationin AlexnetModel_KDHashing')
            return None
        else:
            self.linear = nn.Linear(in_features=hashing_dim, out_features=num_classes)
        
        # Freeze the resnet layers
        if freeze_features:
            print('Error, no freeze_features implementationin AlexnetModel_KDHashing')
            return None
        
    
    def forward(self, x):
        out_o = self.original_model.features(x)
        out_o = self.original_model.hashing(out_o)
        
        out = self.linear(out_o)
        out_kd = self.original_model.logits(out_o)

#         x_norm = out_o.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
#         normed = out_o / x_norm
        
#         out = F.linear(normed, F.normalize(self.linear.weight), None) * x_norm
#         out_kd = F.linear(normed, F.normalize(self.original_model.last_linear.weight), None) * x_norm

        return out,out_kd
    
    
class SEResnetModel(nn.Module):
    def __init__(self, arch, num_classes, pretrained=True, freeze_features=False, ems=False):
        super(SEResnetModel, self).__init__()
        
        self.num_classes = num_classes
        self.modelName = arch
        
        if pretrained:
            original_model = pretrainedmodels.__dict__[arch](num_classes=1000, pretrained='imagenet')
        else:
            original_model = pretrainedmodels.__dict__[arch](num_classes=1000, pretrained=None)
            
        self.features = nn.Sequential(*list(original_model.children())[:-3])
        self.last_block = nn.Sequential(*list(original_model.children())[-3:-1])
        
        self.ems = ems
        if self.ems:
            self.linear = EMSLayer(num_classes, 2048)
        else:
            self.linear = nn.Linear(in_features=2048, out_features=num_classes)
        
        # Freeze the resnet layers
        if freeze_features:
            for ff in self.features:
                for pp in ff.parameters():
                    pp.requires_grad = False
                    
        
                    
    def forward(self, x):
        out = self.features(x)
        out = self.last_block(out)
        out = out.view(out.size()[0],-1)
        out = self.linear(out)
        return out
    

class SEResnetModel_KD(nn.Module):
    def __init__(self, arch, num_classes, pretrained=True, freeze_features=False, ems=False):
        super(SEResnetModel_KD, self).__init__()
        
        self.num_classes = num_classes
        self.modelName = arch
        
        if pretrained:
            original_model = pretrainedmodels.__dict__[arch](num_classes=1000, pretrained='imagenet')
        else:
            original_model = pretrainedmodels.__dict__[arch](num_classes=1000, pretrained=None)
            
        self.features = nn.Sequential(*list(original_model.children())[:-3])
        self.last_block = nn.Sequential(*list(original_model.children())[-3:-1])
        self.original_output = nn.Sequential(*list(original_model.children())[-1:])
        
        self.ems = ems
        if self.ems:
            self.linear = EMSLayer(num_classes, 2048)
        else:
            self.linear = nn.Linear(in_features=2048, out_features=num_classes)
        
        # Freeze the resnet layers
        if freeze_features:
            for ff in self.features:
                for pp in ff.parameters():
                    pp.requires_grad = False
                    
        
                    
    def forward(self, x):
        out_o = self.features(x)
        out_o = self.last_block(out_o)
        out_o = out_o.view(out_o.size()[0],-1)

        out = self.linear(out_o)
        out_kd = self.original_output(out_o)

        return out,out_kd
    
    
class CSEResnetModel_KD(nn.Module):
    def __init__(self, arch, num_classes, pretrained=True, freeze_features=False, ems=False):
        super(CSEResnetModel_KD, self).__init__()
        
        self.num_classes = num_classes
        self.modelName = arch
        
        if pretrained:
            self.original_model = cse_resnet50()
        else:
            self.original_model = cse_resnet50(pretrained=None)
            
        
        self.ems = ems
        if self.ems:
            self.linear = EMSLayer(num_classes, 2048)
        else:
            self.linear = nn.Linear(in_features=2048, out_features=num_classes)
        
        # Freeze the resnet layers
        if freeze_features:
            for ff in self.features:
                for pp in ff.parameters():
                    pp.requires_grad = False
                    
                    
    def forward(self, x, y):
        out_o = self.original_model.features(x,y)
        out = nn.AdaptiveAvgPool2d(1)(out_o)
        out = out.view(out.size()[0],-1)
        out = self.linear(out)
        
        out_kd = self.original_model.logits(out_o)
        return out,out_kd
    
class CSEResnetModel_KDHashing(nn.Module):
    def __init__(self, arch, hashing_dim, num_classes, pretrained=True, freeze_features=False, ems=False):
        super(CSEResnetModel_KDHashing, self).__init__()
        
        self.hashing_dim = hashing_dim
        self.num_classes = num_classes
        self.modelName = arch
        
        if pretrained:
            self.original_model = cse_resnet50_hashing(self.hashing_dim)
        else:
            self.original_model = cse_resnet50_hashing(self.hashing_dim, pretrained=None)
            
        
        self.ems = ems
        if self.ems:
            self.linear = EMSLayer(num_classes, hashing_dim)
        else:
            self.linear = nn.Linear(in_features=hashing_dim, out_features=num_classes)
        
        # Freeze the resnet layers
        if freeze_features:
            for ff in self.features:
                for pp in ff.parameters():
                    pp.requires_grad = False
                    
                    
    def forward(self, x, y, features=False, softmax=False, norm=0):
        out_o = self.original_model.features(x,y)
        out_o = self.original_model.hashing(out_o)
        
        if norm != 0:
            out_o = norm * torch.nn.functional.normalize(out_o, p=1)

        out = self.linear(out_o)
        out_kd = self.original_model.logits(out_o)
        if softmax:
            out_kd = out_kd.softmax(-1)

#         x_norm = out_o.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
#         normed = out_o / x_norm
        
#         out = F.linear(normed, F.normalize(self.linear.weight), None) * x_norm
#         out_kd = F.linear(normed, F.normalize(self.original_model.last_linear.weight), None) * x_norm
        if features:
            return out, out_kd, out_o
            
        return out, out_kd

def Soft_Quantization(X, C, N_books, tau_q):
    L_word = int(C.size()[1]/N_books)
    x = torch.split(X, L_word, dim=1)
    c = torch.split(C, L_word, dim=1)
    for i in range(N_books):
        soft_c = F.softmax(squared_distances(x[i], c[i]) * (-tau_q), dim=-1)
        if i==0:
            Z = soft_c @ c[i]
        else:
            Z = torch.cat((Z, soft_c @ c[i]), dim=1)
    return Z

def squared_distances(x, y):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    diff = x.unsqueeze(1) - y.unsqueeze(0)
    return torch.sum(diff * diff, -1)

class Quantization_Head(nn.Module):
    def __init__(self, N_words, N_books, L_word, tau_q):
        super(Quantization_Head, self).__init__()
        self.fc = nn.Linear(512, N_books * L_word)
        nn.init.xavier_uniform_(self.fc.weight)

        # Codebooks
        self.C = torch.nn.Parameter(Variable((torch.randn(N_words, N_books * L_word)).type(torch.float32), requires_grad=True))
        nn.init.xavier_uniform_(self.C)

        self.N_books = N_books
        self.L_word = L_word
        self.tau_q = tau_q

    def forward(self, input):
        X = self.fc(input)
        Z = Soft_Quantization(X, self.C, self.N_books, self.tau_q)
        return X, Z

class CSEResnetModel_KDHashingBin(nn.Module):
    def __init__(self, arch, hashing_dim, num_classes, pretrained=True, freeze_features=False, ems=False):
        super(CSEResnetModel_KDHashingBin, self).__init__()
        
        self.hashing_dim = hashing_dim
        self.num_classes = num_classes
        self.modelName = arch
        self.q = Quantization_Head(8, 16, 64, 5)
        
        if pretrained:
            self.original_model = cse_resnet50_hashing(self.hashing_dim)
        else:
            self.original_model = cse_resnet50_hashing(self.hashing_dim, pretrained=None)
            
        
        self.ems = ems
        if self.ems:
            self.linear = EMSLayer(num_classes, hashing_dim)
        else:
            self.linear = nn.Linear(in_features=hashing_dim, out_features=num_classes)
            self.linear_bin = nn.Linear(in_features=64, out_features=num_classes)
        
        # Freeze the resnet layers
        if freeze_features:
            for ff in self.features:
                for pp in ff.parameters():
                    pp.requires_grad = False
                    
                    
    def forward(self, x, y, features=False, softmax=False, norm=0):
        out_o = self.original_model.features(x,y)
        out_o = self.original_model.hashing(out_o)
        
        if norm != 0:
            out_o = norm * torch.nn.functional.normalize(out_o, p=1)

        _, out_o_bin = self.q(out_o)
        out = self.linear(out_o)
        out_bin = self.linear_bin(out_o_bin)
        out_kd = self.original_model.logits(out_o)

        if softmax:
            out_kd = out_kd.softmax(-1)

#         x_norm = out_o.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
#         normed = out_o / x_norm
        
#         out = F.linear(normed, F.normalize(self.linear.weight), None) * x_norm
#         out_kd = F.linear(normed, F.normalize(self.original_model.last_linear.weight), None) * x_norm
        if features:
            return out, out_bin, out_kd, out_o, out_o_bin
            
        return out, out_bin, out_kd

class CSEResnetModel_FeatKDHashing(nn.Module):
    def __init__(self, arch, hashing_dim, num_classes, pretrained=True, freeze_features=False, ems=False):
        super(CSEResnetModel_FeatKDHashing, self).__init__()
        
        self.hashing_dim = hashing_dim
        self.num_classes = num_classes
        self.modelName = arch
        
        if pretrained:
            self.original_model = cse_resnet50_hashing(self.hashing_dim)
        else:
            self.original_model = cse_resnet50_hashing(self.hashing_dim, pretrained=None)
            
        
        self.ems = ems
        if self.ems:
            self.linear = EMSLayer(num_classes, hashing_dim)
        else:
            self.linear = nn.Linear(in_features=hashing_dim, out_features=num_classes)
        
        # Freeze the resnet layers
        if freeze_features:
            for ff in self.features:
                for pp in ff.parameters():
                    pp.requires_grad = False
                    
                    
    def forward(self, x, y):
        out_o = self.original_model.features(x,y)
        out_o = self.original_model.hashing(out_o)
        
        out = self.linear(out_o)
        # out_kd = self.original_model.logits(out_o)

#         x_norm = out_o.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
#         normed = out_o / x_norm
        
#         out = F.linear(normed, F.normalize(self.linear.weight), None) * x_norm
#         out_kd = F.linear(normed, F.normalize(self.original_model.last_linear.weight), None) * x_norm

        return out, out_o
    
class EMSLayer(nn.Module):
    def __init__(self, num_classes, num_dimension):
        super(EMSLayer, self).__init__()
        self.cpars = torch.nn.Parameter(torch.randn(num_classes, num_dimension))
        self.relu = torch.nn.ReLU(inplace=True)
        
    def forward(self, x):
        out = pairwise_distances(x, self.cpars)
        out = - self.relu(out).sqrt()
        return out
    
        
    
def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    return dist


# class HashingEncoder(nn.Module):
#     def __init__(self, input_dim, one_dim, two_dim, hash_dim):
#         super(HashingEncoder, self).__init__()
#         self.input_dim = input_dim
#         self.one_dim = one_dim
#         self.two_dim = two_dim
#         self.hash_dim = hash_dim
        
#         self.en1 = nn.Linear(input_dim, one_dim)
#         self.en2 = nn.Linear(one_dim, two_dim)
#         self.en3 = nn.Linear(two_dim, hash_dim)
        
#         self.de1 = nn.Linear(hash_dim, two_dim)
#         self.de2 = nn.Linear(two_dim, one_dim)
#         self.de3 = nn.Linear(one_dim, input_dim)
        
#         self.relu = nn.ReLU(inplace=True)
        
#     def forward(self, x):
#         e = self.en1(x)
#         e = self.relu(e)
#         e = self.en2(e)
#         e = self.relu(e)
#         e = self.en3(e)
        
#         # h = self.relu(torch.sign(e))
        
#         r = self.de1(e)
#         r = self.relu(r)
#         r = self.de2(r)
#         r = self.relu(r)
#         r = self.de3(r)
#         r = self.relu(r)
        
#         return e, r
    
    
class HashingEncoder(nn.Module):
    def __init__(self, input_dim, one_dim, hash_dim):
        super(HashingEncoder, self).__init__()
        self.input_dim = input_dim
        self.one_dim = one_dim
        self.hash_dim = hash_dim
        
        self.en1 = nn.Linear(input_dim, one_dim)
        self.en2 = nn.Linear(one_dim, hash_dim)
        self.de1 = nn.Linear(hash_dim, one_dim)
        self.de2 = nn.Linear(one_dim, input_dim)
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        e = self.en1(x)
        e = self.en2(self.relu(e))
        # e = self.en2(e)
        # e = self.relu(e)
        # e = self.en3(e)
        
        # h = self.relu(torch.sign(e))
        
        r = self.de1(e)
        r = self.de2(self.relu(r))
        # r = self.relu(r)
        # r = self.de2(r)
        # r = self.relu(r)
        # r = self.de3(r)
        r = self.relu(r)
        
        return e, r
    
    
class ScatterLoss(nn.Module):
    def __init__(self):
        super(ScatterLoss, self).__init__()
        
    def forward(self, e, y):
        sample_num = y.shape[0]
        e_norm = e/torch.sqrt(torch.sum(torch.mul(e,e),dim=1,keepdim=True))
        cnter = 0
        loss = 0
        for i1 in range(sample_num-1):
            e1 = e_norm[i1]
            y1 = y[i1]
            for i2 in range(i1+1, sample_num):
                e2 = e_norm[i2]
                y2 = y[i2]
                if y1 != y2:
                    cnter += 1
                    loss += torch.sum(torch.mul(e1, e2))
                    
        return loss/cnter
    
    
class QuantizationLoss(nn.Module):
    def __init__(self):
        super(QuantizationLoss, self).__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, e):
        return self.mse(e, torch.sign(e))
        
                
