import argparse
import os
import shutil
import time
import numpy as np
from models.ResnetModel import CSEResnetModel_KD, CSEResnetModel_KDHashing
from dataloaders.Sketchy import SketchyDataset
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.autograd import Variable
import math
import datetime
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pretrainedmodels
from models.senet import cse_resnet50
import torch.nn.functional as F
from utils.tools import set_seed, accuracy, SoftCrossEntropy, AverageMeter
import wandb

wandb.init(project="ZSSBIR")

model_names = sorted(name for name in pretrainedmodels.__dict__
    if name.islower() and not name.startswith("__"))


parser = argparse.ArgumentParser(description='PyTorch CSE_ResNet Model for Sketchy Training')

parser.add_argument('--savedir', '-s',  metavar='DIR',
                    default='../checkpoints/cse_resnet50_scratch/sketchy/',
                    help='path to save dir')
parser.add_argument('--arch', '-a', metavar='ARCH', default='cse_resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: cse_resnet50)')
parser.add_argument('--num_classes', metavar='N', type=int, default=100,
                    help='number of classes (default: 100)')
parser.add_argument('--num_hashing', metavar='N', type=int, default=512,
                    help='number of hashing dimension (default: 512)')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch_size', default=80, type=int, metavar='N',
                    help='number of samples per batch')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate (default: 0.01)')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-f', '--freeze_features', dest='freeze_features', action='store_true',
                    help='freeze features of the base network')
parser.add_argument('--kd_lambda', metavar='LAMBDA', default='1.0', type=float,
                    help='lambda for kd loss (default: 1)')
parser.add_argument('--sake_lambda', metavar='LAMBDA', default='1.0', type=float,
                    help='lambda for total SAKE loss (default: 1)')
parser.add_argument('--kld_lambda', metavar='LAMBDA', default='0.1', type=float)
parser.add_argument('--kld_step', metavar='LAMBDA', default='0', type=float)
parser.add_argument('--seed', default=-1, type=int)
parser.add_argument('--zero_version', metavar='VERSION', default='zeroshot1', type=str,
                    help='zeroshot version for training and testing (default: zeroshot1)')

def main():
    global args
    args = parser.parse_args()
    if args.seed > 0:
        set_seed(args.seed)
    if args.zero_version == 'zeroshot2':
        args.num_classes = 104
    
    wandb.config.update(args)
    # create model

    model = CSEResnetModel_KDHashing(args.arch, args.num_hashing, args.num_classes, \
                                     pretrained = True, freeze_features = args.freeze_features)
    wandb.config.update({"model": type(model).__name__,})
    # model.cuda()
    model = nn.DataParallel(model).cuda()
    print(str(datetime.datetime.now()) + ' student model inited.')
    model_t = cse_resnet50()
    model_t = nn.DataParallel(model_t).cuda()
    print(str(datetime.datetime.now()) + ' teacher model inited.')
    
    # define loss function (criterion) and optimizer
    criterion_train = nn.CrossEntropyLoss().cuda()
    criterion_train_kd = SoftCrossEntropy().cuda()
    criterion_test = nn.CrossEntropyLoss().cuda()

    wandb.config.update({"criterion": type(criterion_train).__name__,})
    wandb.config.update({"criterion_kd": type(criterion_train_kd).__name__,})
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # load data
    immean = [0.485, 0.456, 0.406] # RGB channel mean for imagenet
    imstd = [0.229, 0.224, 0.225]
    
    transformations = transforms.Compose([transforms.ToPILImage(),
                                          transforms.Resize([224,224]),
                                          transforms.ToTensor(),
                                          transforms.Normalize(immean, imstd)])
    
    sketchy_train = SketchyDataset(split='train', zero_version=args.zero_version, \
                                    transform=transformations, aug=True, cid_mask = True)
    train_loader = DataLoader(dataset=sketchy_train, batch_size=args.batch_size//3, shuffle=True, num_workers=2)
    
    sketchy_train_ext = SketchyDataset(split='train', version='all_photo', zero_version=args.zero_version, \
                                         transform=transformations, aug=True, cid_mask = True)
    
    train_loader_ext = DataLoader(dataset=sketchy_train_ext, \
                                  batch_size=args.batch_size//3*2, shuffle=True, num_workers=2)
    
    
    sketchy_val = SketchyDataset(split='val', zero_version=args.zero_version, transform=transformations, aug=False)
    val_loader = DataLoader(dataset=sketchy_val, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print(str(datetime.datetime.now()) + ' data loaded.')
    
    
    if args.evaluate:
        acc1 = validate(val_loader, model, criterion_test, criterion_train_kd, model_t)
        return
    
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
        
    best_acc1 = 0
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch)
        
        train(train_loader, train_loader_ext, model, criterion_train, criterion_train_kd, \
              optimizer, epoch, model_t)
        acc1 = validate(val_loader, model, criterion_test, criterion_train_kd, model_t)
        
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, filename = os.path.join(args.savedir,'checkpoint.pth.tar'))
        
    
    
def train(train_loader, train_loader_ext, model, criterion, criterion_kd, \
          optimizer, epoch, model_t):
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    losses_kd = AverageMeter()
    losses_kld = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    # switch to train mode
    model.train()
    model_t.eval()
    end = time.time()
    for i, ((input, target, cid_mask),(input_ext, target_ext, cid_mask_ext)) in enumerate(zip(train_loader, train_loader_ext)):
        input_all = torch.cat([input, input_ext],dim=0)
        tag_zeros = torch.zeros(input.size()[0],1)
        tag_ones = torch.ones(input_ext.size()[0],1)
        tag_all = torch.cat([tag_zeros, tag_ones], dim=0)
        
        target_all =  torch.cat([target, target_ext],dim=0)
        cid_mask_all = torch.cat([cid_mask, cid_mask_ext], dim=0)
        
        shuffle_idx = np.arange(input_all.size()[0])
        np.random.shuffle(shuffle_idx)
        input_all = input_all[shuffle_idx]
        tag_all = tag_all[shuffle_idx]
        target_all = target_all[shuffle_idx]
        cid_mask_all = cid_mask_all[shuffle_idx]
        
        input_all = input_all.cuda()
        tag_all = tag_all.cuda()
        target_all = target_all.type(torch.LongTensor).view(-1,).cuda()
        cid_mask_all = cid_mask_all.float().cuda()
        
        output,output_kd,output_feat = model(input_all,tag_all,features=True)
        with torch.no_grad():
            output_t = model_t(input_all,tag_all)
            
        loss = criterion(output, target_all)
        loss_kd = criterion_kd(output_kd, output_t * args.kd_lambda, tag_all)
        loss_kld = torch.Tensor([0]).cuda()
        kld_lambda = 0
        if args.kld_lambda > 0:
            zn = Variable(torch.cuda.FloatTensor(np.random.normal(0,1, (len(input_all), output_feat.size(-1)))))
            loss_kld = F.kl_div(F.log_softmax(output_feat, dim=-1), F.softmax(zn, dim=-1), reduction='batchmean')
            kld_lambda = args.kld_lambda
            if args.kld_step > 0:
                kld_lambda =  min(args.kld_lambda, wandb.run.step * args.kld_step)
        
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target_all, topk=(1, 5))
        losses.update(loss.item(), input_all.size(0))
        losses_kd.update(loss_kd.item(), input_ext.size(0))
        losses_kld.update(loss_kld.item(), input_ext.size(0))
        top1.update(acc1[0], input_all.size(0))
        top5.update(acc5[0], input_all.size(0))

        # online log
        train_log = {
            "train/loss": losses.avg,
            "train/losses_kd": losses_kd.avg,
            "train/losses_kld": losses_kld.avg,
            "train/top1_acc": top1.avg,
            "train/top5_acc": top5.avg,
            "train/kld_lambda": kld_lambda
        }

        wandb.log(train_log)
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss_total = loss+args.sake_lambda*loss_kd+kld_lambda*loss_kld
        loss_total.backward()
        optimizer.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0 or i == len(train_loader)-1:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t'
                  'Loss {loss.val:.3f} {loss_kd.val:.3f} ({loss.avg:.3f} {loss_kd.avg:.3f})\t'
                  'Acc@1 {top1.val:.2f} ({top1.avg:.2f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   loss=losses, loss_kd=losses_kd, top1=top1))
        
        
    
def validate(val_loader, model, criterion, criterion_kd, model_t):
    batch_time = AverageMeter()
    losses = AverageMeter()
    losses_kd = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    model_t.eval()
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input = torch.autograd.Variable(input, requires_grad=False).cuda()
        target = target.type(torch.LongTensor).view(-1,)
        target = torch.autograd.Variable(target).cuda()

        # compute output
        with torch.no_grad():
            output_t = model_t(input, torch.zeros(input.size()[0],1).cuda())
            output,output_kd = model(input, torch.zeros(input.size()[0],1).cuda())
        
        loss = criterion(output, target)
        loss_kd = criterion_kd(output_kd, output_t)
        
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        losses_kd.update(loss_kd.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 or i == len(val_loader)-1:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t'
                  'Loss {loss.val:.3f} {loss_kd.val:.3f} ({loss.avg:.3f} {loss_kd.avg:.3f})\t'
                  'Acc@1 {top1.val:.2f} ({top1.avg:.2f})\t'
                  'Acc@5 {top5.val:.2f} ({top5.avg:.2f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses, loss_kd=losses_kd, 
                   top1=top1, top5=top5))

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    
    # online log
    val_log = {
        "val/top1_acc": top1.avg,
        "val/top5_acc": top5.avg
    }

    wandb.log(val_log)
            
    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        filepath = '/'.join(filename.split('/')[0:-1])
        shutil.copyfile(filename, os.path.join(filepath,'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch):
    # lr = args.lr * 0.5 * (1.0 + math.cos(float(epoch) / args.epochs * math.pi))
    # epoch_curr = min(epoch, 20)
    # lr = args.lr * math.pow(0.001, float(epoch_curr)/ 20 )
    lr = args.lr * math.pow(0.001, float(epoch) / args.epochs)
    print('epoch: {}, lr: {}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    main()