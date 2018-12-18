# %load train.py
#!/usr/bin/env python

# In[1]:
from t_cnn import tcnn

model = tcnn(num_classes = 2,pretrained = True, model_root = '/home/krf/model/BALL/')

import senet
import os
import numpy as np
import torch
from torchvision.datasets import ImageFolder
from utils import TransformImage
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tensorboardX import SummaryWriter

DATA_DIR = "/home/krf/dataset/BALL/"
traindir = DATA_DIR + "train"
valdir = DATA_DIR +"val"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
BATCH_SIZE = 32
WORKERS = 4
START = 0
EPOCHS = 1200
PRINT_FREQ = 20


# In[31]:


#model =  senet.se_resnext50_32x4d(num_classes = 2)
#通过随机变化来进行数据增强
train_tf  = TransformImage(
    model,
    
    random_crop=False,
    random_hflip=True,
    random_vflip=True,
    random_rotate=True,
    preserve_aspect_ratio=True
)
train_loader = torch.utils.data.DataLoader(
#     datasets.ImageFolder(traindir, transforms.Compose([
# #         transforms.RandomSizedCrop(max(model.input_size)),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         normalize,
#     ])),
    datasets.ImageFolder(traindir,train_tf),
    batch_size=BATCH_SIZE, shuffle=True,
    num_workers=WORKERS, pin_memory=True)


val_tf = TransformImage(
    model,
    
    preserve_aspect_ratio=True)
val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(valdir,val_tf),
    batch_size=BATCH_SIZE, shuffle=False,
    num_workers=WORKERS, pin_memory=True)


# In[29]:


def train(train_loader, model, criterion, optimizer, epoch,scheduler):
    # switch to train mode
    model.train()
#     end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
#         data_time.update(time.time() - end)
        input = input.cuda()
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input.float())
        target_var = torch.autograd.Variable(target)
        #print(input_var.type())
        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        #print(output.data)



#         # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

#         # measure elapsed time
#         batch_time.update(time.time() - end)
#         end = time.time()
        meters = trainMeter.update(output,target,loss,input.size(0))

        if i % PRINT_FREQ == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss:.5f}\t'
                  'Acc {Acc:.5f}\t'
                  'Precision {P:.5f}\t'
                  'Recall {R:.5f}\t'
                  'F1 {F1:.5f}'.format(
                   epoch,i, len(train_loader), loss=meters[4],
                   Acc=meters[3],P=meters[0],R=meters[1],F1=meters[2]))
            
            step = epoch*len(train_loader) + i
            
            writer.add_scalar('TRAIN/Precision', meters[0], step)
            writer.add_scalar('TRAIN/Recall', meters[1], step)
            writer.add_scalar('TRAIN/F1', meters[2], step)
            writer.add_scalar('TRAIN/Acc', meters[3], step)
            writer.add_scalar('TRAIN/loss',meters[4], step)
            
    scheduler.step(meters[4])


def validate(val_loader, model, criterion,epoch):
    # switch to evaluate mode
    model.eval()
    
#     end = time.time()
    meters = []
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        input = input.cuda()
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)
        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        meters = valMeter.update(output,target,loss,input.size(0))
        if i % PRINT_FREQ == 0:
            print('Test: [{0}/{1}]\t'
                  'Loss {loss:.5f}\t'
                  'Acc {Acc:.5f}\t'
                  'Precision {P:.5f}\t'
                  'Recall {R:.5f}\t'
                  'F1 {F1:.5f}'.format(
                   i, len(val_loader), loss=meters[4],
                   Acc=meters[3],P=meters[0],R=meters[1],F1=meters[2]))
            
            step = epoch * len(val_loader) + i
            writer.add_scalar('VAL/Precision', meters[0], step)
            writer.add_scalar('VAL/Recall', meters[1], step)
            writer.add_scalar('VAL/F1', meters[2], step)
            writer.add_scalar('VAL/Acc', meters[3], step)
            writer.add_scalar('VAL/loss',meters[4], step)
    print(' * Acc {Acc:.5f} F1 {F1:.5f}'
          .format(Acc=meters[3],F1=meters[2]))
    writer.add_scalar('VAL/EPOCH_F1', meters[2], epoch)
    return meters[2]


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'tcnn5_model_best.pth.tar')

class ModelMeter(object):
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.losses = AverageMeter()
        self.top1 = AverageMeter()
        self.TP = 1e-8
        self.TN = 1e-8
        self.FN = 1e-8
        self.FP = 1e-8
        self.P=1e-8
        self.R=1e-8
        self.F1=1e-8
        self.Acc=1e-8

    def update(self, output,target,loss, n=1):
        _, pred = output.data.topk(1, 1, True, True)
        pred = pred.t()
#         print(pred,target.data)
        # TP    predict 和 label 同时为1
        self.TP += ((pred == 1) & (target.data == 1)).cpu().numpy().sum()
        # TN    predict 和 label 同时为0
        self.TN += ((pred == 0) & (target.data == 0)).cpu().numpy().sum()
        # FN    predict 0 label 1
        self.FN += ((pred == 0) & (target.data == 1)).cpu().numpy().sum()
        # FP    predict 1 label 0
        self.FP += ((pred == 1) & (target.data == 0)).cpu().numpy().sum()
#         print(self.TP,self.TN,self.FN,self.FP)
#         zes=torch.autograd.Variable(torch.zeros(target.size()).type(torch.cuda.LongTensor))#全0变量

#         ons=torch.autograd.Variable(torch.ones(target.size()).type(torch.cuda.LongTensor))#全1变量
#         print(zes,ons)
#         train_correct01 = ((pred==zes)&(target.data.squeeze(1)==ons)).sum()#原标签为1，预测为 0 的总数

#         train_correct10 = ((pred==ons)&(target.data.squeeze(1)==zes)).sum()#原标签为0，预测为1  的总数

#         train_correct11 = ((pred_y==ons)&(target.data.squeeze(1)==ons)).sum()
#         train_correct00 = ((pred_y==zes)&(target.data.squeeze(1)==zes)).sum()
#         self.FN += train_correct01.data[0]        

#         self.FP += train_correct10.data[0]

#         self.TP += train_correct11.data[0]  
#         self.TN += train_correct00.data[0]
      
        self.P = self.TP / (self.TP + self.FP)
        self.R = self.TP / (self.TP + self.FN)
        self.F1 = 2 * self.R * self.P / (self.R + self.P)
        
        self.Acc = (self.TP + self.TN) / (self.TP + self.TN + self.FP + self.FN)
        
        self.losses.update(loss.data[0],n)

        return [self.P,self.R,self.F1,self.Acc,self.losses.avg]

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 1e-8
        self.avg = 1e-8
        self.sum = 1e-8
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(lr,optimizer, epoch):
    if epoch >= 300 and epoch % 100 == 0 :
        lr /= 10
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print("adjuct learning rate to {}".format(lr))
    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# In[ ]:

# 加载模型，解决命名和维度不匹配问题,解决多个gpu并行
def load_state_keywise(model, model_path):
    model_dict = model.state_dict()
    
    print("=> loading checkpoint '{}'".format(model_path))
    checkpoint = torch.load(model_path,map_location='cpu')
    START = checkpoint['epoch']
    best_F1 = checkpoint['best_prec1']
    #model.load_state_dict(checkpoint['state_dict'])
    
    pretrained_dict = checkpoint['state_dict']#torch.load(model_path, map_location='cpu')
    key = list(pretrained_dict.keys())[0]
    # 1. filter out unnecessary keys
    # 1.1 multi-GPU ->CPU
    if (str(key).startswith('module.')):
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if
                           k[7:] in model_dict and v.size() == model_dict[k[7:]].size()}
    else:
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                           k in model_dict and v.size() == model_dict[k].size()}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    print("start at epoch {}, best_f1={}".format(START,best_F1))
    return model,START,best_F1



criterion = nn.CrossEntropyLoss().cuda()
lr = 1e-4
optimizer = torch.optim.SGD(model.parameters(), lr = lr,momentum = 0.9,weight_decay=1e-6)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True)
best_f1 = 0
model,START,best_f1 = load_state_keywise(model,'tcnn5_checkpoint-Copy1.pth.tar')
# model = model.cuda()
model = torch.nn.DataParallel(model).cuda()

# TP = 0,TN = 0,FN = 0, FP = 0
writer = SummaryWriter()

trainMeter = ModelMeter()
valMeter = ModelMeter()
for epoch in range(START,EPOCHS):
    # train for one epoch
    lr = optimizer.param_groups[0]['lr']
    writer.add_scalar('TRAIN/LR', lr, epoch)
    train(train_loader, model, criterion, optimizer, epoch, scheduler)
    # evaluate on validation set
    F1 = validate(val_loader, model, criterion,epoch)
    
    # remember best prec@1 and save checkpoint
    is_best = F1 > best_f1
    best_f1 = max(F1, best_f1)
    save_checkpoint({
        'epoch': epoch + 1,
        'arch': "T-CNN",
        'state_dict': model.state_dict(),
        'best_prec1': best_f1,
    }, is_best,filename='tcnn5_checkpoint.pth.tar')
# export scalar data to JSON for external processing
writer.export_scalars_to_json("./test.json")
writer.close()





