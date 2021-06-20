# coding=utf-8
from dataset import Garbage_Loader
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import torch
import time
import os
import shutil

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

"""
    Author : Jack Cui
    Wechat : https://mp.weixin.qq.com/s/OCWwRVDFNslIuKyiCVUoTA
"""

from tensorboardX import SummaryWriter


def accuracy(output, target, topk=(1,)):
    """
        计算topk的准确率
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)#得到一个batch_size*k大小的矩阵，每一列为该图片的预测值
        pred = pred.t()#得到一个k*batch_size大小的矩阵
        correct = pred.eq(target.contiguous().view(1, -1).expand_as(pred))#将1*batch_size大小的矩阵扩充到k*batch_size,每一列都相同

        class_to = pred[0].cpu().numpy()

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)#计算前k个中1的总个数
            res.append(correct_k.mul_(100.0 / batch_size))#除以batchsize()作为正确率
        return res, class_to


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
        根据 is_best 存模型，一般保存 valid acc 最好的模型
    """
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best_' + filename)


def train(train_loader, model, criterion, optimizer, epoch, writer):
    """
        训练代码
        参数：
            train_loader - 训练集的 DataLoader
            model - 模型
            criterion - 损失函数
            optimizer - 优化器
            epoch - 进行第几个 epoch
            writer - 用于写 tensorboardX
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # compute output
        output = model(input)
        loss = criterion(output, target)
        # measure accuracy and record loss
        [prec1, prec5], class_to = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
    writer.add_scalar('loss/train_loss', losses.val, global_step=epoch)
    # writer.add_scalar('Accuracy/train_accuracy', losses.val, global_step=epoch)



def validate(val_loader, model, criterion, epoch, writer, phase="VAL"):
    """
        验证代码
        参数：
            val_loader - 验证集的 DataLoader
            model - 模型
            criterion - 损失函数
            epoch - 进行第几个 epoch
            writer - 用于写 tensorboardX
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()#否则的话，有输入数据，即使不训练，它也会改变权值。这是model中含有batch normalization层所带来的的性质。

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            # input = input.cuda()
            # target = target.cuda()
            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            [prec1, prec5], class_to = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                print('Test-{0}: [{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    phase, i, len(val_loader),
                    batch_time=batch_time,
                    loss=losses,
                    top1=top1, top5=top5))

        print(' * {} Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(phase, top1=top1, top5=top5))
    writer.add_scalar('loss/valid_loss', losses.val, global_step=epoch)
    return top1.avg, top5.avg


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


if __name__ == "__main__":
    # -------------------------------------------- step 1/4 : 加载数据 ---------------------------
    train_dir_list = 'E:\\ruanjian\\demo\\pachong\\deepleaning\\垃圾图片库\\train.txt'
    valid_dir_list = 'E:\\ruanjian\\demo\\pachong\\deepleaning\\垃圾图片库\\val.txt'
    batch_size = 64
    epochs = 80
    num_classes = 214
    train_data = Garbage_Loader(train_dir_list, train_flag=True)
    valid_data = Garbage_Loader(valid_dir_list, train_flag=False)
    train_loader = DataLoader(dataset=train_data, num_workers=8, pin_memory=True, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, num_workers=8, pin_memory=True, batch_size=batch_size)
    train_data_size = len(train_data)
    print('训练集数量：%d' % train_data_size)
    valid_data_size = len(valid_data)
    print('验证集数量：%d' % valid_data_size)
    # ------------------------------------ step 2/4 : 定义网络 ------------------------------------
    # 迁移学习
    model = models.resnet50(pretrained=True)
    # 提取fc层中固定的参数
    fc_inputs = model.fc.in_features
    # 修改类别为214
    model.fc = nn.Linear(fc_inputs, num_classes)
    # model = model.cuda()
    # ------------------------------------ step 3/4 : 定义损失函数和优化器等 -------------------------
    lr_init = 0.0001  # 定义初始学习率
    lr_stepsize = 20  # 定义每隔多少个epoch动态地更新学习率
    weight_decay = 0.001  # 定义权重衰减参数
    criterion = nn.CrossEntropyLoss()  # 定义交叉熵损失函数
    # 定义Adam优化器
    optimizer = optim.Adam(model.parameters(), lr=lr_init, weight_decay=weight_decay)
    # 动态地调整学习率
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_stepsize, gamma=0.1)
    writer = SummaryWriter('runs/resnet50')
    #  定义一个写入summary的目标文件，dir为写入文件地址  ......(交叉熵、优化器等定义)
    # ------------------------------------ step 4/4 : 训练 -----------------------------------------
    best_prec1 = 0
    for epoch in range(epochs):
        scheduler.step()
        # 开始训练
        train(train_loader, model, criterion, optimizer, epoch, writer)
        # 在验证集上测试效果
        valid_prec1, valid_prec5 = validate(valid_loader, model, criterion, epoch, writer, phase="VAL")
        # 比较每次的准确率，选择最大的那个
        is_best = valid_prec1 > best_prec1
        best_prec1 = max(valid_prec1, best_prec1)
        # 保存最好的模型
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': 'resnet50',
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best,
            filename='checkpoint_resnet50.pth.tar')
    writer.close()