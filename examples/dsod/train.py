from __future__ import print_function

import os
import random
import sys
import time

import matplotlib
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFile
from torch.autograd import Variable
from torchnet.meter import AverageValueMeter
from tqdm import tqdm

from torchcv.datasets import ListDataset
from torchcv.evaluations.voc_eval import voc_eval
from torchcv.loss import SSDLoss
from torchcv.models import DSOD
from torchcv.models import SSDBoxCoder
from torchcv.transform import (random_crop, random_distort, random_flip,
                                random_paste, resize)
from torchcv.utils.config import opt
from torchcv.visualizations import Visualizer

ImageFile.LOAD_TRUNCATED_IMAGES = True
#matplotlib.use('agg')






def caffe_normalize(x):
         return transforms.Compose([
            transforms.Lambda(lambda x:255*x[[2,1,0]]) ,
            transforms.Normalize([104,117,123], (1,1,1)), # make it the same as caffe
                  # bgr and 0-255
        ])(x)
def Transform(box_coder, train=True):
    def train_(img, boxes, labels):
        img = random_distort(img)
        if random.random() < 0.5:
            img, boxes = random_paste(img, boxes, max_ratio=4, fill=(123, 116, 103))
        img, boxes, labels = random_crop(img, boxes, labels)
        img, boxes = resize(img, boxes, size=(opt.img_size, opt.img_size), random_interpolation=True)
        img, boxes = random_flip(img, boxes)
        img = transforms.Compose([
            transforms.ToTensor(),
            caffe_normalize
        ])(img)
        boxes, labels = box_coder.encode(boxes, labels)
        return img, boxes, labels

    def test_(img, boxes, labels):
        img, boxes = resize(img, boxes, size=(opt.img_size, opt.img_size))
        img = transforms.Compose([
            transforms.ToTensor(),
            caffe_normalize
        ])(img)
        boxes, labels = box_coder.encode(boxes, labels)
        return img, boxes, labels

    return train_ if train else test_


def eval(net,test_num=10000):
    net.eval()

    def transform(img, boxes, labels):
        img, boxes = resize(img, boxes, size=(opt.img_size, opt.img_size))
        img = transforms.Compose([
            transforms.ToTensor(),
            caffe_normalize
        ])(img)
        return img, boxes, labels

    dataset = ListDataset(root=opt.eval_img_root, list_file=opt.eval_img_list, transform=transform)
    box_coder = SSDBoxCoder(net.module)

    pred_boxes = []
    pred_labels = []
    pred_scores = []
    gt_boxes = []
    gt_labels = []

#    with open('torchcv/datasets/voc/voc07_test_difficult.txt') as f:
#        gt_difficults = []
#        for line in f.readlines():
#            line = line.strip().split()
#            d = np.array([int(x) for x in line[1:]])
#            gt_difficults.append(d)

    nums_img = dataset.__len__()
    for i in tqdm(range(nums_img)):
        inputs, box_targets, label_targets = dataset.__getitem__(i)
        gt_boxes.append(box_targets)
        gt_labels.append(label_targets)

        inputs = inputs.unsqueeze(0)
        with torch.no_grad() :
            loc_preds, cls_preds = net(Variable(inputs.cuda()))
        box_preds, label_preds, score_preds = box_coder.decode(
            loc_preds.cpu().data.squeeze(),
            F.softmax(cls_preds.squeeze(), dim=1).cpu().data,
            score_thresh=0.1)

        pred_boxes.append(box_preds)
        pred_labels.append(label_preds)
        pred_scores.append(score_preds)

    aps = (voc_eval(
        pred_boxes, pred_labels, pred_scores,
        gt_boxes, gt_labels, gt_difficults=None,
        iou_thresh=0.5, use_07_metric=False))
    net.train()
    return aps


def predict(net, box_coder, img):
    net.eval()
    if isinstance(img, str):
        img = Image.open(img)
        ow = oh = 300
        img = img.resize((ow, oh))
    transform = transforms.Compose([
        transforms.ToTensor(),
        caffe_normalize
    ])
    x = transform(img).cuda()
    x = Variable(x)
    loc_preds, cls_preds = net(x.unsqueeze(0))
    try:
        boxes, labels, scores = box_coder.decode(
            loc_preds.data.cpu().squeeze(), F.softmax(cls_preds.squeeze().cpu(), dim=1).data)
    except:print('except in predict')
    draw = ImageDraw.Draw(img)
    for box in boxes:
        draw.rectangle(list(box), outline='red')
    net.train()
    return img


def main(**kwargs):
    opt._parse(kwargs)

    vis = Visualizer(env=opt.env)

    # Model
    print('==> Building model..')
    net = DSOD(num_classes=21)

    # Dataset
    print('==> Preparing dataset..')
    box_coder = SSDBoxCoder(net)

    trainset = ListDataset(root=opt.train_img_root,
                           list_file=opt.train_img_list,
                           transform=Transform(box_coder, True))

    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=opt.batch_size,
                                              shuffle=True,
                                              num_workers=opt.num_worker,
                                              pin_memory=True)

    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

    criterion = SSDLoss(num_classes=21)
    optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=1e-4)

    best_map_ = 0
    best_loss = 1e100
    start_epoch = 0

    if opt.load_path is not None:
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(opt.load_path)
        net.load_state_dict(checkpoint['net'])
        best_loss = checkpoint['map']
        start_epoch = checkpoint['epoch'] + 1
        print('start_epoch = ', start_epoch, 'best_loss = ', best_loss)

    for epoch in range(start_epoch, start_epoch + 100):
        print('\nEpoch: ', epoch)
        net.train()
        train_loss = 0
        optimizer.zero_grad()
        ix = 0
        for batch_idx, (inputs, loc_targets, cls_targets) in tqdm(enumerate(trainloader)):
            inputs = Variable(inputs.cuda())
            loc_targets = Variable(loc_targets.cuda())
            cls_targets = Variable(cls_targets.cuda())

            loc_preds, cls_preds = net(inputs)
            ix+=1
            loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
            loss.backward()
            train_loss += loss.data.item()
            current_loss = train_loss/(1+batch_idx)

            if (batch_idx+1) % (opt.iter_size) == 0:
                for name,p in net.named_parameters():p.grad.data.div_(ix)
                ix = 0
                optimizer.step()
                optimizer.zero_grad()

            if (batch_idx + 1) % opt.plot_every == 0:
                vis.plot('loss', current_loss)

#                img = predict(net, box_coder, os.path.join(opt.train_img_root, trainset.fnames[batch_idx]))
#                vis.img('predict', np.array(img).transpose(2, 0, 1))

#                if os.path.exists(opt.debug_file):
#                    import ipdb
#                    ipdb.set_trace()


        print('current_loss: ', current_loss, 'best_loss: ', best_loss)

        if (epoch+1) % 20 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1

        if (epoch+1) % opt.save_state_every == 0 :
            state = {
                    'net': net.state_dict(),
                    'map': current_loss,
                    'epoch': epoch,
            }
            torch.save(state, opt.checkpoint + '%s.pth' % epoch)

        if current_loss< best_loss:
            best_loss = current_loss
            print('saving model at epoch: ', epoch)
            state = {
                'net': net.state_dict(),
                'map': best_loss,
                'epoch': epoch,
            }
            torch.save(state, opt.checkpoint + 'dsod.pth')
        
"""
        if (epoch+1)%opt.eval_every ==0:
            aps = eval(net)
            map_ = aps['map']
            if map_ > best_map_:
                print('Saving..')
                state = {
                    'net': net.module.state_dict(),
                    'map': best_map_,
                    'epoch': epoch,
                }
                best_map_ = map_
                if not os.path.isdir(os.path.dirname(opt.checkpoint)):
                    os.mkdir(os.path.dirname(opt.checkpoint))
                best_path = opt.checkpoint + '/%s.pth' % best_map_
                torch.save(state, best_path)
            else:
                net.module.load_state_dict(torch.load(best_path)['net'])
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.1
            vis.log(dict(epoch=(epoch+1),map=map_,loss=train_loss / (batch_idx + 1)))
"""

def test_eval():
    net = DSOD(num_classes = 21)
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    net.load_state_dict(torch.load(opt.load_path)['net'])
    aps = eval(net)
    print(aps['ap'])
    print(aps['map'])
            
if __name__ == '__main__':
    import fire

    fire.Fire()
