from __future__ import print_function

import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms

from tqdm import tqdm
from torch.autograd import Variable
from torchcv.transform import resize
from torchcv.datasets import ListDataset
from torchcv.evaluations.voc_eval import voc_eval
from torchcv.models import DSOD, SSDBoxCoder
from torchcv.utils.config import opt
from PIL import Image
import numpy as np

print('Loading model..')
net = DSOD(num_classes=21)
net.cuda()
net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
net.load_state_dict(torch.load(opt.load_path)['net'])
net.eval()

print('Preparing dataset..')
def caffe_normalize(x):
         return transforms.Compose([
            transforms.Lambda(lambda x:255*x[[2,1,0]]) ,
            transforms.Normalize([104,117,123], (1,1,1)), # make it the same as caffe
                  # bgr and 0-255
        ])(x)
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

#with open('torchcv/datasets/voc/voc07_test_difficult.txt') as f:
#    gt_difficults = []
#    for line in f.readlines():
#        line = line.strip().split()
#        d = np.array([int(x) for x in line[1:]])
#        gt_difficults.append(d)

print('Processing img..')
nums_img = dataset.__len__()
for i in tqdm(range(nums_img)):
    inputs, box_targets, label_targets = dataset.__getitem__(i)
    gt_boxes.append(box_targets)
    gt_labels.append(label_targets)

    inputs = inputs.unsqueeze(0)
    with torch.no_grad():
        loc_preds, cls_preds = net(Variable(inputs.cuda()))
    box_preds, label_preds, score_preds = box_coder.decode(
        loc_preds.cpu().data.squeeze(),
        F.softmax(cls_preds.squeeze(), dim=1).cpu().data,
        score_thresh=0.1)

    pred_boxes.append(box_preds)
    pred_labels.append(label_preds)
    pred_scores.append(score_preds)

print('Caculating AP..')
aps = voc_eval(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, gt_difficults = None, iou_thresh=0.5, use_07_metric=False)

print('ap = ', aps['ap'])
print('map = ', aps['map'])
