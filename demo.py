from __future__ import print_function

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from PIL import Image, ImageDraw
from torch.autograd import Variable
from torchcv.models import DSOD, SSDBoxCoder
from torchcv.datasets import ListDataset
from torchcv.transform import resize
from torchcv.utils.config import opt
import random

print('Loading model..')
net = DSOD(num_classes=21)
net.cuda()
net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
net.load_state_dict(torch.load(opt.load_path)['net'])
net.eval()

print('Loading image..')
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

nums_img = dataset.__len__()
idx = random.randint(0, nums_img)
inputs, _, _ = dataset.__getitem__(idx)
inputs = inputs.unsqueeze(0)
with torch.no_grad():
    loc_preds, cls_preds = net(Variable(inputs.cuda()))
boxes, labels, scores = box_coder.decode(
    loc_preds.cpu().data.squeeze(), F.softmax(cls_preds.squeeze(), dim=1).cpu().data, score_thresh=0.5)

img = Image.open(opt.eval_img_root + dataset.fnames[idx])
sw = float(img.size[0])/float(opt.img_size)
sh = float(img.size[1])/float(opt.img_size)
boxes = boxes.type(torch.FloatTensor) * torch.tensor([sw, sh, sw, sh])
draw = ImageDraw.Draw(img)
nums_boxes = boxes.size()[0]
for i in range(nums_boxes):
    draw.rectangle(list(boxes[i]), outline='red')
    draw.text((boxes[i][0], boxes[i][1]), 'category: %s' % labels[i].item(), 'yellow')
    draw.text((boxes[i][0], boxes[i][1]+10), 'score: %s' % scores[i].item(), 'yellow')
img.show()
