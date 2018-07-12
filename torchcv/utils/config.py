from __future__ import print_function
from pprint import  pprint

class Config:
    lr = 1e-2
    checkpoint = 'checkpoint/'
    train_img_root = '/home/changq/Projects/VOCdevkit/VOC2012/JPEGImages/'
    train_img_list = 'torchcv/datasets/voc/voc12_trainval.txt'
    eval_img_root = '/home/changq/Projects/VOCdevkit/VOC2012/JPEGImages/'
    eval_img_list = 'torchcv/datasets/voc/voc12_trainval.txt'
    batch_size = 8 
    num_worker = 2
    plot_every = 20 # every iter
    save_state_every = 20 # every epoch
    load_path = 'checkpoint/dsod.pth' 
    img_size = 300
    env = 'dsod'
    iter_size = 5 # every iter
    eval_every= 10000 # every epoch

    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: ', k)
            setattr(self, k, v)

        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}


opt = Config()
