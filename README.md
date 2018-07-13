# DSOD-Pytorch
This is an implementation of DSOD in Pytorch. It is based on the code [dsod.pytorch](https://github.com/chenyuntc/dsod.pytorch) and [torchcv](https://github.com/kuangliu/torchcv)  

Origin implementation is [here](https://github.com/szq0214/DSOD) and [here](openaccess.thecvf.com/content_ICCV_2017/papers/Shen_DSOD_Learning_Deeply_ICCV_2017_paper.pdf) is the paper

## Requirment
python 2.7  
pytorch 0.4  
visdom  

## Train
1. Download this repo  
```
git clone git@github.com:qqadssp/DSOD-Pytorch  
cd DSOD-Pytorch  
```
2. Download Pascal VOC dataset and unzip it, its path should be {root_dir}/VOCdevkit  

3. Modify opt.train_img_root in torchcv/utils/config.py with proper img_path  

4. Start visdom server and begin to train  
```
python -m visdom.server  
python train.py main  
```
## Eval
1. After training some epochs, checkpoint will be saved with name 'dsod.pth' or '##.pth' like '39.pth'. Modify opt.load_path in config.py with 'checkpoint/dsod.pth' or 'checkpoint/39.pth'  

2. Download Pascal VOC testset, modify opt.eval_img_root in config.py with proper path  

3. Evaluate the model  
```
python eval.py
```
## Demo
Training dataset is used here. If you have trained some epochs and get checkpoint, modify opt.load_path and run  
```
python demo.py
```
