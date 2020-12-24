import argparse
import template
import torch
import serverUtility
import data
import model
import loss
from trainer import Trainer

parser = argparse.ArgumentParser(description='EDSR and MDSR')

parser.add_argument('--alpha', type=float, default= -1 ,
                    help='controller')

parser.add_argument('--debug', action='store_true',
                    help='Enables debug mode')
parser.add_argument('--template', default='.',
                    help='You can set various templates in option.py')

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=6,
                    help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')

# Data specifications
parser.add_argument('--dir_data', type=str, default='/data2/',
                    help='dataset directory')
parser.add_argument('--dir_demo', type=str, default='../test',
                    help='demo image directory')
parser.add_argument('--data_train', type=str, default='DIV2K',
                    help='train dataset name')
parser.add_argument('--data_test', type=str, default='DIV2K',
                    help='test dataset name')
parser.add_argument('--data_range', type=str, default='1-800/801-810',
                    help='train/test data range')
parser.add_argument('--ext', type=str, default='sep',
                    help='dataset file extension')
parser.add_argument('--scale', type=str, default='4',
                    help='super resolution scale')
parser.add_argument('--patch_size', type=int, default=192,
                    help='output patch size')
parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--chop', action='store_true',
                    help='enable memory-efficient forward')
parser.add_argument('--no_augment', action='store_true',
                    help='do not use data augmentation')

# Model specifications
parser.add_argument('--model', default='EDSR',
                    help='model name')

parser.add_argument('--act', type=str, default='relu',
                    help='activation function')
parser.add_argument('--pre_train', type=str, default='',
                    help='pre-trained model directory')
parser.add_argument('--extend', type=str, default='.',
                    help='pre-trained model directory')
parser.add_argument('--n_resblocks', type=int, default=16,
                    help='number of residual blocks')
parser.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')
parser.add_argument('--res_scale', type=float, default=1,
                    help='residual scaling')
parser.add_argument('--shift_mean', default=True,
                    help='subtract pixel mean from the input')
parser.add_argument('--dilation', action='store_true',
                    help='use dilated convolution')
parser.add_argument('--precision', type=str, default='single',
                    choices=('single', 'half'),
                    help='FP precision for test (single | half)')

# Option for Residual dense network (RDN)
parser.add_argument('--G0', type=int, default=64,
                    help='default number of filters. (Use in RDN)')
parser.add_argument('--RDNkSize', type=int, default=3,
                    help='default kernel size. (Use in RDN)')
parser.add_argument('--RDNconfig', type=str, default='B',
                    help='parameters config of RDN. (Use in RDN)')

# Option for Residual channel attention network (RCAN)
parser.add_argument('--n_resgroups', type=int, default=10,
                    help='number of residual groups')
parser.add_argument('--reduction', type=int, default=16,
                    help='number of feature maps reduction')

# Training specifications
parser.add_argument('--reset', action='store_true',
                    help='reset the training')
parser.add_argument('--test_every', type=int, default=1000,
                    help='do test per every N batches')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=1,
                    help='input batch size for training')
parser.add_argument('--split_batch', type=int, default=1,
                    help='split the batch into smaller chunks')
parser.add_argument('--self_ensemble', action='store_true',
                    help='use self-ensemble method for test')
parser.add_argument('--test_only', action='store_true',
                    help='set this option to test the model')
parser.add_argument('--gan_k', type=int, default=1,
                    help='k value for adversarial loss')

# Optimization specifications
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--decay', type=str, default='200',
                    help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999),
                    help='ADAM beta')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')
parser.add_argument('--gclip', type=float, default=0,
                    help='gradient clipping threshold (0 = no clipping)')

# Loss specifications
parser.add_argument('--loss', type=str, default='1*L1',
                    help='loss function configuration')
parser.add_argument('--skip_threshold', type=float, default='1e8',
                    help='skipping batch that has large error')

# Log specifications
parser.add_argument('--save', type=str, default='test',
                    help='file name to save')
parser.add_argument('--load', type=str, default='',
                    help='file name to load')
parser.add_argument('--resume', type=int, default=0,
                    help='resume from specific checkpoint')
parser.add_argument('--save_models', action='store_true',
                    help='save all intermediate models')
parser.add_argument('--print_every', type=int, default=100,
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_results', action='store_true',
                    help='save output results')
parser.add_argument('--save_gt', action='store_true',
                    help='save low-resolution and high-resolution images together')

args = parser.parse_args()

args.data_test = ["Demo"]
args.scale = [4]
args.pre_train = "../experiment/edsr_baseline_x4/model/model_best.pt"
args.test_only = True
args.save_results = True

torch.manual_seed(args.seed)
checkpoint = serverUtility.checkpoint(args)

if args.alpha != -1:
   print('alpha: ',args.alpha)
global model
if args.data_test == ['video']:
   from videotester import VideoTester
   model = model.Model(args, checkpoint)
   t = VideoTester(args, model, checkpoint)
   t.test()
else:
   if checkpoint.ok:
       
       _loss = loss.Loss(args, checkpoint) if not args.test_only else None
       _model = model.Model(args, checkpoint)
       

#######################################################################################

#-*- coding:utf-8 -*-
"""
__author__ = BlingBling
建立TCP的基本流程
ss = socket() # 创建服务器套接字
ss.bind() # 套接字与地址绑定
ss.listen() # 监听连接
inf_loop: # 服务器无限循环
    cs = ss.accept() # 接受客户端连接
    comm_loop: # 通信循环
        cs.recv()/cs.send() # 对话（接收/发送）
    cs.close() # 关闭客户端套接字
ss.close() # 关闭服务器套接字#（可选）
"""
#!/usr/bin/env python

import os
from socket import *
from time import ctime

HOST = ''  #对bind（）方法的标识，表示可以使用任何可用的地址
PORT = 21567  #设置端口
BUFSIZ = 1024  #设置缓存区的大小
ADDR = (HOST, PORT)

tcpSerSock = socket(AF_INET, SOCK_STREAM)  #定义了一个套接字
tcpSerSock.bind(ADDR)  #绑定地址
tcpSerSock.listen(10)     #规定传入连接请求的最大数，异步的时候适用

while True:
    print('waiting for connection...')
    tcpCliSock, addr = tcpSerSock.accept()
    print ('...connected from:', addr)
    while True:
        rec = tcpCliSock.recv(BUFSIZ)
        print("recv:",rec.decode("utf-8"))
        if not rec:
            break

        args.alpha = rec.decode("utf-8")
        print(args.alpha)
        loader = data.Data(args)
        t = Trainer(args, loader, _model, _loss, checkpoint)
        t.test()

        filename = '../experiment/test/results-Demo/'+ 'result.png' 
        if os.path.exists(filename):
            filesize = str(os.path.getsize(filename))
            print("文件大小为：",filesize)
            tcpCliSock.send(filesize.encode())
            rec = tcpCliSock.recv(BUFSIZ)   #挂起服务器发送，确保客户端单独收到文件大小数据，避免粘包
            print("开始发送")
            f = open(filename, "rb")
            for line in f:
                tcpCliSock.send(line)
        else:
            tcpCliSock.send("0001".encode())   #如果文件不存在，那么就返回该代码
    tcpCliSock.close()
tcpSerSock.close()

########################################################################################
checkpoint.done()
