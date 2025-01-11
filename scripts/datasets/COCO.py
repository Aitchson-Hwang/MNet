from __future__ import print_function, absolute_import

import os
import csv
import numpy as np
import json
import random
import math
import matplotlib.pyplot as plt
from collections import namedtuple
from os import listdir
from os.path import isfile, join

import torch
import torch.utils.data as data

from scripts.utils.osutils import *
from scripts.utils.imutils import *
from scripts.utils.transforms import *
import torchvision.transforms as transforms
from PIL import Image
from PIL import ImageEnhance
from PIL import ImageFilter
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class COCO(data.Dataset):
    def __init__(self,train,config=None, sample=[],gan_norm=False):

        self.train = []
        self.anno = []
        self.mask = []
        self.wm = []
        self.input_size = config.input_size
        self.normalized_input = config.normalized_input
        self.base_folder = config.base_dir
        self.dataset = train+config.data

        if config == None:
            self.data_augumentation = False
        else:
            self.data_augumentation = config.data_augumentation

        self.istrain = False if self.dataset.find('train') == -1 else True
        self.sample = sample
        self.gan_norm = gan_norm
        mypath = join(self.base_folder,self.dataset)
        file_names = sorted([f for f in listdir(join(mypath,'image')) if isfile(join(mypath,'image', f)) ])

        if config.limited_dataset > 0:
            xtrain = sorted(list(set([ file_name.split('-')[0] for file_name in file_names ])))
            tmp = []
            for x in xtrain:
                # get the file_name by identifier
                tmp.append([y for y in file_names if x in y][0])

            file_names = tmp
        else:
            file_names = file_names

        for file_name in file_names:
            self.train.append(os.path.join(mypath,'image',file_name))

            # 以下为mask文件为png时的代码
            # file_name_png = file_name.replace(".jpg", ".png")
            # self.mask.append(os.path.join(mypath,'mask',file_name_png))

            # 以下为mask文件为jpg时的代码
            self.mask.append(os.path.join(mypath,'mask',file_name))

            # 测试需要输入水印
            self.wm.append(os.path.join(mypath,'wm',file_name))   # need wm

            # 以下为natural文件与测试样本名称不一致的情况
            self.anno.append(os.path.join(self.base_folder,'natural',file_name.split('-')[0]+'.jpg'))

            # 以下为natural文件与测试样本名称一致的情况
            # self.anno.append(os.path.join(self.base_folder,'natural',file_name))

        if len(self.sample) > 0 :
            self.train = [ self.train[i] for i in self.sample ] 
            self.mask = [ self.mask[i] for i in self.sample ] 
            self.anno = [ self.anno[i] for i in self.sample ] 

        self.trans = transforms.Compose([
                transforms.Resize((self.input_size,self.input_size)),
                transforms.ToTensor()
            ])

        print('total Dataset of '+self.dataset+' is : ', len(self.train))


    def __getitem__(self, index):
        img = Image.open(self.train[index]).convert('RGB')
        mask = Image.open(self.mask[index]).convert('L')
        anno = Image.open(self.anno[index]).convert('RGB')
        wm = Image.open(self.wm[index]).convert('RGB')    # need wm

        return {"image": self.trans(img),
                "target": self.trans(anno), 
                "mask": self.trans(mask), 
                "wm": self.trans(wm),   # need wm
                "name": self.train[index].split('/')[-1],
                "imgurl":self.train[index],
                "maskurl":self.mask[index],
                "targeturl":self.anno[index],
                "wmurl":self.wm[index]    # need wm
                }

    def __len__(self):

        return len(self.train)
