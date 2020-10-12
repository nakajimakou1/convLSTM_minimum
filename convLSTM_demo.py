#!/usr/bin/env python
import six
import cupy
# 2020.10.09 add
import numpy as np

import chainer
from chainer import computational_graph, Chain, Variable, utils, gradient_check, Function
from chainer import cuda
import chainer.links as L
import chainer.functions as F
from chainer import optimizers
from chainer import serializers
from pathlib import Path
import os ,cv2
from PIL import Image
from mylib import util
import torch

#read Kitti datas
str_dir1 = '../data/coco/' # '65x65data/'
# str_png = '.png'
str_png = '.jpg'
str_03 = 'images/train2014/COCO_train2014_' #'500'
str_02 = 'images/train2014/COCO_train2014_' #'50'
str_01 = 'images/train2014/COCO_train2014_' #'5'

epochNum   = 10 # the number of epoch
len_1data  = 25
channelIn  = 1  #channel number of input image
channelOut = 1  #channel number of output image
width  = 65     #65 #width of input image
height = 65     #65 #height of imput image
ksize  = 5
padsize = (ksize - 1) / 2

imgArrayTrain = cupy.zeros((126, height, width), dtype=cupy.float32)
imgArrayTest = cupy.zeros((26, height, width), dtype=cupy.float32)
# imgArrayTrain = cupy.zeros((126, height, width), dtype=cupy.int32)
# imgArrayTest = cupy.zeros((26, height, width), dtype=cupy.int32)

#load to train array
# for i in range(126):
j = 0
for i in range(10000):
    str_sum = str_dir1
    if i < 10:
        str_sum = str_dir1 + str_03 + str('{0:012d}'.format(i)) + str_png

    elif i < 100:
        str_sum = str_dir1 + str_02 + str('{0:012d}'.format(i)) + str_png

    else:
        str_sum = str_dir1 + str_01 + str('{0:012d}'.format(i)) + str_png
   
    # print("[*] str_sum:", str_sum)
    if(os.path.exists(str_sum) == True): 
        # img_read = util.imread(str_sum)       # Image.open(str_sum)
        # img_read = cv2.resize(img_read,(65, 65))  #(64, 64))
        img_read = Image.open(str_sum).convert('L')
        img_read = img_read.resize((65, 65))  #(64, 64))
        # imgArrayPart = cupy.asarray(np.reshape(img_read, (-1, 65, 65)).astype(dtype=cupy.float32))
        imgArrayPart = cupy.asarray(img_read).astype(dtype=cupy.float32)
        # imgArrayPart = cupy.asarray(img_read).astype(dtype=cupy.int32)
        # print("[*] img_read.shape:", img_read.size)  #shape) 
        # print("[*] imgArrayPart.shape:", imgArrayPart.shape) 
        # print("[*] imgArrayTrain[%]:", i, j, imgArrayTrain[j].shape) 
        imgArrayTrain[j] = imgArrayPart
        j+=1

    if(j>=126):
        break

imgArrayTrain = imgArrayTrain / 255
imgArrayTrain2 = imgArrayTrain.reshape(len(imgArrayTrain), height, width).astype(dtype=cupy.float32)
# imgArrayTrain2 = imgArrayTrain.reshape(len(imgArrayTrain), height, width).astype(dtype=cupy.int32)


#model class
class MyLSTM(chainer.Chain):
    def __init__(self):
        super(MyLSTM, self).__init__(
            Wz = L.Convolution2D(channelIn, channelOut, ksize, stride=1, pad=padsize),
            Wi = L.Convolution2D(channelIn, channelOut, ksize, stride=1, pad=padsize),
            Wf = L.Convolution2D(channelIn, channelOut, ksize, stride=1, pad=padsize),
            Wo = L.Convolution2D(channelIn, channelOut, ksize, stride=1, pad=padsize),
            Rz = L.Convolution2D(channelIn, channelOut, ksize, stride=1, pad=padsize),
            Ri = L.Convolution2D(channelIn, channelOut, ksize, stride=1, pad=padsize),
            Rf = L.Convolution2D(channelIn, channelOut, ksize, stride=1, pad=padsize),
            Ro = L.Convolution2D(channelIn, channelOut, ksize, stride=1, pad=padsize),
            #W = L.Linear(k, k),
        )

    def __call__(self, s): #s is expected to cupyArray(num, height, width)
        print("[*] s:", s.shape)
        accum_loss = None
        chan = channelIn
        hei = len(s[0])
        wid = len(s[0][0])
        h = Variable(cupy.zeros((1, chan, hei, wid), dtype=cupy.float32))
        c = Variable(cupy.zeros((1, chan, hei, wid), dtype=cupy.float32))
        # h = Variable(cupy.zeros((1, chan, hei, wid), dtype=cupy.int32))
        # c = Variable(cupy.zeros((1, chan, hei, wid), dtype=cupy.int32))
        print("[*] h:",h.shape, h.dtype)
        print("[*] c:",c.shape, c.dtype)
        
        print("[*] len(s):", len(s), )
        for i in range(len(s) - 1): #len(s) is expected to 26

            tx = Variable(cupy.array(s[i + 1], dtype=cupy.float32).reshape(1, chan, hei, wid))
            # tx = Variable(cupy.array(s[i + 1], dtype=cupy.int32).reshape(1, chan, hei, wid))
            print("[*] tx:",tx.shape, tx.dtype)
            x_k = Variable(cupy.array(s[i], dtype=cupy.float32).reshape(1, chan, hei, wid))
            # x_k = Variable(cupy.array(s[i], dtype=cupy.int32).reshape(1, chan, hei, wid))
            print("[*] x_k:",x_k.shape, x_k.dtype)
            # print("[*] self.Wz(x_k):",self.Wz(x_k))
            # print("[*] self.Rz(h):",self.Rz(h))
            z0 = self.Wz(x_k.astype(dtype=cupy.int32)) + self.Rz(h.astype(dtype=cupy.int32))
            print("[*] z0:",z0.shape, z0.dtype)
            z1 = F.tanh(z0)
            print("[*] z1:",z1.shape, z1.dtype)
            i0 = self.Wi(x_k) + self.Ri(h)
            i1 = F.sigmoid(i0)
            f0 = self.Wf(x_k) + self.Rf(h)
            f1 = F.sigmoid(f0)
            c = z1 * i1 + f1 * c
            o0 = self.Wo(x_k) + self.Ro(h)
            o1 = F.sigmoid(o0)
            h = o1 * F.tanh(c)
            loss = F.mean_squared_error(h, tx)
            accum_loss = loss if accum_loss is None else accum_loss + loss

        return accum_loss

#optimize
model = MyLSTM()
# cuda.get_device(0).use() #for GPU
cuda.get_device(2).use() #for GPU
model.to_gpu() #for GPU
optimizer = optimizers.Adam()
optimizer.setup(model)

#learning phase
for epoch in range(epochNum):
    
    print("epoch =", epoch)
    for j in range(5):
        
        print("now j is", j)
        s = imgArrayTrain[j*25:(j+1)*25 + 1,:]
        # s = imgArrayTrain[j*25:(j+1)*25 + 1,:].astype(cupy.int32)
        model.zerograds()
        # loss = model(s)
        # loss = model(s.astype(cupy.int32))
        loss = model(s.astype(np.int32))
        loss.backward()    
        optimizer.update()

print( 'learning is finished')

