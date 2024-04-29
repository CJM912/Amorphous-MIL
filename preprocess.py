# -*- coding: utf-8 -*-
"""
生成train.lib和test.lib
要求必须保证正例聚集在前，反例聚集在后
"""

#数据预处理，处理成标准lib

import torch, os, time
import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

moment = time.time()
box = 50 #切割小块的边长
positives = os.listdir('positives')
negatives = os.listdir('negatives')
positives_t = os.listdir('positives_t')
negatives_t = os.listdir('negatives_t')
sum=0
aaanum=0

strongtile = os.listdir('strongtile')

sum=len(positives)+len(negatives)+len(positives_t)+len(negatives_t)+len(strongtile)
print(sum)
trainlibrary = {}
testlibrary = {}

#切块
def makelist(typ,inputnum):
    aaanum=inputnum
    global box
    slides = []
    grid = []
    targets = []
    names = os.listdir(typ)
    if typ=='strongtile':
        for file in names:

            img=Image.open('%s/%s'%(typ, file))
            img=img.convert('RGB')
            img=img.resize((50,50)).convert('RGB')
            data = np.array(img)
            height, width= data.shape[:2]
            h_num = (height - box) / (box // 2) + 1
            hs = []
            for i in range(int(h_num)):
                hs.append((i+1)*(box // 2))
            if int(h_num) != h_num:
                hs.append(height-(box // 2))
            w_num = (width - box) / (box // 2) + 1
            ws = []
            for i in range(int(w_num)):
                ws.append((i+1)*(box // 2))
            if int(w_num) != w_num:
                ws.append(width-(box // 2))
            hws = []
            for h in hs:
                for w in ws:
                    hws.append([h,w])
            grid.append(np.array(hws))
            slides.append(file)
            targets.append(1)
            aaanum=aaanum+1
            print(aaanum/sum)
    else:
        for file in names:
            img=Image.open('%s/%s'%(typ, file))
            img=img.convert('RGB')
            img=img.resize((512,512)).convert('RGB')
            data = np.array(img)
            height, width= data.shape[:2]
            h_num = (height - box) / (box // 2) + 1
            hs = []
            for i in range(int(h_num)):
                hs.append((i+1)*(box // 2))
            if int(h_num) != h_num:
                hs.append(height-(box // 2))
            w_num = (width - box) / (box // 2) + 1
            ws = []
            for i in range(int(w_num)):
                ws.append((i+1)*(box // 2))
            if int(w_num) != w_num:
                ws.append(width-(box // 2))
            hws = []
            for h in hs:
                for w in ws:
                    hws.append([h,w])
            grid.append(np.array(hws))
            slides.append(file)
            targets.append(1 if typ=='positives' or typ=='positives_t' else 0)
            aaanum=aaanum+1
            print(aaanum/sum)
    return grid, slides, targets, aaanum

#切块后压缩
pos_grid, pos_slides, pos_targets,pos_num = makelist('positives',0)
neg_grid, neg_slides, neg_targets,neg_num = makelist('negatives',pos_num)
str_grid, str_slides, str_targets,str_num = makelist('strongtile',neg_num)

pos_gridt, pos_slidest, pos_targetst,post_num = makelist('positives_t',str_num)
neg_gridt, neg_slidest, neg_targetst,negt_tem = makelist('negatives_t',post_num)

trainlibrary['targets'] = pos_targets[:] + neg_targets[:] + str_targets[:]
trainlibrary['slides'] = pos_slides[:] + neg_slides[:] + str_slides[:]
trainlibrary['grid'] = pos_grid[:] + neg_grid[:] + str_grid[:]
trainlibrary['size'] = box
torch.save(trainlibrary, 'train-%d.lib'%box)

testlibrary['targets'] = pos_targetst[:] + neg_targetst[:]
testlibrary['slides'] = pos_slidest[:] + neg_slidest[:]
testlibrary['grid'] = pos_gridt[:] + neg_gridt[:]
testlibrary['size'] = box
torch.save(testlibrary, 'test-%d.lib'%box)

print(testlibrary)
print('len(trainlibrary)',len(trainlibrary))
print('len(testlibrary)',len(testlibrary))

print(time.time() - moment)
