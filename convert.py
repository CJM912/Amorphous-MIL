# -*- coding:utf-8  -*-
from PIL import Image
import os
import string
from matplotlib import pyplot as plt

#转换图像数据大小

path=''
filelist=os.listdir(path)

for file in filelist:
    whole_path = os.path.join(path, file)
    print(whole_path)
    img = Image.open(whole_path)  
    img = img.resize((512,512)).convert("RGB")
    save_path = ''
    #img.save(save_path + img1)
    img.save(os.path.join(save_path,file))