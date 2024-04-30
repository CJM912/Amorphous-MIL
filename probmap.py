# -*- coding: utf-8 -*-
"""

使用训练好的模型返回整张图片的概率矩阵
"""

import torch, time
from torch import nn
from MyLoader import OrigDataset as XDataset
from torchvision import transforms
import torch.backends.cudnn as cudnn
import resnet18
from utils import inference
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
p_path = ""
def main():
    # 定义网络
    #moment = time.time()
    model = resnet18.resnet18(False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load('checkpoint_50_best.pth')['state_dict'])
    #model.cuda()
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.CrossEntropyLoss().cuda()
    cudnn.benchmark = True
    # 定义数据集
    normalize = transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.1,0.1,0.1])
    trans = transforms.Compose([transforms.ToTensor(), normalize])
    test_dset = XDataset(p_path+'testphi-%d.lib'%(phi*1000),transform=trans)
    test_loader = torch.utils.data.DataLoader(
        test_dset,
        batch_size=128,shuffle=False,
        pin_memory=False)
    # 定义log文件
    # 开始迭代
    moment = time.time()
    test_dset.setmode(1)
    loss, probs = inference(0, test_loader, model, criterion)
    print(time.time() - moment)
    torch.save(probs, p_path+'probsphi-%d.pth'%(phi*1000))

if __name__ == '__main__':
    main()
