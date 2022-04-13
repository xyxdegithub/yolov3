'''
Author: xyx
Date: 2022-03-31 19:01:57
LastEditTime: 2022-04-01 16:28:51
'''

import torch
import torch.nn as nn
from collections import OrderedDict

# 定义一个残差块的类
# 残差之后的结果没有改变大小
class ResidualBlock(nn.Module):
    # outchannels是个list
    def __init__(self, inchannels, outchannels) -> None:
        super(ResidualBlock,self).__init__()
        # Conv2d(in_channels,out_channels, kernel_size, stride, padding:, bias)
        # 使用bn，bias要设置为False
        self.conv1 = nn.Conv2d(inchannels,outchannels[0], (1, 1),1,0,bias=False)
        self.bn1 = nn.BatchNorm2d(outchannels[0])
        self.leakyrelu1 = nn.LeakyReLU(0.1)

        self.conv2=nn.Conv2d(outchannels[0],outchannels[1],(3,3),1,1,bias=False)
        self.bn2=nn.BatchNorm2d(outchannels[1])
        self.leakyrelu2=nn.LeakyReLU(0.1)

    def forward(self,x):
        residual=x

        out=self.leakyrelu1(self.bn1(self.conv1(x)))
        out=self.leakyrelu2(self.bn2(self.conv2(out)))

        return (residual+out)


#创建DarkNet主干网络类
#主干网络输入三个不同层次特征
class DarkNet(nn.Module):
    def __init__(self,num_repeat_list) -> None:
        super().__init__()
        self.inchannels=32
        #darknet53的第一层卷积
        self.conv1=nn.Conv2d(3,32,(3,3),1,1,bias=False)
        self.bn1=nn.BatchNorm2d(32)
        self.leakyrelu=nn.LeakyReLU(0.1)

        #组合相加
        self.layer1=self.make_layer([32,64],num_repeat_list[0])
        self.layer2=self.make_layer([64,128],num_repeat_list[1])
        self.layer3=self.make_layer([128,256],num_repeat_list[2])
        self.layer4=self.make_layer([256,512],num_repeat_list[3])
        self.layer5=self.make_layer([512,1024],num_repeat_list[4])  
   
    #make_layer函数包含一个卷积和残差块的组合
    #这里outchannels是一个list,re_num_repeat是残差块重复次数
    def make_layer(self,outchannels,re_num_repeat):
        layers=[]
        
        #一层卷积
        layers.append(("ds_conv",nn.Conv2d(self.inchannels,outchannels[1],(3,3),2,1,bias=False)))
        layers.append(("ds_bn",nn.BatchNorm2d(outchannels[1])))
        layers.append(("ds_relu",nn.LeakyReLU(0.1)))

        #这里复用上面的残差块
        #把上层的输出变成输入
        self.inchannels=outchannels[1]
        for i in range(re_num_repeat):
            layers.append(("Residual{}".format(i+1),ResidualBlock(self.inchannels,outchannels)))
        
        return nn.Sequential(OrderedDict(layers))


    def forward(self,x):
        out=self.leakyrelu1(self.bn1(self.conv1(x)))
        out=self.layer1(out)
        out=self.layer2(out)
        out52=self.layer3(out)
        out26=self.layer4(out52)
        out13=self.layer5(out26)
        return out52,out26,out13

def DarkNet53():
    #[1,2,8,8,4]是残差块重复的次数
    net=DarkNet(num_repeat_list=[1,2,8,8,4])
    print(net)
    return net

DarkNet53()

# def test():
#     re=ResidualBlock(64,[32,64])
#     print(re)

# test()
