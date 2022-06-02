'''DenseNet的实现参考知乎-->https://zhuanlan.zhihu.com/p/37189203
    与github-->https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py'''

import re
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torchvision.models.utils import load_state_dict_from_url

__all__ = ['DenseNet', 'densenet121', 'densenet169', 'densenet201'] # http://c.biancheng.net/view/2401.html

model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
}


'''在每个denseblock中，特征层的传递之间，输入是前面所有特征图的堆叠，输出都是k个通道数的特征图，用下面这个函数来实现，
即通过一次bn+relu+conv来实现'''
def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1) # dim = 1 沿通道方向堆叠
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function

# _DenseLayer-->_DenseBlock+_Transition-->DenseNet

class _DenseLayer(nn.Sequential): # DenseBlock中的内部结构，这里是BN+ReLU+1x1 Conv+BN+ReLU+3x3 Conv结构
    '''Basic unit of DenseBlock (using bottleneck layer) '''
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
        super(_DenseLayer, self).__init__()

        # BN+ReLU+1x1 Conv+BN+ReLU+3x3 Conv
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        # 通过add_module注册网络，等价于self.norm1 = nn.BatchNorm2d(num_input_features)
        '''
            注册网络有三种方式：https://www.cnblogs.com/datasnail/p/14903643.html
            1.self.norm1 = nn.BatchNorm2d(num_input_features)
            2.self.add_module('norm1', nn.BatchNorm2d(num_input_features))
            3.nn.ModuleList封装：self.layers = nn.ModuleList([nn.Linear(28*28,28*28) for _ in range(layer_num)])
            
            所以如果想在网络中注册注意力机制，下面这种方式可以：
            self.attention = eca(channel)
            out = self.attention(input)
            而这种方式就是错误的，无法注册到网络中：
            attention = eca(channel)
            out = attention(input)
        '''

        self.add_module('relu1', nn.ReLU(inplace=True)),

        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),

        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate          = drop_rate
        self.memory_efficient   = memory_efficient # 默认FALSE

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        '''为什么要用到下面这个条件判断？因为DenseNet会占用大量显存，用checkpoint是一种以时间换空间的策略，
        具体见：https://www.cnblogs.com/jiangkejie/p/13049684.html'''
        if self.memory_efficient and any(prev_feature.requires_grad for prev_feature in prev_features): #any中只要有一个不为false就返回True
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features

class _DenseBlock(nn.Module):
    '''依据_DenseLayer实现DenseBlock模块，内部是密集连接方式（输入特征数线性增长）'''
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False):
        super(_DenseBlock, self).__init__()
        #---------------------------------------------------------#
        #   Denselayer的功能就是将具有一定通道的输入
        #   压缩成growth_rate通道的特征层
        #---------------------------------------------------------#
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate         = growth_rate,
                bn_size             = bn_size,
                drop_rate           = drop_rate,
                memory_efficient    = memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)

class _Transition(nn.Sequential): # DenseBlock与DenseBlock之间通过_Transition保持特征图大小一致,主要是一个卷积层和一个池化层
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        #--------------------------------------#
        #   进行高宽的压缩,降低特征图大小。
        #--------------------------------------#
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

class DenseNet(nn.Module):
    '''所有DenseBlock中各个层卷积之后均输出k个特征图，即得到的特征图的channel数为k，
    或者说采用k个卷积核。 k在DenseNet称为growth rate，这是一个超参数。'''
    """
    :param growth_rate: (int) number of filters used in DenseLayer, `k` in the paper
    :param block_config: (list of 4 ints) number of layers in each DenseBlock
    :param num_init_features: (int) number of filters in the first Conv2d
    :param bn_size: (int) the factor using in the bottleneck layer
    :param compression_rate: (float) the compression rate used in Transition Layer
    :param drop_rate: (float) the drop rate after each DenseLayer
    :param num_classes: (int) number of classes for classification
    """
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000, memory_efficient=False):
        super(DenseNet, self).__init__()
        #--------------------------------------#
        #   第一次卷积和最大池化
        #--------------------------------------#
        self.features = nn.Sequential(OrderedDict([
            # 416, 416, 3 -> 208, 208, 64
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            # 208, 208, 64 -> 104, 104, 64
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        #--------------------------------------#
        #   构建四个阶段的dense_block
        #--------------------------------------#
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            #--------------------------------------#
            #   构建密集残差结构
            #--------------------------------------#
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            #--------------------------------------#
            #   计算经过密集残差后的通道数
            #--------------------------------------#
            num_features = num_features + num_layers * growth_rate

            if i != len(block_config) - 1:
                #--------------------------------------#
                #   进行高宽的压缩，除了最后一个_Denseblock
                #--------------------------------------#
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                #--------------------------------------#
                #   计算下一时刻的输入通道数
                #--------------------------------------#
                num_features = num_features // 2

        #--------------------------------------#
        #   标准化和激活函数
        #--------------------------------------#
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer，分类层，YOLO用不到，不用管。
        self.classifier = nn.Linear(num_features, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

def _load_state_dict(model, model_url, progress): # 还得从网上下载，有可能下载很慢，不建议用
    pattern = re.compile(r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

    state_dict = load_state_dict_from_url(model_url, model_dir="model_data", progress=progress)
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    model.load_state_dict(state_dict)

import numpy as np
def _load_weights(model, weights_path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(weights_path)
    load_key, no_load_key, temp_dict = [], [], {}
    for k, v in pretrained_dict.items():
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            temp_dict[k] = v
            load_key.append(k)
        else:
            no_load_key.append(k)
    model_dict.update(temp_dict)
    model.load_state_dict(model_dict)

def _densenet(arch, growth_rate, block_config, num_init_features, pretrained, progress,
              **kwargs):
    model = DenseNet(growth_rate, block_config, num_init_features, **kwargs)
    if pretrained: # 若修改结构，pretrained最好设置为true加载主干的权重
        # _load_state_dict(model, model_urls[arch], progress)
        if arch == 'densenet121':
            # model.load_state_dict(torch.load('../model_data/densenet121-a639ec97.pth'))  --有的地方不匹配
            _load_weights(model, '../model_data/densenet121-a639ec97.pth')
        elif arch == 'densenet169':
            # model.load_state_dict(torch.load('../model_data/densenet169-b2777c0a.pth'))  --有的地方不匹配
            _load_weights(model, '../model_data/densenet169-b2777c0a.pth')
        else:
            # model.load_state_dict(torch.load('../model_data/densenet201-c1103571.pth'))  --有的地方不匹配
            _load_weights(model, '../model_data/densenet201-c1103571.pth')
    return model

def densenet121(pretrained=False, progress=True, **kwargs):
    return _densenet('densenet121', 32, (6, 12, 24, 16), 64, pretrained, progress,
                     **kwargs)


def densenet169(pretrained=False, progress=True, **kwargs):
    return _densenet('densenet169', 32, (6, 12, 32, 32), 64, pretrained, progress,
                     **kwargs)


def densenet201(pretrained=False, progress=True, **kwargs):
    return _densenet('densenet201', 32, (6, 12, 48, 32), 64, pretrained, progress,
                     **kwargs)

if __name__ == "__main__":
    import torch
    from torchsummary import summary

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m = densenet121(True).to(device)
    # print(m)
    summary(m, input_size=(3, 416, 416))
