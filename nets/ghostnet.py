import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['ghostnet']  # http://c.biancheng.net/view/2401.html

def _make_divisible(v, divisor, min_value=None): # _make_divisible() 函数保证输出的数new_v可以整除divisor
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def hard_sigmoid(x, inplace: bool = False): # Sigmoid的近似，因为没有指数函数，所以计算比较快
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.

class SqueezeExcite(nn.Module): # 一个简单的注意力机制，和SENet结构类似
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)

        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x    

class ConvBnAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size,
                 stride=1, act_layer=nn.ReLU):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chs)
        self.act1 = act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x

class GhostModule(nn.Module): # 功能是代替普通卷积
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        '''inp, oup --> GhostModule的输入和输出通道数'''
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio) # 1*1卷积之后的通道数减半，因为后面要堆叠
        new_channels = init_channels*(ratio-1)
        # 利用1*1卷积对输入进来的特征图进行通道压缩，得到一个浓缩的特征（跨通道的特征整合）
        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        # 获得浓缩特征图之后，使用逐层卷积（深度可分离卷积是逐层+逐点），获得额外的特征图（跨特征点的特征提取）
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False), # groups=init_channels就设置了逐层卷积
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]


class GhostBottleneck(nn.Module):
    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3, stride=1, act_layer=nn.ReLU, se_ratio=0.):
        super(GhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride

        # 首先利用一个ghost模块进行特征提取，此时指定的通道数会比较大，可以看作是逆残差结构
        self.ghost1 = GhostModule(in_chs, mid_chs, relu=True)

        if self.stride > 1: # 根据输入进来的步长判断是否进行特征图高宽的压缩，如果是，则进行一个逐层卷积
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                             padding=(dw_kernel_size-1)//2,
                             groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        if has_se: # 但是在图片里面没有呈现
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None

        self.ghost2 = GhostModule(mid_chs, out_chs, relu=False) # 再次用ghost模块
        
        if (in_chs == out_chs and self.stride == 1): # 判断输入与输出通道是否一样，且步长为1，即不进行宽高调整
            self.shortcut = nn.Sequential()
        else: # 如果不满足，那么残差边需利用深度可分离卷积和1*1卷积调整通道数与宽高，保证两部分可以相加
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                       padding=(dw_kernel_size-1)//2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )


    def forward(self, x):
        residual = x

        x = self.ghost1(x) # 第一个ghost模块，特征提取

        if self.stride > 1: # 根据步长判断是否使用深度可分离卷积进行高宽的压缩
            x = self.conv_dw(x)
            x = self.bn_dw(x)

        if self.se is not None:
            x = self.se(x)

        x = self.ghost2(x) # 第二个ghost模块，特征提取
        
        x += self.shortcut(residual)
        return x


class GhostNet(nn.Module): # GhostModule -> GhostBottleneck -> GhostNet
    def __init__(self, cfgs, num_classes=1000, width=1.0, dropout=0.2):
        super(GhostNet, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        self.dropout = dropout

        # building first layer
        output_channel = _make_divisible(16 * width, 4) # 16

        # 416,416,3 --> 208,208,16 高宽压缩通道扩张
        self.conv_stem = nn.Conv2d(3, output_channel, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.act1 = nn.ReLU(inplace=True)
        input_channel = output_channel # 16

        # building inverted residual blocks
        stages = []
        block = GhostBottleneck # 一个class
        for cfg in self.cfgs:
            layers = []
            for k, exp_size, c, se_ratio, s in cfg: # 对每一行内的所有元素进行遍历，得到具体配置参数
                output_channel = _make_divisible(c * width, 4) # c
                hidden_channel = _make_divisible(exp_size * width, 4) # exp_size
                layers.append(block(input_channel, hidden_channel, output_channel, k, s,
                              se_ratio=se_ratio))
                input_channel = output_channel
            stages.append(nn.Sequential(*layers))


        # 最后这些用于调整通道数与分类层的在YOLOv4里面用不到，不用看了
        output_channel = _make_divisible(exp_size * width, 4)
        stages.append(nn.Sequential(ConvBnAct(input_channel, output_channel, 1)))
        input_channel = output_channel
        
        self.blocks = nn.Sequential(*stages)        

        # building last several layers
        output_channel = 1280
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_head = nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=True)
        self.act2 = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(output_channel, num_classes)

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        
        x = self.blocks(x)

        x = self.global_pool(x)

        x = self.conv_head(x)
        x = self.act2(x)
        x = x.view(x.size(0), -1)
        if self.dropout > 0.:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)
        return x


def ghostnet(**kwargs):
    """
    Constructs a GhostNet model
    """
    cfgs = [
        # k, t, c, SE, s -->
        # 卷积核大小（表示跨特征点的特征提取能力），第一个ghost模块的通道数（一般值较大），瓶颈结构最终的输出通道数，是否使用注意力机制，瓶颈结构的步长（大于1就进行宽高的压缩）

        # stage1  208,208,16 --> 208,208,16
        [[3,  16,  16, 0, 1]],
        # stage2  208,208,16 --> 104,104,24
        [[3,  48,  24, 0, 2]],
        [[3,  72,  24, 0, 1]],
        # stage3  104,104,24 --> 52,52,40（out）
        [[5,  72,  40, 0.25, 2]],
        [[5, 120,  40, 0.25, 1]],
        # stage4  52,52,40 --> 26,26,112（out）
        [[3, 240,  80, 0, 2]],
        [[3, 200,  80, 0, 1],
         [3, 184,  80, 0, 1],
         [3, 184,  80, 0, 1],
         [3, 480, 112, 0.25, 1],
         [3, 672, 112, 0.25, 1]
        ],
        # stage5 26,26,112 --> 13,13,160（out）
        [[5, 672, 160, 0.25, 2]],
        [[5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1],
         [5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1]
        ]
    ]
    return GhostNet(cfgs, **kwargs)


if __name__ == "__main__":
    from torchsummary import summary

    # 需要使用device来指定网络在GPU还是CPU运行
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ghostnet().to(device)
    summary(model, input_size=(3,224,224))

