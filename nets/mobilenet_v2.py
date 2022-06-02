from torch import nn
import torch
from torchvision.models.utils import load_state_dict_from_url

model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )

# https://blog.csdn.net/u011995719/article/details/79135818
class InvertedResidual(nn.Module): # Resnet是先压缩通道数再扩张，MobileNetv2则是先扩再压，因此称为InvertedResidual
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))

        # 针对stride=1 和stride=2，在block上有稍微不同，主要是为了与shortcut的维度匹配，因此，stride=2时，不采用shortcut
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
            
        layers.extend([
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim), # 设置了groups参数，这一步为深度可分离卷积

            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), # 1*1卷积降维
            nn.BatchNorm2d(oup), 
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0, inverted_residual_setting=None, round_nearest=8):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s ---升维比例expand_ratio，大结构块输出通道数，大结构块中InvertedResidual重复的次数，输入进大结构块后高宽是否压缩
                # 208,208,32 -> 208,208,16
                [1, 16, 1, 1],
                # 208,208,16 -> 104,104,24
                [6, 24, 2, 2],
                # 104,104,24 -> 52,52,32（out1）
                [6, 32, 3, 2],

                # 52,52,32 -> 26,26,64
                [6, 64, 4, 2],
                # 26,26,64 -> 26,26,96（out2）
                [6, 96, 3, 1],
                
                # 26,26,96 -> 13,13,160
                [6, 160, 3, 2],
                # 13,13,160 -> 13,13,320（out3）
                [6, 320, 1, 1],
            ]

        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)

        # 416,416,3 -> 208,208,32
        features = [ConvBNReLU(3, input_channel, stride=2)]

        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        self.features = nn.Sequential(*features)

        # 下面都是分类网络，在YOLOv4里面用不到
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x

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
def mobilenet_v2(pretrained=False, progress=True):
    model = MobileNetV2()
    if pretrained: # 改成加载下载好了的权重
        _load_weights(model, '../model_data/mobilenet_v2-b0353104.pth')
        # state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'], model_dir="model_data",progress=progress)
        # model.load_state_dict(state_dict)
    return model


if __name__ == "__main__":

    from torchsummary import summary
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m = mobilenet_v2(True).to(device)
    # summary(m, input_size=(3, 416, 416))
    for i,feature in enumerate(m.features):
        print(i, feature)