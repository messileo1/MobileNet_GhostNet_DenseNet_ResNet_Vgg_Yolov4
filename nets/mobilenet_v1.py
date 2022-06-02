import torch
import torch.nn as nn


def conv_bn(inp, oup, stride = 1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )
# 深度可分离卷积的实现：卷积group设置成in_filters（逐层卷积），然后再利用1x1卷积调整channels数（逐点卷积）
def conv_dw(inp, oup, stride = 1):
    return nn.Sequential(
        # part1
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU6(inplace=True),

        # part2
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True),
    )

class MobileNetV1(nn.Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()
        self.stage1 = nn.Sequential(
            # 416,416,3 -> 208,208,32
            conv_bn(3, 32, 2),
            # 208,208,32 -> 208,208,64
            conv_dw(32, 64, 1), 

            # 208,208,64 -> 104,104,128
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),

            # 104,104,128 -> 52,52,256（特征层1）
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1), 
        )
            # 52,52,256 -> 26,26,512（特征层2）
        self.stage2 = nn.Sequential(
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1), 
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
        )
            # 26,26,512 -> 13,13,1024（特征层3）
        self.stage3 = nn.Sequential(
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
        )

        # 下面这俩用于分类，不用管
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(1024, 1000)

    # 其实三个阶段最后输出的特征层就是构建网络所需要的初步的特征层
    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avg(x)
        # x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x

def mobilenet_v1(pretrained=False, progress=True):
    model = MobileNetV1()
    if pretrained:
        state_dict = torch.load('../model_data/mobilenet_v1_weights.pth')
        model.load_state_dict(state_dict, strict=True)
    return model

if __name__ == "__main__":
    import torch
    from torchsummary import summary

    # 需要使用device来指定网络在GPU还是CPU运行
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = mobilenet_v1(pretrained=False).to(device)
    summary(model, input_size=(3, 416, 416))
