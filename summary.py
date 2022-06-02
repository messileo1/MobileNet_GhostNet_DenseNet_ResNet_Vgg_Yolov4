#--------------------------------------------#
#   该部分代码用于看网络结构
#--------------------------------------------#
import torch
from thop import clever_format, profile
from torchsummary import summary

from nets.yolo import YoloBody

if __name__ == "__main__":
    input_shape     = [416, 416]
    anchors_mask    = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    num_classes     = 80
    # mobilenetv1, mobilenetv2, mobilenetv3, ghostnet, vgg, densenet121, densenet169, densenet201, resnet50
    backbone        = 'mobilenetv3'
    
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m       = YoloBody(anchors_mask, num_classes, backbone=backbone).to(device)
    summary(m, (3, input_shape[0], input_shape[1]))

    # 原版的YOLOv4为 64,363,101

    # mobilenetv1-yolov4 40,952,893(参数量除以1000000就是M，所以有40.95M)
    # mobilenetv2-yolov4 39,062,013
    # mobilenetv3-yolov4 39,989,933

    # 修改了panet（用深度可分离卷积代替普通3*3卷积）的mobilenetv1-yolov4 12,692,029
    # 修改了panet的mobilenetv2-yolov4 10,801,149
    # 修改了panet的mobilenetv3-yolov4 11,729,069
    # 修改了panet的ghostnet-yolov4 11,428,545
    # 修改了panet的densenet121-yolov4 16,438,909
    # 修改了panet的densenet201-yolov4 28,135,037
    # 修改了panet的densenet169-yolov4 22, 329, 981
    # 修改了panet的vgg-yolov4 23,937,597
    # 修改了panet的resnet50-yolov4 33,681,213

    dummy_input     = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
    flops, params   = profile(m.to(device), (dummy_input, ), verbose=False)
    #--------------------------------------------------------#
    #   flops * 2是因为profile没有将卷积作为两个operations
    #   有些论文将卷积算乘法、加法两个operations。此时乘2
    #   有些论文只考虑乘法的运算次数，忽略加法。此时不乘2
    #   本代码选择乘2，参考YOLOX。
    #--------------------------------------------------------#
    flops           = flops * 2
    flops, params   = clever_format([flops, params], "%.3f")
    print('Total GFLOPS: %s' % (flops)) # 计算量
    print('Total params: %s' % (params)) # 参数量
    