import numpy as np
from PIL import Image

#---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
#---------------------------------------------------------#
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 
    
#---------------------------------------------------#
#   对输入图像进行resize
#---------------------------------------------------#
def resize_image(image, size, letterbox_image):
    iw, ih  = image.size
    w, h    = size
    if letterbox_image: # letterbox_image为bool类型，用于判断是否进行不失真的resize
        scale   = min(w/iw, h/ih)
        nw      = int(iw*scale)
        nh      = int(ih*scale)

        image   = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image

#---------------------------------------------------#
#   获得类
#---------------------------------------------------#
def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

#---------------------------------------------------#
#   获得先验框
#---------------------------------------------------#
def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path, encoding='utf-8') as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    anchors = np.array(anchors).reshape(-1, 2)
    return anchors, len(anchors)

#---------------------------------------------------#
#   获得学习率
#---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups: # param_group是以字典形式返回的，所以可以用键获取对应的键值
        return param_group['lr']

def preprocess_input(image):
    image /= 255.0
    return image

def show_config(**kwargs): # 用来看：主干，权重路径，classes_path等配置参数
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)

# 已经下载好了，记得注释掉
def download_weights(backbone, model_dir="./model_data"):
    import os
    from torch.hub import load_state_dict_from_url
    
    download_urls = {
        'densenet121'   : 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
        'densenet169'   : 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
        'densenet201'   : 'https://download.pytorch.org/models/densenet201-c1103571.pth',
        'resnet50'      : 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
        'vgg'           : "https://download.pytorch.org/models/vgg16-397923af.pth",
        'mobilenetv1'   : 'https://github.com/bubbliiiing/mobilenet-yolov4-pytorch/releases/download/v1.0/mobilenet_v1_weights.pth',
        'mobilenetv2'   : 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
        'mobilenetv3'   : 'https://github.com/bubbliiiing/mobilenet-yolov4-pytorch/releases/download/v1.0/mobilenetv3-large-1cd25616.pth',
        'ghostnet'      : 'https://github.com/bubbliiiing/mobilenet-yolov4-pytorch/releases/download/v1.0/ghostnet_weights.pth',
    }
    url = download_urls[backbone]
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    load_state_dict_from_url(url, model_dir)