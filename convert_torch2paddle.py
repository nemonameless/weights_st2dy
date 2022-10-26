import os
import sys
import paddle
from IPython import embed
import numpy as np
import torch

def convert(weights, weight_name_file, target_name):
    weight_name_map = {}
    with open(weight_name_file) as f:
        for line in f.readlines():
            fields = line.split()
            weight_name_map[fields[0]] = fields[1]
    dst = {}
    if 0:
        src = torch.hub.load('ultralytics/yolov5', 'yolov5s').state_dict() #torch.load(weights)#['state_dict']
    else:
        #src = torch.load(weights)['model'].state_dict()
        src = torch.load(weights)['state_dict']
    for k, v in weight_name_map.items():
        if 0: #'model.0.conv.weight' in k:  ### yolov5 use letterbox so no need to do bgr2rgb # ppdet默认读为rgb格式，yolox读入的bgr格式，因此第一个卷积核要转换一下
            w=src[k].cpu().numpy().astype('float32')
            rgb_w = np.copy(w)
            rgb_w[:, 0, :, :] = w[:, 2, :, :]
            rgb_w[:, 2, :, :] = w[:, 0, :, :]
            dst[v] = rgb_w
        else:
            dst[v] = src[k].cpu().numpy().astype('float32')
    #pickle.dump(dst, open(target_name, 'wb'), protocol=2)

    paddle.save(dst, target_name)

if __name__ == "__main__":
    weights = sys.argv[1]
    weight_name_file = sys.argv[2]
    target_name = sys.argv[3]
    convert(weights, weight_name_file, target_name)

