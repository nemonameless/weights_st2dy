import os
import sys
import paddle
import torch
import numpy as np

def convert(weights, weight_name_file, target_name):
    weight_name_map = {}
    with open(weight_name_file) as f:
        for line in f.readlines():
            fields = line.split()
            weight_name_map[fields[0]] = fields[1]
    dst = {}
    src = torch.load(weights)
    src = src['state_dict']
    for k, v in weight_name_map.items():
        dst[v] = np.array(src[k].cpu())
    paddle.save(dst, '{}'.format(target_name))


if __name__ == "__main__":
    weight_path = sys.argv[1]
    weight_name_file = sys.argv[2]
    target_name = sys.argv[3]
    convert(weight_path, weight_name_file, target_name)