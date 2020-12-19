import os
import sys
import pickle

def convert(weights, weight_name_file, target_name):
    weight_name_map = {}
    with open(weight_name_file) as f:
        for line in f.readlines():
            fields = line.split()
            weight_name_map[fields[0]] = fields[1]
    dst = {}
    src = pickle.load(open(weights))
    for k, v in weight_name_map.items():
        dst[v] = src[k]
    pickle.dump(dst, open(target_name, 'wb'), protocol=2)


if __name__ == "__main__":
    weight_path = sys.argv[1]
    weight_name_file = sys.argv[2]
    target_name = sys.argv[3]
    convert(weight_path, weight_name_file, target_name)
