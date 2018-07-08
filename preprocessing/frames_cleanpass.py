#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image
import os, sys
import glob
from tqdm import tqdm, trange
import argparse

param_dict_list = [
    ["15mm_focallength", "35mm_focallength"],
    ["scene_backwards", "scene_forwards"],
    ["fast", "slow"],
    ["left", "right"]
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Preprocess')
    parser.add_argument('-base_path', type=str, default='/home/kaji/data/frames_cleanpass/')
    parser.add_argument('-out_dir', type=str, default='/home/kaji/data/frames_cleanpass/np_out/')
    parser.add_argument('-img_width', type=int, default=80)
    parser.add_argument('-img_height', type=int, default=40)
    args = parser.parse_args()
    base_path = args.base_path
    out_dir = args.out_dir
    img_width = args.img_width
    img_height = args.img_height
    ttt = 0
    for i1 in param_dict_list[0]:
        for i2 in param_dict_list[1]:
            for i3 in param_dict_list[2]:
                for i4 in param_dict_list[3]:
                    img_dir = base_path + i1 + "/" + i2 + "/" + i3 + "/" + i4 + "/"
                    img_paths = glob.glob(img_dir+"*.png")
                    array_list = []
                    img_paths = sorted(img_paths)
                    for img_path in img_paths:
                        if ttt == 0:
                            print(img_path)
                        im = Image.open(img_path).convert("RGB")
                        im = im.resize((img_width, img_height), Image.LANCZOS)
                        data = np.asarray(im)
                        data = data / 255.
                        array_list.append(data.tolist())
                    array_list = np.array(array_list).astype("float32")
                    np.save(out_dir+str(ttt)+'.npy', array_list)
                    print(ttt, img_dir)
                    ttt = ttt + 1
