import cv2
import numpy as np
import os
import sys

import torch
import torch.utils.data as tData
import glob

import random
import json 
import sys

def read_json(json_path):
    with open(json_path,'r') as f:
        file_list = json.load(f)
    return file_list

def load_data(json_path = '/data/home/hailangwu/zk/project/RTA_CVPR2022/models/VDB/VideoDeblur.json',\
                rootPath = '/home/hailangwu/zk/data/quantitative_datasets',split='test'):
    file_list = read_json(json_path)
    seq_list = []
    for fp in file_list:
        folder_name,phase,sample = fp['name'],fp['phase'],fp['sample']
        if phase not in split:
            continue
        folder_path = os.path.join(rootPath,folder_name)
        fp_blur_list = [os.path.join(folder_path,'input/{}.jpg'.format(sample_name)) for sample_name in sample]
        fp_gt_list = [os.path.join(folder_path,'GT/{}.jpg'.format(sample_name)) for sample_name in sample]
        sample_len = len(fp_blur_list)
        assert len(fp_blur_list) == len(fp_gt_list)
        seq_list.append([fp_blur_list,fp_gt_list,sample_len])
    
    
    print(split,len(seq_list))
    return seq_list
