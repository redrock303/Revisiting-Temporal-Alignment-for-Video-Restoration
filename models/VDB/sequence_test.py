import cv2
import os

import torch
import torchvision
import sys 
sys.path.append('/home/hailangwu/zk/project/vsr/')
from utils.common import tensor2img, calculate_psnr, calculate_ssim, bgr2ycbcr
import numpy as np
import glob
if __name__ == '__main__':
    from config import config
    from network import RTA_VDB

    from utils.model_opr import load_model
    from utils.common import *
    from load_VDB_Data import load_data


    model =   RTA_VDB(cfg = config)
    device = torch.device('cuda')
    model = model.to(device)

    load_model(model, config.INIT_MODEL)
    model.eval()
    print("RTA_VDB have {:.3f}M paramerters in total".format(sum(x.numel() for x in model.parameters())/1000000.0))


    result_path = './result_IMG_0030'
    if os.path.exists(result_path) is False:
        os.mkdir(result_path)

    input_path = '/home/hailangwu/zk/data/quantitative_datasets/IMG_0030/input'
    files_bl = sorted(glob.glob(input_path+'/*.jpg'))
    n = len(files_bl)
    print('files_bl',input_path,len(files_bl))
    for i in range(0,n,5):
        if i +5 >=n-1:
            break
        img_name = files_bl[i].split('/')[-1]
        idx_left_list = [None] * 2
        idx_right_list = [None] * 2
        for j in range(1,3):
            idx_left = max(i - j,0)
            idx_right = min(i+j,n-1)
            idx_left_list[2-j] = idx_left
            idx_right_list[j-1] = idx_right
        idx_list = idx_left_list + [i] + idx_right_list

        frame_lr = [cv2.imread(files_bl[idx_j])[:,:] for idx_j in idx_list]

        print(i,idx_list)
        lr_data =  np.stack(frame_lr,0)
       
        lr_data = lr_data.transpose(0,3,1,2)

     
        lr_tensor = torch.from_numpy(lr_data.astype(np.float32) / 255.0).float().unsqueeze(0).to(device)

        with torch.no_grad():
            h, w = lr_tensor.size()[3:]

            deblurred_img = model(lr_tensor)
        deblurred_img = deblurred_img.clamp(0,1)
        deblurred_img = deblurred_img.detach().cpu().numpy()[0].astype(np.float32)    
        deblurred_img = np.transpose(deblurred_img,(1,2,0))

        img_stack = np.hstack([frame_lr[2].astype(np.float)/255.0,deblurred_img])
        cv2.imwrite('{}/{}'.format(result_path,img_name),img_stack*255.0)