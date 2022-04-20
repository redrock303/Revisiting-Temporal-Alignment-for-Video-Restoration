import cv2
import os

import torch
import torchvision
import sys 
sys.path.append('/home/hailangwu/zk/project/RTA_CVPR2022/')
import numpy as np
import glob 
def PSNR(output, target, max_val = 1.0):
    output = output.clamp(0.0,1.0)
    mse = torch.pow(target - output, 2).mean()
    if mse == 0:
        return torch.Tensor([100.0])
    return 10 * torch.log10(max_val**2 / mse)
def load_Vimeo90KT(cfg):
    dataPath = cfg.DATASET.DATASETS
    sep_txt = os.path.join(dataPath,'sep_testlist.txt')
    file_folder_list = []
    with open(sep_txt,'r') as f:
        while True:
            _lineStr = f.readline().strip('\n')
            if len(_lineStr) < 2:
                break
            folder_path = os.path.join(dataPath,'sequences',_lineStr)
            
            file_folder_list.append(folder_path)
    return file_folder_list
if __name__ == '__main__':
    from config import config
    from network import RTA_VSR

    from utils.model_opr import load_model
    from utils.common import *

    model =   RTA_VSR(cfg = config)
    device = torch.device('cuda')
    model = model.to(device)

    load_model(model, config.INIT_MODEL)
    model.eval()
    print("RTA_VSR have {:.3f}M paramerters in total".format(sum(x.numel() for x in model.parameters())/1000000.0))

    
    write_path = '/home/hailangwu/zk/project/RTA_CVPR2022/models/VSR_VIMEO90K/result'
    if os.path.exists(write_path) is False:
        os.mkdir(write_path)

    seq_list = load_Vimeo90KT(config)
    gidx = 0
    psnr_list = []
    ssim_list = []
    idx_list =  [idx+1 for idx in range(config.DATASET.NFRAME )]
    for fidx,seq in enumerate(seq_list) :
        folder_path = seq_list[fidx]

        frame_hr = cv2.imread(os.path.join(folder_path,'im4.png'))
        frame_lr = [cv2.imread(os.path.join(folder_path,'x{}'.format(4),'im{}_bicx{}.png'.format(idx,4))) for idx in idx_list]
        frame_sta = np.stack(frame_lr,0)

        hr_data = frame_hr.transpose(2,0,1)
        lr_data = frame_sta.transpose(0,3,1,2)

        hr_tensor = torch.from_numpy(hr_data.astype(np.float32) / 255.0).float().unsqueeze(0).to(device)
        lr_tensor = torch.from_numpy(lr_data.astype(np.float32) / 255.0).float().unsqueeze(0).to(device)
        with torch.no_grad():
            sr_vsr = model(lr_tensor)

        sr_vsr = sr_vsr.clamp(0,1)
        sr_vsr = sr_vsr.detach().cpu().numpy()[0].astype(np.float32)

        sr_vsr = np.transpose(sr_vsr,(1,2,0))


        gt = hr_tensor.cpu().numpy()[0].astype(np.float32)
        gt = np.transpose(gt,(1,2,0))

            
        
        # cv2.imwrite('{}/{}'.format(write_path,'{}.png'.format(str(fidx).zfill(6))),sr_vsr*255.0)

        if True:
            sr_vsr = bgr2ycbcr(sr_vsr[2:-2,2:-2], only_y=True)
            gt = bgr2ycbcr(gt[2:-2,2:-2], only_y=True)
                


        psnr = calculate_psnr(sr_vsr*255.0, gt*255.0)
        ssim = calculate_ssim(sr_vsr*255.0, gt*255.0)



        psnr_list.append(psnr)
        ssim_list.append(ssim)



    avg_psnr,avg_ssim = sum(psnr_list) / len(ssim_list),sum(ssim_list) / len(ssim_list)
    print('avg_psnr,avg_ssim',avg_psnr,avg_ssim)
    with open('/home/hailangwu/zk/project/RTA_CVPR2022/models/VSR_VIMEO90K/latest.txt','a') as f:
        f.write('\n')
        f.write('{}/{}'.format(avg_psnr,avg_ssim))