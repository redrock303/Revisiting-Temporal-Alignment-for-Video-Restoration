import cv2
import os

import torch
import torchvision
import sys 
sys.path.append('/home/hailangwu/zk/project/RTA_CVPR2022/')
import numpy as np

def PSNR(output, target, max_val = 1.0):
    output = output.clamp(0.0,1.0)
    mse = torch.pow(target - output, 2).mean()
    if mse == 0:
        return torch.Tensor([100.0])
    return 10 * torch.log10(max_val**2 / mse)

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

    
    model_name = './test'
    if os.path.exists(model_name) is False:
        os.mkdir(model_name)

    psnr_l = []
    ssim_l = []

    seq_list = load_data()
    for fidx,seq in enumerate(seq_list) :
        fp_blur_list,fp_gt_list,sample_len = seq

        n = len(fp_blur_list)

        
        
        for i in range(n):
            gt_img = cv2.imread(fp_gt_list[i])
            img_name = fp_gt_list[i].split('/')[-1]
            idx_left_list = [None] * 2
            idx_right_list = [None] * 2
            for j in range(1,3):
                idx_left = max(i - j,0)
                idx_right = min(i+j,n-1)
                idx_left_list[2-j] = idx_left
                idx_right_list[j-1] = idx_right
            idx_list = idx_left_list + [i] + idx_right_list

            frame_lr = [cv2.imread(fp_blur_list[idx_j]) for idx_j in idx_list]

            # print(i,idx_list)
            lr_data =  np.stack(frame_lr,0)
            hr_data = gt_img.transpose(2,0,1)
            lr_data = lr_data.transpose(0,3,1,2)

            hr_tensor = torch.from_numpy(hr_data.astype(np.float32) / 255.0).float().unsqueeze(0).to(device)
            lr_tensor = torch.from_numpy(lr_data.astype(np.float32) / 255.0).float().unsqueeze(0).to(device)
            
            with torch.no_grad():
                deblurred_img = model(lr_tensor) 
                deblurred_img = deblurred_img.clamp(0,1)

            psnr = float(PSNR(deblurred_img,hr_tensor))

            

            gt = hr_tensor.cpu().numpy()[0].astype(np.float32)
            gt = np.transpose(gt,(1,2,0))

            deblurred_img = deblurred_img.detach().cpu().numpy()[0].astype(np.float32)    
            deblurred_img = np.transpose(deblurred_img,(1,2,0))

            model_name_path = '{}/{}'.format(model_name,fidx)
            if os.path.exists(model_name_path) is False:
                os.mkdir(model_name_path)
            cv2.imwrite('{}/{}'.format(model_name_path,img_name),deblurred_img*255.0)


            if True:
                deblurred_img = bgr2ycbcr(deblurred_img, only_y=True)
                gt = bgr2ycbcr(gt, only_y=True)

            ssim1 = calculate_ssim(deblurred_img*255,gt*255)

            psnr_l.append(psnr)
            ssim_l.append(ssim1)

            print('category',fidx,i,psnr,ssim1)

    print(sum(psnr_l) / len(psnr_l))
    print(sum(ssim_l) / len(ssim_l))
    with open('60000_ft_rf.txt','w') as f:
        f.write('{}/{}'.format(sum(psnr_l) / len(psnr_l),sum(ssim_l) / len(ssim_l)))
