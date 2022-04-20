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
def load_reds4(cfg):
    dataPath = cfg.DATASET.DATASETS
    hr_rootPath = '{}/{}'.format(dataPath,'train')
    lr_rootPath = '{}/{}'.format(dataPath,'train_sharp_bicubic/X4')


    folder_list = ['000','011','015','020']
    
    seq_list = []
    for folder in folder_list:
        folder_hr_path = '{}/{}'.format(hr_rootPath,folder)
        hr_urls = sorted(glob.glob(folder_hr_path+'/*.png'))
        # print('hr_urls',hr_urls)
        folder_lr_path = '{}/{}'.format(lr_rootPath,folder)
        lr_urls =  sorted(glob.glob(folder_lr_path+'/*.png'))

        assert len(hr_urls) == len(lr_urls),'not matched'
        seq_list.append([lr_urls,hr_urls,len(hr_urls)])
    return seq_list
if __name__ == '__main__':
    from config import config
    from network import RTA_VSR

    from utils.model_opr import load_model
    from utils.common import *
    


    model_root = '/home/hailangwu/zk/project/vsr/logs/baseline_reds/frame_align/models'
    path_files = glob.glob(model_root+'/*.pth')
    model_path = max(path_files,key = os.path.getmtime)

    model =   RTA_VSR(cfg = config)
    device = torch.device('cuda')
    model = model.to(device)

    load_model(model, model_path)
    model.eval()
    print("RTA_VSR(REDS) have {:.3f}M paramerters in total".format(sum(x.numel() for x in model.parameters())/1000000.0))

    
    write_path = '/home/hailangwu/zk/project/RTA_CVPR2022/models/VSR_REDS'
    if os.path.exists(write_path) is False:
        os.mkdir(write_path)

    seq_list = load_reds4(config)
    gidx = 0
    g_psnr_list = []
    g_ssim_list = []
    gg_psnr_list = []
    gg_ssim_list = []
    for fidx,seq in enumerate(seq_list) :
        fp_blur_list,fp_gt_list,sample_len = seq

        n = len(fp_blur_list)
        psnr_l = []
        ssim_l = []
        step = 1
        for i in range(n):
            gt_img = cv2.imread(fp_gt_list[i])
            img_name = fp_gt_list[i].split('/')[-1]
            idx_left_list = [None] * 2
            idx_right_list = [None] * 2
            for j in range(1,3):
                idx_left = max(i - j*step,0)
                idx_right = min(i+j*step,n-1)
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
            
            # print('lr_tensor',lr_tensor.shape)
            with torch.no_grad():
                sr_vsr = model(lr_tensor)
            
            sr_vsr = sr_vsr.detach().cpu().numpy()[0].astype(np.float32)
            sr_vsr = np.transpose(sr_vsr,(1,2,0))

       

            gt = hr_tensor.cpu().numpy()[0].astype(np.float32)
            gt = np.transpose(gt,(1,2,0))
          

            model_name_path = '{}/{}'.format(write_path,fidx)
            if os.path.exists(model_name_path) is False:
                os.mkdir(model_name_path)
            cv2.imwrite('{}/{}'.format(model_name_path,img_name),sr_vsr*255.0)


            sr_vsr = sr_vsr[2:-2,2:-2]
            gt = gt[2:-2,2:-2]
            psnr = calculate_psnr(sr_vsr*255.0, gt*255.0)
            ssim = calculate_ssim(sr_vsr*255.0, gt*255.0)

            psnr_l.append(psnr)
            ssim_l.append(ssim)
            g_psnr_list.append(psnr)
            g_ssim_list.append(ssim)

            print(gidx,psnr,ssim,i,idx_list)
            gidx +=1
            # break
        print(fidx,sum(psnr_l) / len(psnr_l),sum(ssim_l) / len(ssim_l))
        gg_psnr_list.append(sum(psnr_l) / len(psnr_l))
        gg_ssim_list.append(sum(ssim_l) / len(ssim_l))



    avg_psnr,avg_ssim = sum(gg_psnr_list) / len(gg_psnr_list),sum(gg_ssim_list) / len(gg_ssim_list)
    with open('/home/hailangwu/zk/project/RTA_CVPR2022/models/VSR_REDS/latest.txt','a') as f:
        f.write('\n')
        f.write('{}/{}'.format(avg_psnr,avg_ssim))