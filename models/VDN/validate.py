import cv2
import os

import torch
import torchvision
import sys 
sys.path.append('/home/hailangwu/zk/project/RTA_CVPR2022/')
from utils.common import tensor2img, calculate_psnr, calculate_ssim, bgr2ycbcr
from utils.core import imresize
import numpy as np

def PSNR(output, target, max_val = 1.0):
    output = output.clamp(0.0,1.0)
    mse = torch.pow(target - output, 2).mean()
    if mse == 0:
        return torch.Tensor([100.0])
    return 10 * torch.log10(max_val**2 / mse)

def calc_PSNR(img1, img2):
        '''
        img1 and img2 have range [0, 255]
        '''
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * math.log10(255.0 / math.sqrt(mse))

if __name__ == '__main__':
    from config import config
    from network import RTA_VDN

    
    from utils.model_opr import load_model
    from utils.common import *
    import glob 



    model_path = config.INIT_MODEL

    model =  RTA_VDN(config)
    device = torch.device('cuda')
    # print('model',model.d0_align)
    model = model.to(device)
    load_model(model, model_path)
    import glob
    model.eval()
    print("RTA_VDN have {:.3f}M paramerters in total".format(sum(x.numel() for x in model.parameters())/1000000.0))

    dataPath = '/home/hailangwu/zk/data/denoise_test/derf_540p_seqs' 
    # 28.3988 0.804 200K
    categories = sorted(os.listdir(dataPath)) 
    print(categories,len(categories))
    # ['aerobatics', 'car-race', 'carousel', 'cats-car', 'chamaleon', 'deer', 'giant-slalom', 'girl-dog', 'golf', 'guitar-violin', 'gym', 'helicopter', 
    # 'horsejump-stick', 'hoverboard', 'lock', 'man-bike', 'monkeys-trees', 'mtb-race', 'orchid', 
    # 'people-sunset', 'planes-crossing', 'rollercoaster', 'salsa', 'seasnake', 'skate-jump', 'slackline', 'subway', 'tandem', 'tennis-vest', 'tractor']

    # ['hypersmooth', 'motorbike', 'park_joy', 'rafting', 'snowboard', 'sunflower', 'touchdown', 'tractor']
    sigma_list = [10,20,30,40,50]
    # sigma_list = [10,20,50]
    # sigma_list = [40]
    model_name = './output_{}'.format('full_c')
    # categories = ['touchdown']
    if os.path.exists(model_name) is False:
        os.mkdir(model_name)



    psnr_dict = {}
    for sigma in sigma_list:
        psnr_dict[sigma] = []
    for category in categories:
        gt_path = os.path.join(dataPath,category)
        files_gt = sorted(glob.glob(gt_path+'/*.png'))
        n = len(files_gt)
        model_name_path = '{}/{}'.format(model_name,category)
        # if os.path.exists(model_name_path) is False:
        #     os.mkdir(model_name_path)
        # else:
        #     continue
        
        data_file_path = '{}/{}/metric.txt'.format(model_name,category)
        psnr_folder_dict = {}
        for sigma in sigma_list:
            psnr_folder_dict[sigma] = []
        for i in range(n):
            if i > 84:
                break
            print('files_gt[i]',files_gt[i])
            gt_img = cv2.imread(files_gt[i])
            img_name = files_gt[i].split('/')[-1]

            idx_left_list = [None] * 2
            idx_right_list = [None] * 2
            for j in range(1,3):
                idx_left = max(i - j,0)
                idx_right = min(i+j,n-1)
                idx_left_list[2-j] = idx_left
                idx_right_list[j-1] = idx_right
            idx_list = idx_left_list + [i] + idx_right_list

            frame_lr = [cv2.imread(files_gt[idx_j]) for idx_j in idx_list]

            print(i,idx_list)
            
            hr_data = gt_img.transpose(2,0,1)
            

            # print(lr_data.shape)
            hr_tensor = torch.from_numpy(hr_data.astype(np.float32) / 255.0).float().unsqueeze(0)

            gt = hr_tensor.cpu().numpy()[0].astype(np.float32)
            gt = np.transpose(gt,(1,2,0))
            gt = gt[2:-2,2:-2]
            model_name_path = '{}/{}'.format(model_name,category)
            if os.path.exists(model_name_path) is False:
                os.mkdir(model_name_path)
                with open(data_file_path,'w') as f:
                    f.write('\n')

            for jdx,sigma_i in enumerate(sigma_list) :
                noise = np.random.normal(0, sigma_i, (5,3,hr_tensor.shape[-2], hr_tensor.shape[-1]))
                # noise = np.random.normal(0, sigma, (3,lr_data.shape[-2], lr_data.shape[-1]))

                noise_map = np.zeros((1,hr_tensor.shape[-2], hr_tensor.shape[-1]),dtype=float)
                noise_map[:] = sigma_i

                lr_data =  np.stack(frame_lr,0)
                lr_data = lr_data.transpose(0,3,1,2)
                lr_data = lr_data.astype(np.float)
                lr_data += noise
                lr_data = np.clip(lr_data,0,255.0)

                
                lr_tensor = torch.from_numpy(lr_data.astype(np.float32) / 255.0).float().unsqueeze(0)
                noise_map = torch.from_numpy(noise_map.astype(np.float32) / 255.0).float().unsqueeze(0)

                
                with torch.no_grad():
                    sr_vsr  = model(lr_tensor.to(device),noise_map.to(device))

            
            
                sr_vsr = sr_vsr.clamp(0,1)
                output_vsr = sr_vsr.detach().cpu().numpy()[0].astype(np.float32)    
                output_vsr = np.transpose(output_vsr,(1,2,0))

                output_vsr = output_vsr[2:-2,2:-2]
                

                psnr = calculate_psnr(output_vsr*255.0, gt*255.0)
                
                print(category,sigma_i,psnr)
                with open(data_file_path,'a') as f:
                    f.write('{} '.format(psnr))

                psnr_folder_dict[sigma_i].append(psnr)
                

            with open(data_file_path,'a') as f:
                f.write('\n')

        
        for k in psnr_folder_dict:
            psnr_dict[k].append(sum(psnr_folder_dict[k])/len(psnr_folder_dict[k]))

    for k in psnr_dict:
        tmp_psnr = sum(psnr_dict[k])/len(psnr_dict[k])
        print(k,len(psnr_dict[k]),tmp_psnr)