import sys 
sys.path.append('/home/hailangwu/zk/project/vsr/')

import torch
import torch.nn as nn
import functools
import torch.nn.functional as F
import torch.nn.init as init

torch.backends.cudnn.enabled = False


import math 

from libs.DCNv2_latest.dcn_v2 import *
class DCN_sep(DCNv2):
    '''Use other features to generate offsets and masks'''

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1,
                 deformable_groups=1):
        super(DCN_sep, self).__init__(in_channels, out_channels, kernel_size, stride, padding,
                                      dilation, deformable_groups)

        channels_ = self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1]
        self.conv_offset_mask = nn.Conv2d(self.in_channels, channels_, kernel_size=self.kernel_size,
                                          stride=self.stride, padding=self.padding, bias=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, input, fea):
        '''input: input features for deformable conv
        fea: other features used for generating offsets and mask'''
        out = self.conv_offset_mask(fea)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)

        # offset_mean = torch.mean(torch.abs(offset))
        # if offset_mean > 100:
        #     logger.warning('Offset mean is {}, larger than 100.'.format(offset_mean))

        mask = torch.sigmoid(mask)
        return dcn_v2_conv(input, offset, mask, self.weight, self.bias, self.stride, self.padding,
                           self.dilation, self.deformable_groups)

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())

    return nn.Sequential(*layers)

class LocalCorr(torch.nn.Module):
    def __init__(self,nf,nbr_size = 3,alpha=-1.0):
        super(LocalCorr,self).__init__()
        self.nbr_size = nbr_size
        self.alpha = alpha
        pass 
    def forward(self,nbr_list,ref):
        mean = torch.stack(nbr_list,1).mean(1).detach().clone()
        # print(mean.shape)
        b,c,h,w = ref.size()
        ref_clone = ref.detach().clone()
        ref_flat = ref_clone.view(b,c,-1,h*w).permute(0,3,2,1).contiguous().view(b*h*w,-1,c)
        ref_flat = torch.nn.functional.normalize(ref_flat,p=2,dim=-1)
        pad = self.nbr_size // 2
        afea_list = []
        for i in range(len(nbr_list)):
            nbr = nbr_list[i]
            weight_diff = (nbr - mean)**2
            weight_diff = torch.exp(self.alpha*weight_diff)
            
            nbr_pad = torch.nn.functional.pad(nbr,(pad,pad,pad,pad),mode='reflect')
            nbr = torch.nn.functional.unfold(nbr_pad,kernel_size=self.nbr_size).view(b,c,-1,h*w)
            nbr = torch.nn.functional.normalize(nbr,p=2,dim=1)
            nbr = nbr.permute(0,3,1,2).contiguous().view(b*h*w,c,-1)
            d = torch.matmul(ref_flat,nbr).squeeze(1)
            weight_temporal = torch.nn.functional.softmax(d,-1)
            agg_fea = torch.einsum('bc,bnc->bn',weight_temporal,nbr).view(b,h,w,c).contiguous().permute(0,3,1,2)

            agg_fea = agg_fea * weight_diff
            
            afea_list.append(agg_fea)
        al_fea = torch.stack(afea_list+[ref],1)
        return al_fea

class ResidualBlock_noBN(nn.Module):
    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, 90, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(90, nf, 3, 1, 1, bias=True)
        self.scale = torch.nn.Parameter(torch.FloatTensor([0.5]))
    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=False)
        out = self.conv2(out)
        return identity + self.scale * out




class HR_Align(nn.Module):
    def __init__(self, nf=64, groups=16):
        super(HR_Align, self).__init__()
        self.offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.joint_combine = nn.Conv2d(nf*2, nf, 3, 1, 1, bias=True)
        RB_f = functools.partial(ResidualBlock_noBN, nf=nf)
        self.rbs = make_layer(RB_f, 8)

        self.dcnpack = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1,
                                  deformable_groups=groups)

        self.scaleing = torch.nn.Sequential(
            nn.Conv2d(nf*2, nf, 3, 1, 1, bias=True),
            torch.nn.Sigmoid(),
        )

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=False)
    def forward(self,nbr_fea,ref_fea,pre_offset_fea=None):
        _offset = torch.cat([nbr_fea,ref_fea], dim=1)
        _offset = self.lrelu(self.offset_conv1(_offset))
        _offset = self.lrelu(self.offset_conv2(_offset))

        if pre_offset_fea is None:
            offset_fea = torch.cat([_offset,_offset],1)
        else:
            offset_fea_init = torch.cat([_offset,pre_offset_fea],1)
            pre_offset_fea = pre_offset_fea * self.scaleing(offset_fea_init)
            offset_fea = torch.cat([_offset,pre_offset_fea],1) 
        offset = self.rbs(self.joint_combine(offset_fea))
        # align_fea = self.lrelu(self.hr_align(nbr_fea, offset,self.sep_module))
        align_fea = self.lrelu(self.dcnpack(nbr_fea, offset))
        return align_fea,offset


class easy_fuse(nn.Module):
    def __init__(self, nf=64, nframes=3, has_relu=True):
        super(easy_fuse, self).__init__()
        self.has_relu = has_relu
        self.fea_fusion = nn.Conv2d(nf * nframes, nf, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=False)

    def forward(self, aligned_fea):
        B, N, C, H, W = aligned_fea.size()
        aligned_fea = aligned_fea.view(B, N * C, H, W)
        out = self.fea_fusion(aligned_fea)

        if self.has_relu:
            out = self.lrelu(out)

        return out


class encoder(nn.Module):
    def __init__(self, nf=64, N_RB=5):
        super(encoder, self).__init__()
        RB_f = functools.partial(ResidualBlock_noBN, nf=nf)
        
        self.conv_first = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
        self.rbs = make_layer(RB_f, N_RB)

        self.d2_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.d2_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.d4_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.d4_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.py_conv = nn.Conv2d(nf*3, nf, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=False)

    def forward(self, x):
        fea = self.lrelu(self.conv_first(x))
        fea_lr = self.rbs(fea)

        fea_d2 = self.lrelu(self.d2_conv2(self.lrelu(self.d2_conv1(fea_lr))))
        fea_d4 = self.lrelu(self.d4_conv2(self.lrelu(self.d4_conv1(fea_d2))))

        fea_d2 = F.interpolate(fea_d2, size =(x.size()[-2],x.size()[-1]) , mode='bilinear', align_corners=False)
        fea_d4 = F.interpolate(fea_d4, size =(x.size()[-2],x.size()[-1]) , mode='bilinear', align_corners=False)

        out = self.lrelu(self.py_conv(torch.cat([fea_lr, fea_d2, fea_d4],1)))
        return out

class Motion_FeaFusion(nn.Module):
    def __init__(self, nf=64):
        super(Motion_FeaFusion, self).__init__()
        self.scaleing = torch.nn.Sequential(
            nn.Conv2d(nf*2, nf, 3, 1, 1, bias=True),
            torch.nn.Sigmoid(),
        )
        self.conv_out = nn.Conv2d(nf*2, nf, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=False)
    def forward(self,m0,m1):
        m_init = torch.cat([m0,m1],1)
        weighting = self.scaleing(m_init)
        # print('we',weighting.shape,m0.shape,m1.shape)
        mf = torch.cat([weighting*m0,(1.0-weighting)*m1],1)
        return self.lrelu( self.conv_out(mf) )


class RTA_VSR(nn.Module):
    def __init__(self, cfg=None):
        super(RTA_VSR, self).__init__()

        self.nbr = cfg.DATASET.NFRAME //2
        self.nframes = 2 * self.nbr + 1
        front_RB = cfg.MODEL.EXT_BLOCKS 
        back_RB = cfg.MODEL.RECONS_BLOCKS
        nf = cfg.MODEL.N_CHANNEL
        self.interpolation = cfg.MODEL.INTERPOLATION

        RB_f = functools.partial(ResidualBlock_noBN, nf=nf)

        self.fea_extract = encoder(nf=nf, N_RB=front_RB)


        self.hr_align_l = HR_Align(nf=nf)

        self.pre_motion_fusion = Motion_FeaFusion(nf)

        self.motion_fusion = Motion_FeaFusion(nf)

        self.fuse = easy_fuse(nf=nf, nframes=self.nframes)


        self.recon = make_layer(RB_f, back_RB)
        self.up_conv1 = nn.Conv2d(nf, 64 * 4, 3, 1, 1, bias=True)
        self.up_conv2 = nn.Conv2d(64, 64 * 4, 3, 1, 1, bias=True)
        self.hr_conv = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.out_conv = nn.Conv2d(64, 3, 1, 1, bias=True)
        self.ps = nn.PixelShuffle(upscale_factor=2)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1)

        
        self.localCorr = LocalCorr(nf=nf)
        
        self._initialize_weights()

    def _initialize_weights(self, scale=0.1):
        # for residual block
        for M in [self.fea_extract.rbs, self.recon]:
            for m in M.modules():
                if isinstance(m, nn.Conv2d):
                    init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                    m.weight.data *= scale
                    if m.bias is not None:
                        m.bias.data.zero_()

    def forward(self, x,gt = None):

        B, N, C, H, W = x.size()
        nbr_l = []

        for i in range(self.nbr * 2 + 1):
            if i != self.nbr:
                nbr_l.append(self.fea_extract(x[:, i, ...].contiguous()))
            else:
                ref_fea = self.fea_extract(x[:, i, ...].contiguous())

        base = x[:, self.nbr].contiguous()

        f_2_a,offset_2_a = self.hr_align_l(nbr_l[self.nbr-1],ref_fea)
        f_3_a,offset_3_a = self.hr_align_l(nbr_l[self.nbr],  ref_fea)

        f_10_a,offset_10_a = self.hr_align_l(nbr_l[self.nbr-2],\
                        nbr_l[self.nbr-1])
        f_11_a,offset_11_a = self.hr_align_l(f_10_a,ref_fea,offset_2_a)

        f_40_a,offset_40_a = self.hr_align_l(nbr_l[self.nbr+1],\
                        nbr_l[self.nbr])
        f_41_a,offset_41_a = self.hr_align_l(f_40_a,\
                        ref_fea,offset_3_a)
        
        f_00_a,offset_00_a = self.hr_align_l(nbr_l[self.nbr-3],\
                        nbr_l[self.nbr-2])
        f_01_a,offset_01_a = self.hr_align_l(f_00_a,\
                        nbr_l[self.nbr-1],offset_10_a)
        f_02_a,offset_02_a = self.hr_align_l(f_01_a,\
                        ref_fea,self.pre_motion_fusion(offset_2_a,offset_11_a))
        
        f_50_a,offset_50_a = self.hr_align_l(nbr_l[self.nbr+2],\
                        nbr_l[self.nbr+1])
        f_51_a,offset_51_a = self.hr_align_l(f_50_a,\
                        nbr_l[self.nbr],offset_40_a)
        f_52_a,offset_52_a = self.hr_align_l(f_51_a,\
                        ref_fea,self.motion_fusion(offset_3_a,offset_41_a))

        
        
        al_fea = self.localCorr([f_52_a,f_02_a,f_41_a,f_11_a,f_3_a,f_2_a],ref_fea)
    
        
        fuse_fea = self.fuse(al_fea)

        
        recon_fea = self.recon(fuse_fea)
        mr_fea = self.lrelu(self.ps(self.up_conv1(recon_fea)))
        hr_fea = self.lrelu(self.ps(self.up_conv2(mr_fea)))
        out = self.out_conv(self.lrelu(self.hr_conv(hr_fea)))
        base = F.interpolate(base, scale_factor=4, mode=self.interpolation, align_corners=False)
        out += base

        return out
            
    


if __name__ == '__main__':
    from config import config
    import time

    torch.cuda.set_device(0)
    device = torch.device('cuda')
    net = RTA_VSR(config)
    net.eval()
    net.to(device)
    print("RTA_VSR(vimeo) have {:.3f}M paramerters in total".format(sum(x.numel() for x in net.parameters())/1000000.0))
    input = torch.rand((1, 7, 3, 64, 64))
    input = input.to(device)

    with torch.no_grad():
        out = net(input)[0]
    print(out.shape)
