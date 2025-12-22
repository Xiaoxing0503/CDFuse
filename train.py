# -*- coding: utf-8 -*-

'''
------------------------------------------------------------------------------
Import packages
------------------------------------------------------------------------------
'''

from net1 import Restormer_Encoder, Restormer_Decoder, Restormer_FDecoder
from utils.dataset import H5Dataset
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  
import sys
import time
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.loss import Fusionloss, cc
import kornia



'''
------------------------------------------------------------------------------
Configure our network
------------------------------------------------------------------------------
'''


os.environ['CUDA_VISIBLE_DEVICES'] = '2'
criteria_fusion = Fusionloss()
model_str = 'CDFusion'

# . Set the hyper-parameters for training
num_epochs = 100 # total epoch
epoch_gap = 100  # epoches of Phase I 

lr = 1e-4
weight_decay = 0
batch_size = 4
GPU_number = os.environ['CUDA_VISIBLE_DEVICES']
# Coefficients of the loss function
coeff_mse_loss_VF = 1. # alpha1
coeff_mse_loss_IF = 1.
coeff_decomp = 2.      # alpha2 and alpha4
coeff_tv = 5.

clip_grad_norm_value = 0.01
optim_step = 20
optim_gamma = 0.5


# Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
DIDF_Encoder = nn.DataParallel(Restormer_Encoder()).to(device)
DIDF_Encoder2 = nn.DataParallel(Restormer_Encoder()).to(device)
DIDF_Decoder = nn.DataParallel(Restormer_Decoder()).to(device)
DIDF_Decoder2 = nn.DataParallel(Restormer_Decoder()).to(device)
DIDF_Decoder3 = nn.DataParallel(Restormer_FDecoder()).to(device)

# optimizer, scheduler and loss function
optimizer1 = torch.optim.Adam(
    DIDF_Encoder.parameters(), lr=lr, weight_decay=weight_decay)
optimizer2 = torch.optim.Adam(
    DIDF_Decoder.parameters(), lr=lr, weight_decay=weight_decay)
optimizer5 = torch.optim.Adam(
    DIDF_Encoder2.parameters(), lr=lr, weight_decay=weight_decay)
optimizer6 = torch.optim.Adam(
    DIDF_Decoder2.parameters(), lr=lr, weight_decay=weight_decay)
optimizer7 = torch.optim.Adam(
    DIDF_Decoder3.parameters(), lr=lr, weight_decay=weight_decay)

scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=optim_step, gamma=optim_gamma)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=optim_step, gamma=optim_gamma)
scheduler5 = torch.optim.lr_scheduler.StepLR(optimizer5, step_size=optim_step, gamma=optim_gamma)
scheduler6 = torch.optim.lr_scheduler.StepLR(optimizer6, step_size=optim_step, gamma=optim_gamma)
scheduler7 = torch.optim.lr_scheduler.StepLR(optimizer7, step_size=optim_step, gamma=optim_gamma)

MSELoss = nn.MSELoss()
# L1Loss = nn.L1Loss()
# Loss_ssim = kornia.losses.SSIM(11, reduction='mean')


# data loader
trainloader = DataLoader(H5Dataset(r"data/MIF_train_imgsize_128_stride_200.h5"),
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=0)

loader = {'train': trainloader, }
timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")

'''
------------------------------------------------------------------------------
Train
------------------------------------------------------------------------------
'''

step = 0
torch.backends.cudnn.benchmark = True
prev_time = time.time()

def poe_fusion(mu1, logvar1, mu2, logvar2, T=5.0, eps=1e-8):
    """
    mu1, logvar1, mu2, logvar2: [B, C, H, W]
    return: mu3, logvar3
    """
    var1 = torch.exp(logvar1) + eps
    var2 = torch.exp(logvar2) + eps

    T1, T2 = 1.0 / (var1*T), 1.0 / (var2*T)   # precision
    var3 = 1.0 / (T1 + T2)            # fused variance
    mu3 = (mu1 * T1 + mu2 * T2) * var3  # fused mean

    logvar3 = torch.log(var3 + eps)
    return mu3, logvar3

def kl_divergence(mu, logvar):
    """
    KL(N(mu, sigma^2) || N(0,I))
    mu, logvar: [B, C, H, W]
    return: scalar loss
    """
    kl_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    # 求所有样本的平均损失
    return torch.mean(kl_per_sample)



for epoch in range(num_epochs):
    ''' train '''
    for i, (data_VIS, data_IR) in enumerate(loader['train']):
        data_VIS, data_IR = data_VIS.cuda(), data_IR.cuda()
        DIDF_Encoder.train()
        DIDF_Decoder.train()
        DIDF_Encoder2.train()
        DIDF_Decoder2.train()
        DIDF_Decoder3.train()

        DIDF_Encoder.zero_grad()
        DIDF_Decoder.zero_grad()
        DIDF_Encoder2.zero_grad()
        DIDF_Decoder2.zero_grad()
        DIDF_Decoder3.zero_grad()

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer5.zero_grad()
        optimizer6.zero_grad()
        optimizer7.zero_grad()

        if epoch < epoch_gap: #Phase I
            feature_V_B, feature_V_D, _ = DIDF_Encoder(data_VIS)
            feature_I_B, feature_I_D, _ = DIDF_Encoder2(data_IR)

            means_hat_V_B, scales_hat_V_B = feature_V_B.chunk(2, 1)
            means_hat_V_D, scales_hat_V_D = feature_V_D.chunk(2, 1)
            means_hat_I_B, scales_hat_I_B = feature_I_B.chunk(2, 1)
            means_hat_I_D, scales_hat_I_D = feature_I_D.chunk(2, 1)

            means_hat_S, scales_hat_S = poe_fusion(means_hat_V_B, scales_hat_V_B, means_hat_I_B, scales_hat_I_B)
            # means_hat_S = (means_hat_V_B + means_hat_I_B)/2
            # scales_hat_S = (scales_hat_V_B + scales_hat_I_B)/2

            data_VIS_hat, _ = DIDF_Decoder(means_hat_S, scales_hat_S, means_hat_V_D, scales_hat_V_D)
            data_IR_hat, _ = DIDF_Decoder2(means_hat_S, scales_hat_S, means_hat_I_D, scales_hat_I_D)
            Fused = DIDF_Decoder3(means_hat_S, scales_hat_S, means_hat_V_D, scales_hat_V_D, means_hat_I_D, scales_hat_I_D)

            mse_loss_V = MSELoss(data_VIS, data_VIS_hat)
            mse_loss_I = MSELoss(data_IR, data_IR_hat)

            means_hat = torch.cat((means_hat_V_D, means_hat_I_D, means_hat_S), dim=1)
            scales_hat = torch.cat((scales_hat_V_D, scales_hat_I_D, scales_hat_S), dim=1)
            logvar = torch.log(scales_hat.pow(2) + 1e-8)
            kl_loss = kl_divergence(means_hat, logvar) * 1e-2
            # cc_loss = abs(cc(feature_V_D, feature_I_D)) * 100
            fusionloss = criteria_fusion(data_VIS, data_IR, Fused)

            re_loss = (mse_loss_V + mse_loss_I) * 100
            loss = re_loss + fusionloss + kl_loss
            loss.backward()
            nn.utils.clip_grad_norm_(
                DIDF_Encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                DIDF_Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                DIDF_Encoder2.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                DIDF_Decoder2.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                DIDF_Decoder3.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            optimizer1.step()  
            optimizer2.step()
            optimizer5.step()  
            optimizer6.step()
            optimizer7.step()


        # Determine approximate time left
        batches_done = epoch * len(loader['train']) + i
        batches_left = num_epochs * len(loader['train']) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [loss: %f][re_loss: %f][kl_loss: %f][fusionloss: %f] ETA: %.10s"
            % (
                epoch,
                num_epochs,
                i,
                len(loader['train']),
                loss.item(),
                re_loss.item(),
                # cc_loss.item(),
                kl_loss.item(),
                fusionloss.item(),
                time_left,
            )
        )

    # adjust the learning rate

    scheduler1.step()  
    scheduler2.step()
    scheduler5.step()  
    scheduler6.step()
    scheduler7.step()  
    # if not epoch < epoch_gap:
    #     scheduler3.step()
    #     scheduler4.step()

    if optimizer1.param_groups[0]['lr'] <= 1e-6:
        optimizer1.param_groups[0]['lr'] = 1e-6
    if optimizer2.param_groups[0]['lr'] <= 1e-6:
        optimizer2.param_groups[0]['lr'] = 1e-6
    if optimizer5.param_groups[0]['lr'] <= 1e-6:
        optimizer5.param_groups[0]['lr'] = 1e-6
    if optimizer6.param_groups[0]['lr'] <= 1e-6:
        optimizer6.param_groups[0]['lr'] = 1e-6
    if optimizer7.param_groups[0]['lr'] <= 1e-6:
        optimizer7.param_groups[0]['lr'] = 1e-6
    
if True:
    checkpoint = {
        'DIDF_Encoder': DIDF_Encoder.state_dict(),
        'DIDF_Decoder': DIDF_Decoder.state_dict(),
        'DIDF_Encoder2': DIDF_Encoder2.state_dict(),
        'DIDF_Decoder2': DIDF_Decoder2.state_dict(),
        'DIDF_Encoder3': DIDF_Decoder3.state_dict()
    }
    torch.save(checkpoint, os.path.join("models/CDFusion_"+timestamp+'.pth'))


