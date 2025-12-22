import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import numpy as np
from utils.Evaluator import Evaluator
import torch
import torch.nn as nn
from net1 import Restormer_Encoder, Restormer_Decoder, Restormer_FDecoder
from utils.img_read_save import img_save,image_read_cv2
import torchvision.transforms.functional as TF
import warnings
import logging
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL) 


def save_image(data_Fuse,img_name,test_out_decode):
    data_Fuse=(data_Fuse-torch.min(data_Fuse))/(torch.max(data_Fuse)-torch.min(data_Fuse))
    fi = np.squeeze((data_Fuse * 255).cpu().numpy())
    fi = fi.astype(np.uint8)
    img_save(fi, img_name.split(sep='.')[0], test_out_decode)

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

def pad_to_8x(image):
    """
    将图像pad到长和宽均为8的倍数
    
    参数:
        image: 输入图像，形状为[B, C, H, W]
    返回:
        padded_image: pad后的图像
        pad_h: 高度方向的pad量 (top, bottom)
        pad_w: 宽度方向的pad量 (left, right)
    """
    # 获取图像尺寸 (假设输入形状为[B, C, H, W])
    B, C, H, W = image.shape
    
    # 计算需要pad到8的倍数的量
    pad_h = (8 - H % 8) % 8  # 若已为8的倍数则pad 0
    pad_w = (8 - W % 8) % 8
    
    # 分配pad量（上下、左右平均分配，若为奇数则底部/右侧多1）
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    
    # 使用边缘值进行pad（可根据需求改为0填充等）
    padded_image = np.pad(
        image, 
        pad_width=((0, 0), (0, 0), (top, bottom), (left, right)),
        mode='edge'  # 边缘填充，保持图像连续性
    )
    
    return padded_image, (top, bottom), (left, right)

def crop_to_original(image, pad_h, pad_w):
    """
    将pad后的图像crop回原尺寸
    
    参数:
        image: pad后的图像
        pad_h: 高度方向的pad量 (top, bottom)
        pad_w: 宽度方向的pad量 (left, right)
    返回:
        cropped_image: 裁剪后的原尺寸图像
    """
    top, bottom = pad_h
    left, right = pad_w
    
    # 计算裁剪区域
    h_start = top
    h_end = image.shape[2] - bottom
    w_start = left
    w_end = image.shape[3] - right
    
    # 裁剪图像
    cropped_image = image[:, :, h_start:h_end, w_start:w_end]
    
    return cropped_image


ckpt_path=r"/data1/xml/CDFusion/models/CDFusion_10-18-18-20.pth"
for dataset_name in ["MSRS"]:
    print("\n"*2+"="*80)
    model_name="CDFusion"
    print("The test result of "+dataset_name+' :')
    test_folder=os.path.join('/data1/xml/Detection/ultralytics-main/data/detection/',dataset_name) 
    test_out_folder='/data1/xml/Detection/ultralytics-main/data/detection/proposed1018_gray/'

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device1 = torch.device("cuda:0")
    device2 = torch.device("cuda:0")

    Encoder = nn.DataParallel(Restormer_Encoder()).to(device1)
    Encoder2 = nn.DataParallel(Restormer_Encoder()).to(device1)
    Decoder = nn.DataParallel(Restormer_Decoder(training=False)).to(device2)
    Decoder2 = nn.DataParallel(Restormer_Decoder(training=False)).to(device2)
    Decoder3 = nn.DataParallel(Restormer_FDecoder(training=False)).to(device2)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    Encoder.load_state_dict(ckpt['DIDF_Encoder'])
    Decoder.load_state_dict(ckpt['DIDF_Decoder'])
    Encoder2.load_state_dict(ckpt['DIDF_Encoder2'])
    Decoder2.load_state_dict(ckpt['DIDF_Decoder2'])
    Decoder3.load_state_dict(ckpt['DIDF_Encoder3'])
    Encoder.eval()
    Decoder.eval()
    Encoder2.eval()
    Decoder2.eval()
    Decoder3.eval()

    with torch.no_grad():
        for img_name in os.listdir(os.path.join(test_folder,"ir")):
            data_IR=image_read_cv2(os.path.join(test_folder,"ir",img_name),mode='GRAY')[np.newaxis,np.newaxis, ...]/255.0
            data_VIS = image_read_cv2(os.path.join(test_folder,"vi",img_name), mode='GRAY')[np.newaxis,np.newaxis, ...]/255.0

            data_IR,data_VIS = torch.FloatTensor(data_IR),torch.FloatTensor(data_VIS)
            data_VIS, data_IR = data_VIS.cuda(), data_IR.cuda()

            feature_V_B, feature_V_D, _ = Encoder(data_VIS)
            feature_I_B, feature_I_D, _ = Encoder2(data_IR)

            means_hat_V_B, scales_hat_V_B = feature_V_B.chunk(2, 1)
            means_hat_V_D, scales_hat_V_D = feature_V_D.chunk(2, 1)
            means_hat_I_B, scales_hat_I_B = feature_I_B.chunk(2, 1)
            means_hat_I_D, scales_hat_I_D = feature_I_D.chunk(2, 1)

            means_hat_S, scales_hat_S = poe_fusion(means_hat_V_B, scales_hat_V_B, means_hat_I_B, scales_hat_I_B)
            # means_hat_S = (means_hat_V_B + means_hat_I_B)/2
            # scales_hat_S = (scales_hat_V_B + scales_hat_I_B)/2
            
            means_hat_S, scales_hat_S = means_hat_S.to(device2), scales_hat_S.to(device2)
            means_hat_V_D, scales_hat_V_D = means_hat_S.to(device2), scales_hat_S.to(device2)
            means_hat_I_D, scales_hat_I_D = means_hat_S.to(device2), scales_hat_S.to(device2)


            data_VIS_hat, _ = Decoder(means_hat_S, scales_hat_S, means_hat_V_D, scales_hat_V_D)
            data_IR_hat, _ = Decoder2(means_hat_S, scales_hat_S, means_hat_I_D, scales_hat_I_D)
            Fused = Decoder3(means_hat_S, scales_hat_S, means_hat_V_D, scales_hat_V_D, means_hat_I_D, scales_hat_I_D)

            save_image(Fused,img_name,test_out_folder)
            print(f"{img_name}")
