import torch
import numpy as np

from pytorch_msssim import ssim as compute_ssim
import lpips as lpips_lib

#### metric functions ####

loss_fn_lpips = lpips_lib.LPIPS(net='alex')
if torch.cuda.is_available():
    loss_fn_lpips = loss_fn_lpips.cuda()

def psnr(uref,ut,M=1):
    rmse = np.sqrt(np.mean((uref.cpu().numpy()-ut.cpu().numpy())**2))
    return 20*np.log10(M/rmse)

def lpips(img1, img2):
    img1 = img1.clamp(0, 1)
    img2 = img2.clamp(0, 1)   

    with torch.no_grad():
        d = loss_fn_lpips(img1, img2)
    return d.item()

def ssim(img1, img2):
    img1 = img1.clamp(0, 1)
    img2 = img2.clamp(0, 1)
    with torch.no_grad():
        ssim_val = compute_ssim(img1, img2, data_range=1.0)
    
    return ssim_val.item()

def mypsnr(x,y):
  error = torch.mean((x-y)**2).item()
  psnr = 10*np.log10(2**2/error)
  return(psnr)