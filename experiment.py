import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm 
import json

import deepinv as dinv
from tp6.guided_diffusion.unet import create_model
ffhq_model_path = 'tp6/ffhq_10m.pt'

import warnings
warnings.filterwarnings("ignore")
from models_diffusion import DDPMPNP
from utils import viewimage, A, f
import perform_algorithm as pfa
import utils
from metrics import psnr, ssim, lpips
from skimage.transform import rescale
import tempfile


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_config = {'image_size': 256,
                'num_channels': 128,
                'num_res_blocks': 1,
                'channel_mult': '',
                'learn_sigma': True,
                'class_cond': False,
                'use_checkpoint': False,
                'attention_resolutions': 16,
                'num_heads': 4,
                'num_head_channels': 64,
                'num_heads_upsample': -1,
                'use_scale_shift_norm': True,
                'dropout': 0.0,
                'resblock_updown': True,
                'use_fp16': False,
                'use_new_attention_order': False,
                'model_path': ffhq_model_path}

def viewimage(im, normalize=True, vmin=0, vmax=1, z=2, order=0, titre='', save_path=None):
    if save_path:
        os.makedirs(save_path, exist_ok=True)
    
    im = im.detach().cpu().permute(2,3,1,0).squeeze()
    imin = np.array(im).astype(np.float32)
    
    channel_axis = 2 if len(im.shape)>2 else None
    imin = rescale(imin, z, order=order, channel_axis=channel_axis)
    
    if normalize:
        if vmin is None:
            vmin = imin.min()
        if vmax is None:
            vmax = imin.max()
        imin -= vmin
        if np.abs(vmax-vmin)>1e-10:
            imin = (imin.clip(vmin,vmax)-vmin)/(vmax-vmin)
        else:
            imin = vmin
    else:
        imin = imin.clip(0,255)/255
    
    imin = (imin*255).astype(np.uint8)
    if save_path:
        filename = os.path.join(save_path, f"{titre}.png")
        plt.imsave(filename, imin, cmap='gray')
        print(f"Image saved to {filename}")

def plot_metrics_comparison(results_dir='restoration_results'):
    metrics_methods = [
        ('PnP-PGD DRUnet', metrics_pnp_drunet),
        ('PnP-PGD DDPM', metrics_pnp_ddpm),
        ('SNORE DRUnet', metrics_snore_drunet),
        ('SNORE DDPM', metrics_snore_ddpm),
        ('Annealed SNORE DRUnet', metrics_ann_snore_drunet),
        ('Annealed SNORE DDPM', metrics_ann_snore_ddpm)
    ]
    
    colors = [
        'blue', 'navy', 
        'green', 'darkgreen', 
        'red', 'darkred'
    ]
    
    metrics_to_plot = ['psnr', 'ssim', 'lpips']
    
    for metric in metrics_to_plot:
        plt.figure(figsize=(12, 6))
        
        for (name, method_metrics), color in zip(metrics_methods, colors):
            if metric in method_metrics:
                plt.plot(method_metrics[metric], label=name, color=color)
        
        plt.title(f'{metric.upper()} Comparison Across Restoration Methods')
        plt.xlabel('Iterations')
        plt.ylabel(metric.upper())
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.savefig(os.path.join(results_dir, f'{metric}_comparison.png'))
        plt.close()

    print("Metrics comparison plots have been saved in the restoration_results directory.")

if __name__ == '__main__':

    results_dir = 'restoration_results'
    os.makedirs(results_dir, exist_ok=True)

    model = create_model(**model_config)
    model = model.to(device)
    model.eval()

    # choose images
    directory = 'C:\\Users\\eliot\\OneDrive\\Documents\\3A\\PRIM\\tp mva\\tp6\\ffhq256-1k-validation'
    image_path = f"{directory}\\00010.png"
    x0 = torch.tensor(plt.imread(image_path), device=device).permute(2, 0, 1).unsqueeze(0)
    C, M, N = x0.shape[1:]

    # choose blur kernel
    kt = torch.tensor(np.loadtxt('tp8/kernels/levin4.txt'))
    fk = utils.load_kernel(kt, M, N, device)

    # choose noise level
    drunet = dinv.models.DRUNet(pretrained='tp8/ckpts/drunet_color.pth').to(device)
    DDPM = DDPMPNP(model)
    nu = .01

    # choose the number of noise instance to apply
    instance_noise = 1

    # choose n_iter to perform the algorithm, I choose 800 to have artifacts and need to be divided by 8 to have the same parameter than SNORE paper
    n_iter = 800

    for k in range(instance_noise):
        y = A(x0, fk) + nu * torch.randn_like(x0)
        # torch.save(y, f'restoration_results/noisy_image_{k}.pt')
        viewimage(y, titre=f'noisy_image_{k}', save_path=results_dir)


        # pnp_pgd drunet ddpm
        print('Perform PnP-PGD with DRUnet and DDPM...')

        x_pnp_drunet, metrics_pnp_drunet, _ = pfa.perform_pnp_pgd(x0, y, drunet, fk, nu, n_iter=n_iter,titre='')
        # torch.save(x_pnp_drunet, f'restoration_results/restored_image_{k}.pt')
        print(f"PSNR: {psnr(x0, x_pnp_drunet):.2f}, SSIM: {ssim(x0, x_pnp_drunet):.2f}, LPIPS: {lpips(x0, x_pnp_drunet):.2f}")
        viewimage(x_pnp_drunet, titre=f'restored_pnp_pgd_drunet_{k}', save_path=results_dir)

        
        x_pnp_ddpm, metrics_pnp_ddpm, _ = pfa.perform_pnp_pgd(x0, y, DDPM, fk, nu, n_iter=n_iter, is_ddpm=True, titre='')
        # torch.save(x_pnp_ddpm, f'restoration_results/restored_image_{k}.pt')
        print(f"PSNR: {psnr(x0, x_pnp_ddpm):.2f}, SSIM: {ssim(x0, x_pnp_ddpm):.2f}, LPIPS: {lpips(x0, x_pnp_ddpm):.2f}")
        viewimage(x_pnp_ddpm, titre=f'restored_pnp_pgd_ddpm_{k}', save_path=results_dir)
  
        #### snore drunet ddpm
        print('Perform SNORE with DRUnet and DDPM...')


        x_snore_drunet, metrics_snore_drunet= pfa.perform_snore(x0, y, drunet, fk, nu, n_iter=n_iter)
        # torch.save(x_snore_drunet, f'restoration_results/restored_image_{k}.pt')
        print(f"PSNR: {psnr(x0, x_snore_drunet):.2f}, SSIM: {ssim(x0, x_snore_drunet):.2f}, LPIPS: {lpips(x0, x_snore_drunet):.2f}")
        viewimage(x_snore_drunet, titre=f'restored_snore_drunet_{k}', save_path=results_dir)

        
        
        x_snore_ddpm, metrics_snore_ddpm = pfa.perform_snore(x0, y, DDPM, fk, nu, n_iter=n_iter, is_ddpm=True)
        # torch.save(x_snore_ddpm, f'restoration_results/restored_image_{k}.pt')
        print(f"PSNR: {psnr(x0, x_snore_ddpm):.2f}, SSIM: {ssim(x0, x_snore_ddpm):.2f}, LPIPS: {lpips(x0, x_snore_ddpm):.2f}")
        viewimage(x_snore_drunet, titre=f'restored_snore_drunet_{k}', save_path=results_dir)

        
        #### annealed snore drunet ddpm
        m = 16  # Number of annealing levels
        sigma_0 = 1.8 * nu  # Initial noise level
        sigma_m1 = 0.5 * nu  # Final noise level
        alpha_0 = 0.1 * sigma_0**2 / nu**2  # Initial alpha
        alpha_m1 = sigma_m1**2 / nu**2 # Final alpha


        print('Perform Annealed SNORE with DRUnet and DDPM...')
        x_ann_snore_drunet, metrics_ann_snore_drunet = pfa.perform_annealed_snore(x0, y, drunet, fk, nu, 
                                                                             n_iter = n_iter,sigma_0 = sigma_0, 
                                                                             sigma_m1=sigma_m1, alpha_0=alpha_0, 
                                                                             alpha_m1=alpha_m1, m=m)
        # torch.save(x_ann_snore_drunet, f'restoration_results/restored_image_{k}.pt')
        print(f"PSNR: {psnr(x0, x_ann_snore_drunet):.2f}, SSIM: {ssim(x0, x_ann_snore_drunet):.2f}, LPIPS: {lpips(x0, x_ann_snore_drunet):.2f}")
        viewimage(x_ann_snore_drunet, titre=f'restored_annealed_snore_drunet_{k}', save_path=results_dir)

        
         
        x_ann_snore_ddpm, metrics_ann_snore_ddpm = pfa.perform_annealed_snore(x0, y, DDPM, fk, nu,
                                                                       n_iter = n_iter, sigma_0 = sigma_0, 
                                                                       sigma_m1=sigma_m1, alpha_0=alpha_0, 
                                                                       alpha_m1=alpha_m1, m=m, is_ddpm=True)
        # torch.save(x_ann_snore_ddpm, f'restoration_results/restored_image_{k}.pt')
        print(f"PSNR: {psnr(x0, x_ann_snore_ddpm):.2f}, SSIM: {ssim(x0, x_ann_snore_ddpm):.2f}, LPIPS: {lpips(x0, x_ann_snore_ddpm):.2f}")
        viewimage(x_ann_snore_ddpm, titre=f'restored_annealed_snore_ddpm_{k}', save_path=results_dir)
        
        
        plot_metrics_comparison()


    print("Restoration completed. Results and metrics saved in restoration_results directory.")
