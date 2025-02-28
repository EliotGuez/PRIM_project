#### import file ####
import tempfile
import IPython
from skimage.transform import rescale
import matplotlib.pyplot as plt
import numpy as np 
import torch
from torch.fft import fft2, ifft2
import os 



#### image processing functions ####
def rgb2gray(u):
    return 0.2989 * u[:,:,0] + 0.5870 * u[:,:,1] + 0.1140 * u[:,:,2]
def tensor2im(x):
    return x.detach().cpu().permute(2,3,1,0).squeeze().clip(0,1)
def str2(chars):
    return "{:.2f}".format(chars)


def viewimage(im, normalize=True,vmin=0,vmax=1,z=2,order=0,titre='',displayfilename=False):
    im = im.detach().cpu().permute(2,3,1,0).squeeze()
    imin= np.array(im).astype(np.float32)
    channel_axis = 2 if len(im.shape)>2 else None
    imin = rescale(imin, z, order=order, channel_axis=channel_axis)
    if normalize:
        if vmin is None:
            vmin = imin.min()
        if vmax is None:
            vmax = imin.max()
        imin-=vmin
        if np.abs(vmax-vmin)>1e-10:
            imin = (imin.clip(vmin,vmax)-vmin)/(vmax-vmin)
        else:
            imin = vmin
    else:
        imin=imin.clip(0,255)/255
    imin=(imin*255).astype(np.uint8)
    filename=tempfile.mktemp(titre+'.png')
    if displayfilename:
        print (filename)
    plt.imsave(filename, imin, cmap='gray')
    if titre!='':
        plt.savefig(f"results/{titre}.png", dpi='figure', bbox_inches='tight')
    IPython.display.display(IPython.display.Image(filename))

def viewimages(images, iter, normalize=True, vmin=0, vmax=1, z=2, order=0, titre='', psnr = None):

    fig, axes = plt.subplots(1, len(images), figsize=(18, 12))    
    for i, img in enumerate(images):
        img = img.detach().cpu().squeeze().permute(1, 2, 0).numpy()  # Move channels to last dim
        img = rescale(img, z, order=order, channel_axis=2 if img.ndim == 3 else None)
        
        if normalize:
            img = np.clip((img - vmin) / (vmax - vmin), 0, 1)
        
        axes[i].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[i].axis('off')
        if psnr is not None:
            if i ==0:
                axes[0].set_title(f"Original Image")
            else: 
                axes[i].set_title(f"PSNR: {psnr[iter[i-1]]:.2f}, Iter {iter[i-1]}")
        else : 
            axes[i].set_title(f"Iter {i}")
    plt.savefig(f"results/{titre}.png", dpi='figure', bbox_inches='tight')
    plt.show()

def A(x, fk): 
    return ifft2(fft2(x) * fk).real

def f(x, y, nu, fk): 
    return 1/ (2 *nu**2) * torch.sum((A(x, fk) - y)**2)

#### KERNEL AND DATASET LOADING ####
def load_kernel(kt,M,N,device):
    (m,n) = kt.shape
    k = torch.zeros((M,N),device=device)
    k[0:m,0:n] = kt/torch.sum(kt)
    k = torch.roll(k,(-int(m/2),-int(n/2)),(0,1))
    k = k[None,None,:,:]
    fk = fft2(k)
    return fk

### there are two size of images in csbb68 : $[321 \times 481]$ and $[481 \times 321]$.

def load_batch_images(directory, device, batch_size=10, is_cbsd68=False):
    image_tensors = []
    for i in range(batch_size):
        if not is_cbsd68:
            filename =f"{i:05d}.png"
        else: 
            filename = f"{i:04d}.png"
        filepath = os.path.join(directory, filename)
        img = torch.tensor(plt.imread(filepath), device = device).float()
        img_tensor = img.permute(2, 0, 1).unsqueeze(0)
        if is_cbsd68:
            # Crop the image from the center
            _, _, H, W = img_tensor.shape
            start_h = (H - 321) // 2
            start_w = (W - 321) // 2
            img_tensor = img_tensor[:, :, start_h:start_h+320, start_w:start_w+320]
        image_tensors.append(img_tensor)
    return torch.cat(image_tensors, dim=0)


#### PLOT FUNCTION ####

def save_metrics(metrics,  titre):
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    ax[0].plot(metrics['psnr'])
    ax[0].set_title("PSNR")
    ax[0].set_xlabel("Iterations")
    ax[0].set_ylabel("PSNR")

    ax[1].semilogy(metrics['residual'])
    ax[1].set_title("Residual")
    ax[1].set_xlabel("Iterations")
    ax[1].set_ylabel("Residual")

    ax[2].plot(metrics['ssim'])
    ax[2].set_title("SSIM")
    ax[2].set_xlabel("Iterations")
    ax[2].set_ylabel("SSIM")

    ax[3].plot(metrics['lpips'])
    ax[3].set_title("LPIPS")
    ax[3].set_xlabel("Iterations")
    ax[3].set_ylabel("LPIPS")
    
    plt.savefig(f"results/{titre}.png", dpi='figure', bbox_inches='tight')
    plt.show()

def save_psnr(psnr, titre):
    plt.plot(psnr)
    plt.title("PSNR")
    plt.xlabel("Iterations")
    plt.ylabel("PSNR")
    plt.savefig(f"results/{titre}.png", dpi='figure', bbox_inches='tight')
    plt.show()

def plot_lipschitz_comparison(lip_constants_1, lip_constants_2, model_name_1, model_name_2, title = ''):
    """Plot comparison of Lipschitz constants between models"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    # do a histogram for each model
    ax.hist(lip_constants_1, bins=20, alpha=0.5, label=model_name_1)
    ax.hist(lip_constants_2, bins=20, alpha=0.5, label=model_name_2)
    ax.set_title(f"Lipschitz Constant Comparison")
    ax.set_xlabel("Lipschitz Constant")
    ax.legend()
    plt.savefig(f"results/{title}.png", dpi='figure', bbox_inches='tight')
    plt.show()
