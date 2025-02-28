import torch 
from utils import viewimages, A, f
from metrics import ssim, lpips, psnr
from tqdm import tqdm 

def perform_pnp_pgd(x0,y,  D, fk, nu, likelihood=f, n_iter = 50, 
                    is_ddpm=False, relax=1, timestep=1,  titre=''):
    
    x = y.clone()

    tau = 1.9 * nu**2 
    strength = 2 * nu

    metrics = {
        'psnr': [psnr(x0, x)],
        'residual': [],
        'lpips': [lpips(x0, x)],
        'ssim': [ssim(x0, x)]
    }

    images = [x0, x]
    iters = [0]

    for it in tqdm(range(n_iter)):
        xpre = x.clone()
        x.requires_grad_(True)
        fx = likelihood(x, y, nu, fk)
        fx.backward()
        with torch.no_grad():
            x -= tau * x.grad
        x.grad.zero_()

        if not is_ddpm: 
            x = D(x, sigma=strength).detach()
        else:
            x = D.denoise_step(x, t=timestep, a=relax).detach()
            if it==5:
                print(torch.min(x), torch.max(x), torch.mean(x))

        metrics['psnr'].append(psnr(x0, x))
        metrics['residual'].append((torch.linalg.norm(x.detach() - xpre) / torch.linalg.norm(x0)).cpu())
        metrics['lpips'].append(lpips(x0, x))
        metrics['ssim'].append(ssim(x0, x))

        if (it) % (n_iter // 4) == (n_iter // 4)-1:
            images.append(x.clone())
            iters.append(it+1)
    viewimages(images, iters, titre=titre, psnr=metrics['psnr'])

    return x, metrics, images

def perform_pnp_pgd_stochastic(x0,y, D, fk,nu, likelihood = f,  n_iter = 50, 
                    noise=0.01, is_ddpm=False, relax=1, timestep=1,  titre=''):
    
    x = y.clone()

    tau = 1.9 * nu**2 
    strength = 2 * nu

    metrics = {
        'psnr': [psnr(x0, x)],
        'residual': [],
        'lpips': [lpips(x0, x)],
        'ssim': [ssim(x0, x)]
    }

    images = [x0, x]
    iters = [0]

    for it in tqdm(range(n_iter)):
        xpre = x.clone()
        x.requires_grad_(True)
        fx = likelihood(x, y, nu, fk)
        fx.backward()
        with torch.no_grad():
            x -= tau * x.grad
        x.grad.zero_()
        eps = noise * torch.randn_like(x)
        if not is_ddpm:
            x = D(x+eps, sigma=strength).detach()
        else:
            x = D.denoise_step(x+eps, t=timestep, a=relax).detach()


        metrics['psnr'].append(psnr(x0, x))
        metrics['residual'].append((torch.linalg.norm(x.detach() - xpre) / torch.linalg.norm(x0)).cpu())
        metrics['lpips'].append(lpips(x0, x))
        metrics['ssim'].append(ssim(x0, x))

        if (it) % (n_iter // 4) == (n_iter // 4)-1:
            images.append(x.clone())
            iters.append(it+1)

    viewimages(images, iters, titre=titre, psnr=metrics['psnr'])
    return x, metrics, images

def perform_annealed_snore(x0, D,fk, n_iter, nu, sigma_0, sigma_m1, alpha_0, alpha_m1, m, f=f , A= A,  titre=''):
    tau = 1.9 * nu**2  
    s = 2 * nu  

    sigma_schedule = torch.linspace(sigma_0, sigma_m1, m).tolist()
    alpha_schedule = torch.linspace(alpha_0, alpha_m1, m).tolist()

    y = A(x0, fk) + nu * torch.randn_like(x0)
    x = y.clone()


    metrics = {
        'psnr': [psnr(x0, x)],
        'residual': [],
        'lpips': [lpips(x0, x)],
        'ssim': [ssim(x0, x)]
    }

    images = [x0, x]
    iters = [0]

    for i in range(m):
        sigma = sigma_schedule[i]
        alpha = alpha_schedule[i]
        lmbda = alpha / sigma**2    

        for it in tqdm(range(n_iter // m), desc=f'Annealing Level {i+1}/{m} - sigma: {sigma:.4f}, alpha: {alpha:.4f}'):
            xpre = x.clone()
            x.requires_grad_(True)

            epsilon = torch.randn_like(x)
            x_tilde = x + sigma * epsilon
            fx = f(x, y, nu, fk)
            fx.backward()

            with torch.no_grad():
                x -= ( tau * x.grad + tau*lmbda * (x - D(x_tilde, sigma=2*sigma)) )

            x.grad.zero_()
            x = x.detach()
            # Store metrics
            metrics['psnr'].append(psnr(x0, x))
            metrics['residual'].append((torch.linalg.norm(x.detach() - xpre) / torch.linalg.norm(x0)).cpu())
            metrics['lpips'].append(lpips(x0, x))
            metrics['ssim'].append(ssim(x0, x))

        if (i + 1) % (m // 4) == 0:
            images.append(x.clone())
            iters.append(i)
    viewimages(images, iters, titre=titre, psnr=metrics['psnr'])

    return x, metrics, images

def estimate_lipschitz_constant(model, images, nu = 0.01, num_samples=10, is_ddpm=False, apply_blur = False, A= None, fk = None, t=1, a=1):
    lip_constants = []
    
    with torch.no_grad():
        for img in tqdm(images):
            img = img.clone()
            img = img.unsqueeze(0)

            if apply_blur:
                img = A(img, fk)
            ratios = []

            for _ in range(num_samples):
                noise = nu * torch.randn_like(img)
                noisy_img = img + noise
                if is_ddpm:
                    out_orig = model.denoise_step(img, t=t, a=a)
                    out_pert = model.denoise_step(noisy_img, t=t, a=a)
                else: 
                    out_orig = model(img, sigma = 2*nu)
                    out_pert = model(noisy_img, sigma = 2*nu)
                
                output_diff = torch.norm(out_pert - out_orig)
                input_diff = torch.norm(noise)
                ratio = output_diff / input_diff
                
                ratios.append(ratio.item())
            
            lip_constants.append(max(ratios))
    
    return lip_constants

def estimate_lipschitz_constant(model, images, nu = 0.01, num_samples=10, is_ddpm=False, apply_blur = False, A= None, fk = None, t=1, a=1):
    lip_constants = []
    
    with torch.no_grad():
        for img in tqdm(images):
            img = img.clone()
            img = img.unsqueeze(0)

            if apply_blur:
                img = A(img, fk)
            ratios = []

            for _ in range(num_samples):
                noise = nu * torch.randn_like(img)
                noisy_img = img + noise
                if is_ddpm:
                    out_orig = model.denoise_step(img, t=t, a=a)
                    out_pert = model.denoise_step(noisy_img, t=t, a=a)
                else: 
                    out_orig = model(img, sigma = 2*nu)
                    out_pert = model(noisy_img, sigma = 2*nu)
                
                output_diff = torch.norm(out_pert - out_orig)
                input_diff = torch.norm(noise)
                ratio = output_diff / input_diff
                
                ratios.append(ratio.item())
            
            lip_constants.append(max(ratios))
    
    return lip_constants

def estimate_lipschitz_constant_real(model, images, nu = 0.01, num_samples=10, is_ddpm=False, apply_blur = False, A= None, fk = None, t=1, a=1):
    lip_constants = []
    
    with torch.no_grad():
        for img in tqdm(images):
            img = img.clone()
            img = img.unsqueeze(0)
            if apply_blur:
                img = A(img, fk)
            ratios = []

            for _ in range(num_samples):
                noise = nu * torch.randn_like(img)
                noisy_img = img + noise
                if is_ddpm :
                    out_orig = model.denoise_step(img, t=t, a=a) - img
                    out_pert = model.denoise_step(noisy_img, t=t, a=a) - noisy_img
                else:
                    out_orig = model(img, sigma = 2*nu) - img
                    out_pert = model(noisy_img, sigma = 2*nu) - noisy_img
                
                output_diff = torch.norm(out_pert - out_orig)
                input_diff = torch.norm(noise)
                ratio = output_diff / input_diff
                
                ratios.append(ratio.item())
            
            lip_constants.append(max(ratios))
    
    return lip_constants

def deblurring(x0, x, model, nu, is_ddpm=False, t=1, a=1):
    with torch.no_grad():
        if is_ddpm:
            debl =  model.denoise_step(x, t=t, a=a)
        else:
            debl = model(x, sigma=2*nu)
    psnr_val = psnr(x0, debl)
    lpips_val = lpips(x0, debl)
    ssim_val = ssim(x0, debl)
    return debl, psnr_val, lpips_val, ssim_val


