import torch 
from utils import viewimages, A, f, viewimages_2, viewimage
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

        metrics['psnr'].append(psnr(x0, x))
        metrics['residual'].append((torch.linalg.norm(x.detach() - xpre) / torch.linalg.norm(x0)).cpu())
        metrics['lpips'].append(lpips(x0, x))
        metrics['ssim'].append(ssim(x0, x))

        if (it) % (n_iter // 4) == (n_iter // 4)-1:
            images.append(x.clone())
            iters.append(it+1)
    # viewimages(images, iters, titre=titre, psnr=metrics['psnr'])

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

def perform_snore(x0,y,  D, fk, nu, likelihood=f, n_iter = 50, is_ddpm=False):
    
    x = y.clone()
    alpha = nu
    tau = 1.4*nu**2

    metrics = {
        'psnr': [psnr(x0, x)],
        'residual': [],
        'lpips': [lpips(x0, x)],
        'ssim': [ssim(x0, x)]
    }

    for it in tqdm(range(n_iter)):
        lmbda = alpha / nu**2
        xpre = x.clone()
        x.requires_grad_(True)
        x_tilde = x + nu * torch.randn_like(x)
        fx = likelihood(x, y, nu, fk)
        fx.backward()

        with torch.no_grad():
            if is_ddpm: 
                t = D.get_timestep_from_sigma(nu)
                x -= ( tau * x.grad + lmbda*tau *(x - D.denoise_step(x_tilde, t=max(t,1))))
            else: 
                x -= ( tau * x.grad  + lmbda*tau * (x - D(x_tilde, sigma=2*nu)))  

        x.grad.zero_()
        x = x.detach()
        metrics['psnr'].append(psnr(x0, x))
        metrics['residual'].append((torch.linalg.norm(x.detach() - xpre) / torch.linalg.norm(x0)).cpu())
        metrics['lpips'].append(lpips(x0, x))
        metrics['ssim'].append(ssim(x0, x))

    return x, metrics

def perform_annealed_snore(x0, y, D,fk, nu, likelihood=f, n_iter=50, sigma_0=None, sigma_m1=None, alpha_0=None, alpha_m1=None, m=None, is_ddpm=False):
    x = y.clone()
    tau = 1.9 * nu**2
    sigma_schedule = torch.linspace(sigma_0, sigma_m1, m).tolist()
    alpha_schedule = torch.linspace(alpha_0, alpha_m1, m).tolist()

    metrics = {
        'psnr': [psnr(x0, x)],
        'residual': [],
        'lpips': [lpips(x0, x)],
        'ssim': [ssim(x0, x)]
    }
    for i in range(m):
        sigma = sigma_schedule[i]
        alpha = alpha_schedule[i]
        lmbda = alpha / sigma**2 

        for it in tqdm(range(n_iter // m), desc=f'Annealing Level {i+1}/{m} - sigma: {sigma:.4f}, alpha: {alpha:.4f}'):            
            xpre = x.clone()
            x.requires_grad_(True)
            x_tilde = x + nu * torch.randn_like(x)
            fx = likelihood(x, y, nu, fk)
            fx.backward()

            with torch.no_grad():
                if is_ddpm: 
                    t = D.get_timestep_from_sigma(nu)
                    x -= ( tau * x.grad + tau*lmbda * (x - D.denoise_step(x_tilde, t=max(t,1))))
                else: 
                    x -= ( tau * x.grad + tau*lmbda * (x - D(x_tilde, sigma=2*sigma)) )

            x.grad.zero_()
            x = x.detach()
            metrics['psnr'].append(psnr(x0, x))
            metrics['residual'].append((torch.linalg.norm(x.detach() - xpre) / torch.linalg.norm(x0)).cpu())
            metrics['lpips'].append(lpips(x0, x))
            metrics['ssim'].append(ssim(x0, x))
    return x, metrics   




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

# def perform_pnp_pgd_inpainting(x0,y,  D, mask, nu, likelihood=f, n_iter = 50):
#     x = y.clone()

#     tau = 1.9 * nu**2 
#     metrics = {'psnr': [psnr(x0, x)]}

#     noise_schema = [0.1, 0.08, 0.07, 0.06, 0.05, 0.04, 0.02, 0.01]
#     timestep = [28, 21, 18, 14, 12, 9, 3, 1]
#     images = [x0, x]
#     iters = [0]

#     for it in tqdm(range(n_iter)):
#         x.requires_grad_(True)
#         fx = likelihood(x, y, nu, mask)
#         fx.backward()
#         with torch.no_grad():
#             x -= tau * x.grad
#         x.grad.zero_()
#         if (it-1)// (n_iter//8) != it // (n_iter//8):
#             print('Noise level changed to ', noise_schema[it // (n_iter // 8)])
#             noise_values = noise_schema[it // (n_iter // 8)]
#             t = timestep[it // (n_iter // 8)]
#         eps = noise_values * torch.randn_like(x)
#         # x = D.denoise_step(x+noise_values, t=t).detach()
#         strength = 2 * noise_values
#         x = D(x+noise_values, sigma=strength).detach()
#         metrics['psnr'].append(psnr(x0, x))
#         if (it) % (n_iter // 4) == (n_iter // 4)-1:
#             images.append(x.clone())
#             iters.append(it+1)
#     viewimages(images, iters, titre='', psnr=metrics['psnr'])

#     return x, metrics, images

# def perform_annealed_snore(x0, D,fk, n_iter, nu, sigma_0, sigma_m1, alpha_0, alpha_m1, m, f=f , A= A,  titre=''):
#     tau = 1.9 * nu**2  
#     s = 2 * nu  

#     sigma_schedule = torch.linspace(sigma_0, sigma_m1, m).tolist()
#     alpha_schedule = torch.linspace(alpha_0, alpha_m1, m).tolist()

#     y = A(x0, fk) + nu * torch.randn_like(x0)
#     x = y.clone()
#     viewimage(y, z=1)


#     metrics = {
#         'psnr': [psnr(x0, x)],
#         'residual': [],
#         'lpips': [lpips(x0, x)],
#         'ssim': [ssim(x0, x)]
#     }

#     images = [y]
#     iters = [0]

#     for i in range(m):
#         sigma = sigma_schedule[i]
#         alpha = alpha_schedule[i]
#         lmbda = alpha / sigma**2    

#         for it in tqdm(range(n_iter // m), desc=f'Annealing Level {i+1}/{m} - sigma: {sigma:.4f}, alpha: {alpha:.4f}'):
#             xpre = x.clone()
#             x.requires_grad_(True)

#             epsilon = torch.randn_like(x)
#             x_tilde = x + sigma * epsilon
#             fx = f(x, y, nu, fk)
#             fx.backward()

#             with torch.no_grad():
#                 x -= ( tau * x.grad + tau*lmbda * (x - D(x_tilde, sigma=2*sigma)) )

#             x.grad.zero_()
#             x = x.detach()
#             # Store metrics
#             metrics['psnr'].append(psnr(x0, x))
#             metrics['residual'].append((torch.linalg.norm(x.detach() - xpre) / torch.linalg.norm(x0)).cpu())
#             metrics['lpips'].append(lpips(x0, x))
#             metrics['ssim'].append(ssim(x0, x))

#         if (i + 1) % (m // 4) == 0:
#             images.append(x.clone())
#             iters.append(i)
#     images.append(x0)
#     viewimages_2(images, iters, titre=titre, psnr=metrics['psnr'])

#     return x, metrics, images


# def perform_annealed_snore_ddpm(x0, D,fk, n_iter, nu, sigma_0, sigma_m1, alpha_0, alpha_m1, m, f=f , A= A,  titre=''):
#     tau = 1.9 * nu**2  
#     sigma_schedule = torch.linspace(sigma_0, sigma_m1, m).tolist()
#     alpha_schedule = torch.linspace(alpha_0, alpha_m1, m).tolist()
#     y = A(x0, fk) + nu * torch.randn_like(x0)
#     viewimage(y, z=1)
#     x = y.clone()
#     metrics = {
#         'psnr': [psnr(x0, x)],
#         'residual': [],
#         'lpips': [lpips(x0, x)],
#         'ssim': [ssim(x0, x)]
#     }

#     images = [y]
#     iters = [0]
#     t = 0 
#     for i in range(m):
#         sigma = sigma_schedule[i]
#         alpha = alpha_schedule[i]
#         lmbda = alpha / sigma**2    
#         for it in tqdm(range(n_iter // m), desc=f'Annealing Level {i+1}/{m} - sigma: {sigma:.4f}, alpha: {alpha:.4f}'):
#             xpre = x.clone()
#             x.requires_grad_(True)

#             epsilon = torch.randn_like(x)
#             x_tilde = x + sigma * epsilon
#             fx = f(x, y, nu, fk)
#             fx.backward()
#             t = D.get_timestep_from_sigma(sigma)
#             with torch.no_grad():
#                 x -= ( tau * x.grad + tau*lmbda * (x - D.denoise_step(x_tilde, t=max(t,1))) )

#             x.grad.zero_()
#             x = x.detach()
#             # Store metrics
#             metrics['psnr'].append(psnr(x0, x))
#             metrics['residual'].append((torch.linalg.norm(x.detach() - xpre) / torch.linalg.norm(x0)).cpu())
#             metrics['lpips'].append(lpips(x0, x))
#             metrics['ssim'].append(ssim(x0, x))

#         if (i + 1) % (m // 4) == 0:
#             images.append(x.clone())
#             iters.append(i)
#     images.append(x0)
#     viewimages_2(images, iters, titre=titre, psnr=metrics['psnr'])

#     return x, metrics, images

# def perform_annealed_snore_inpainting(x0, D, delta, mask, n_iter, sigma_0, sigma_m1, alpha_0, alpha_m1, m,titre=''):
#     sigma_schedule = torch.linspace(sigma_0, sigma_m1, m).tolist()
#     alpha_schedule = torch.linspace(alpha_0, alpha_m1, m).tolist()
#     mask = mask.to(x0.device).float()
    
#     y = x0 * mask
#     y = torch.where(mask == 0, 0.5, y)
#     x = y.clone()
#     metrics = {'psnr': [psnr(x0, x)]}
#     images = [x]
#     iters = [0]
    
#     for i in range(m):
#         sigma = sigma_schedule[i]
#         alpha = alpha_schedule[i]
#         lmbda = alpha / sigma**2
#         for it in tqdm(range(n_iter // m), desc=f'Annealing Level {i+1}/{m} - sigma: {sigma:.4f}, alpha: {alpha:.4f}'):
#             epsilon = torch.randn_like(x)
#             x_tilde = x + sigma * epsilon
#             x = x - (delta * mask * (x - y) + delta * lmbda * (x - D(x_tilde, sigma=2*sigma).detach()))
#             metrics['psnr'].append(psnr(x0, x))
        
#         if (i + 1) % (m // 4) == 0:
#             images.append(x.clone())
#             iters.append(i)
#     images.append(x0)
#     viewimages_2(images, iters, titre=titre, psnr=metrics['psnr'])
    
#     return x, metrics, images

# def perform_annealed_snore_inpainting_ddpm(x0, D, delta, mask, n_iter, sigma_0, sigma_m1, alpha_0, alpha_m1, m,titre=''):
#     sigma_schedule = torch.linspace(sigma_0, sigma_m1, m).tolist()
#     alpha_schedule = torch.linspace(alpha_0, alpha_m1, m).tolist()
#     mask = mask.to(x0.device).float()
    
#     y = x0 * mask
#     x = torch.where(mask == 0, 0.5, y)
#     metrics = {
#         'psnr': [psnr(x0, x)],
#         'lpips': [lpips(x0, x)],
#         'ssim': [ssim(x0, x)]
#     }
    
#     images = [x]
#     iters = [0]
    
#     for i in range(m):
#         sigma = sigma_schedule[i]
#         alpha = alpha_schedule[i]
#         lmbda = alpha / sigma**2
#         t = D.get_timestep_from_sigma(sigma) 
#         for it in tqdm(range(n_iter // m), desc=f'Annealing Level {i+1}/{m} - sigma: {sigma:.4f}, alpha: {alpha:.4f}'):
#             epsilon = torch.randn_like(x)
#             x_tilde = x + sigma * epsilon
#             x -= (delta * mask * (x - y) + delta * lmbda * (x - D.denoise_step(x_tilde, t=t)))
#             x = x.detach()
            
#             metrics['psnr'].append(psnr(x0, x))
#             metrics['lpips'].append(lpips(x0, x))
#             metrics['ssim'].append(ssim(x0, x))
        
#         if (i + 1) % (m // 4) == 0:
#             images.append(x.clone())
#             iters.append(i)
#     images.append(x0)
#     viewimages(images, iters, titre=titre, psnr=metrics['psnr'])
    
#     return x, metrics, images