import numpy as np
import torch
from tqdm import tqdm 
from utils import viewimage


class DDPMPNP:
    def __init__(self, model, fixed_t=1):
        self.model = model
        self.fixed_t = fixed_t
        self.num_diffusion_timesteps = 1000
        beta_start = 0.0001
        beta_end = 0.02
        self.betas = np.linspace(beta_start, beta_end, self.num_diffusion_timesteps, dtype=np.float64)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)

    def denoise_step(self, x, t=None, a=0.3):
        if t is None: 
            t = self.fixed_t
        eps = self.model(x, torch.tensor(t, device=x.device).unsqueeze(0))
        eps = eps[:,:3,:,:]
        x_start = x - a * np.sqrt(1 - self.alphas_cumprod[t]) * eps
        return x_start 

class DPS:
  def __init__(self, model):
    self.num_diffusion_timesteps = 1000
    self.reversed_time_steps = np.arange(self.num_diffusion_timesteps)[::-1]
    beta_start = 0.0001
    beta_end = 0.02
    self.betas = np.linspace(beta_start, beta_end, self.num_diffusion_timesteps,
                              dtype=np.float64)
    self.alphas = 1.0 - self.betas
    self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
    self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
    self.model = model
    self.imgshape = (1,3,256,256)

  def get_eps_from_model(self, x, t):
    model_output = self.model(x, torch.tensor(t, device=x.device).unsqueeze(0))
    return model_output[:,:3,:,:]


  def predict_xstart_from_eps(self, x, eps, t):
    x_start = (
        np.sqrt(1.0 / self.alphas_cumprod[t])* x
        - np.sqrt(1.0 / self.alphas_cumprod[t] - 1) * eps
    )
    return x_start.clamp(-1.,1.)

  def posterior_sampling(self, linear_operator, y, fk, x_true=None, show_steps=True):
    x = torch.randn(self.imgshape,device=y.device)
    x.requires_grad = True
    for t in tqdm(self.reversed_time_steps):  
      z = torch.randn_like(x)

      eps = self.get_eps_from_model(x, t)
      x_start = self.predict_xstart_from_eps(x, eps, t)
      x_prime = np.sqrt(self.alphas[t])* (1 - self.alphas_cumprod_prev[t])/(1 - self.alphas_cumprod[t]) * x
      x_prime += self.betas[t]*np.sqrt(self.alphas_cumprod_prev[t])*x_start / (1 - self.alphas_cumprod[t])
      x_prime += np.sqrt(self.betas[t])*z
      
      sqdist = torch.sum((linear_operator(x_start, fk) - y)**2)
      grad = torch.autograd.grad(sqdist, x)[0]
      zeta = 1/ torch.sqrt(sqdist)

      x = x_prime - zeta*grad

      if show_steps and (t % 100 == 0 or t==999):
        print(f"Iteration = {t}")
        viewimage(x_start, z=0.5)
        # viewimage(torch.cat(((x+1)/2, (x_start+1)/2, (x_true+1)/2), dim=3))

    return x.detach().clamp(-1., 1.)

  
   
# import sys
# sys.path.append('GS_denoising/')
# from lightning_GSDRUNet import GradMatch
# from argparse import ArgumentParser

# class GradientStepDenoiser:
#     def __init__(self, checkpoint_path, device='cuda', act_mode='E', grayscale=False):
#         self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
#         # Initialize denoiser model
#         parser = ArgumentParser()
#         parser = GradMatch.add_model_specific_args(parser)
#         parser = GradMatch.add_optim_specific_args(parser)
#         hparams = parser.parse_known_args()[0]
#         hparams.act_mode = act_mode
#         hparams.grayscale = grayscale
        
#         self.model = GradMatch(hparams)
#         checkpoint = torch.load(checkpoint_path, map_location=self.device)
#         self.model.load_state_dict(checkpoint['state_dict'], strict=False)
#         self.model.eval()
#         for param in self.model.parameters():
#             param.requires_grad = False
#         self.model = self.model.to(self.device)

#     def __call__(self, x, sigma, weight=1.0):
#         torch.set_grad_enabled(True)
#         Dg, N, g = self.model.calculate_grad(x, sigma)
#         torch.set_grad_enabled(False)
#         Dg = Dg.detach()
#         Dx = x - weight * Dg
#         return Dx
    