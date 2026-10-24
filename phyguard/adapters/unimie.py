from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from phyguard.likelihood import poisson_gaussian_nll


MODEL_FLAGS = dict(
    attention_resolutions="32,16,8",
    class_cond=False,
    diffusion_steps=1000,
    image_size=256,
    learn_sigma=True,
    noise_schedule="linear",
    num_channels=256,
    num_head_channels=64,
    num_heads=4,
    num_heads_upsample=-1,
    num_res_blocks=2,
    resblock_updown=True,
    use_fp16=True,
    use_scale_shift_norm=True,
    channel_mult="",
    dropout=0.0,
    timestep_respacing="",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    use_checkpoint=False,
    use_new_attention_order=False,
)


class UniMIEPhysicsSampler:
    def __init__(self, repo: str | Path, model_path: str | Path, device="cuda:0", denoise_steps=25, guidance_scale=3.0):
        self.repo = Path(repo).resolve()
        scripts = self.repo / "scripts"
        if str(scripts) not in sys.path:
            sys.path.insert(0, str(scripts))
        from guided_diffusion.script_util_x0_enhancement import create_model_and_diffusion_direct

        self.device = torch.device(device)
        self.model, self.diffusion = create_model_and_diffusion_direct(**MODEL_FLAGS)
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.to(self.device)
        if MODEL_FLAGS["use_fp16"] and self.device.type == "cuda":
            self.model.convert_to_fp16()
        self.model.eval()
        self.denoise_steps = int(denoise_steps)
        self.guidance_scale = float(guidance_scale)

    def run(self, observation_normalized: torch.Tensor, observation_counts: torch.Tensor, microscope) -> torch.Tensor:
        if int(getattr(microscope, "downsample", 1)) != 1:
            raise NotImplementedError(
                "The current UniMIE adapter supports downsample=1 only; "
                "extend measurement patch alignment before using SR operators."
            )
        x_lr = observation_normalized.to(self.device)
        if x_lr.ndim == 2:
            x_lr = x_lr[None, None]
        if x_lr.shape[1] == 1:
            x_lr = x_lr.repeat(1, 3, 1, 1)
        if min(x_lr.shape[-2:]) < 256:
            raise ValueError(
                "Official UniMIE sampling uses 256x256 overlapping patches; "
                "provide an observation of at least 256 pixels in each spatial dimension."
            )

        x_lr = x_lr.clamp(0, 1) * 2 - 1
        obs_counts = observation_counts.to(self.device)
        if obs_counts.ndim == 2:
            obs_counts = obs_counts[None, None]
        if obs_counts.shape[1] == 1:
            obs_counts = obs_counts.repeat(1, 3, 1, 1)

        def model_fn(x, t, y=None):
            return self.model(x, t, None)

        def cond_fn(x, t, light_factor=None, light_mask=None, corner=None, y=None):
            with torch.enable_grad():
                xin = x.detach().requires_grad_(True)
                candidate = ((xin + 1) / 2).clamp(0, 1)
                hi, wi, psz = corner
                obs_patch = obs_counts[:, :, hi:hi+psz, wi:wi+psz]

                mu = microscope.expected_counts(candidate)
                loss = poisson_gaussian_nll(obs_patch, mu, microscope.read_noise_std, reduction="mean")
                grad = -self.guidance_scale * torch.autograd.grad(loss, xin)[0]
                return light_factor, light_mask, grad

        classes = torch.zeros((x_lr.shape[0],), device=self.device, dtype=torch.long)
        model_kwargs = {"y": classes}
        light_factor = torch.zeros([1], device=self.device)
        light_mask = torch.zeros(x_lr.shape[-2:], device=self.device)
        sample, _, _ = self.diffusion.p_sample_loop(
            model_fn,
            list(x_lr.shape),
            light_factor,
            light_mask,
            clip_denoised=True,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=self.device,
            noise=x_lr,
            denoise_steps=self.denoise_steps,
        )
        return ((sample + 1) / 2).clamp(0, 1).mean(dim=1, keepdim=True)
