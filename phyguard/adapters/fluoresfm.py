from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import numpy as np
import torch


os.environ.setdefault("HF_HUB_OFFLINE", "1")


def _install_evaluation_stub() -> None:

    if "utils.evaluation" in sys.modules:
        return
    stub = types.ModuleType("utils.evaluation")

    def snr(ground_truth, estimate):
        gt = np.asarray(ground_truth, dtype=float)
        est = np.asarray(estimate, dtype=float)
        denom = float(np.sum((gt - est) ** 2))
        if denom < 1e-12:
            return float("inf")
        return 10.0 * np.log10(float(np.sum(gt**2)) / denom)

    stub.SNR = snr
    sys.modules["utils.evaluation"] = stub


class FluoResFMInference:
    def __init__(
        self,
        repo: str | Path,
        checkpoint: str | Path,
        biomedclip_json: str | Path,
        biomedclip_bin: str | Path,
        device: str = "cuda:0",
        context_length: int = 160,
        patch_size: int = 256,
    ):
        self.repo = Path(repo).resolve()
        if str(self.repo) not in sys.path:
            sys.path.insert(0, str(self.repo))
        _install_evaluation_stub()
        from models.biomedclip_embedder import BiomedCLIPTextEmbedder
        from models.unet_sd_c import UNetModel
        import utils.data as utils_data
        import utils.optim as utils_optim

        self.torch_device = torch.device(device)
        self.utils_data = utils_data
        self.patch_size = int(patch_size)
        self.normalizer = utils_data.NormalizePercentile(0.03, 0.995)
        self.stitcher = utils_data.Patch_stitcher(
            patch_size=self.patch_size, overlap=self.patch_size // 4, padding_mode="reflect"
        )
        self.embedder = BiomedCLIPTextEmbedder(
            path_json=str(biomedclip_json),
            path_bin=str(biomedclip_bin),
            context_length=context_length,
            device=self.torch_device,
        )
        self.embedder.eval()
        self.model = UNetModel(
            in_channels=1,
            out_channels=1,
            channels=320,
            n_res_blocks=1,
            attention_levels=[0, 1, 2, 3],
            channel_multipliers=[1, 2, 4, 4],
            n_heads=8,
            tf_layers=1,
            d_cond=768,
            pixel_shuffle=False,
            scale_factor=4,
            structural_prompt=False,
            n_tokens=0,
        ).to(self.torch_device)
        state = torch.load(checkpoint, map_location=self.torch_device, weights_only=True)
        state_dict = state["model_state_dict"] if "model_state_dict" in state else state
        state_dict = utils_optim.on_load_checkpoint(checkpoint=state_dict)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self._install_sdpa_attention()

    @staticmethod
    def _install_sdpa_attention() -> None:

        import models.unet_attention as ua

        def sdpa_normal_attention(module, q, k, v):
            bq, sq, _ = q.shape
            bk, sk, _ = k.shape
            q = q.view(bq, sq, module.n_heads, -1).transpose(1, 2)
            k = k.view(bk, sk, module.n_heads, -1).transpose(1, 2)
            v = v.view(bk, sk, module.n_heads, -1).transpose(1, 2)
            if bk != bq:
                k = k.expand(bq, -1, -1, -1)
                v = v.expand(bq, -1, -1, -1)
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, scale=module.scale
            )
            out = out.transpose(1, 2).reshape(bq, sq, -1)
            return module.to_out(out)

        ua.CrossAttention.normal_attention = sdpa_normal_attention

    @torch.inference_mode()
    def __call__(self, image: np.ndarray, prompt: str, amp: bool = True) -> np.ndarray:
        x = np.asarray(image, np.float32)
        x = np.clip(x, 0, None)
        x = self.normalizer(x)
        if x.ndim != 2:
            x = np.squeeze(x)
        xt = torch.tensor(x[None, None], device=self.torch_device)
        cond = self.embedder(prompt).to(self.torch_device)
        h0, w0 = xt.shape[-2:]


        if h0 < self.patch_size or w0 < self.patch_size:
            ph, pw = max(0, self.patch_size - h0), max(0, self.patch_size - w0)
            xt = torch.nn.functional.pad(xt, (0, pw, 0, ph), mode="reflect")

        with torch.autocast(self.torch_device.type, torch.float16, enabled=amp and self.torch_device.type == "cuda"):
            if max(xt.shape[-2:]) > self.patch_size:
                patches = self.stitcher.unfold(xt, showinfo=False)
                outs = []


                for i in range(0, patches.shape[0], 1):
                    outs.append(self.model(patches[i:i+1], None, cond))
                out_patches = torch.cat(outs, dim=0)
                out = self.stitcher.fold_linear_ramp(
                    patches=out_patches,
                    original_image_shape=(1, 1, xt.shape[-2], xt.shape[-1]),
                )
                out = torch.as_tensor(out, device=self.torch_device)
            else:
                out = self.model(xt, None, cond)
        return out[0, 0, :h0, :w0].float().cpu().numpy()
