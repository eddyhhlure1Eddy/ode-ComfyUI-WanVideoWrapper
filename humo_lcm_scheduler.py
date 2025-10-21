"""
Original source from kijai's ComfyUI-WanVideoWrapper
https://github.com/kijai/ComfyUI-WanVideoWrapper/tree/main/wanvideo/schedulers

Modified and optimized by eddy
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput, is_scipy_available, logging
from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_utils import SchedulerMixin

if is_scipy_available():
    import scipy.stats

logger = logging.get_logger(__name__)


@dataclass
class HumoLCMSchedulerOutput(BaseOutput):
    prev_sample: torch.FloatTensor


class HumoLCMScheduler(SchedulerMixin, ConfigMixin):
    _compatibles = []
    order = 1

    @register_to_config
    def __init__(
            self,
            num_train_timesteps: int = 1000,
            shift: float = 1.0,
            use_dynamic_shifting: bool = False,
            base_shift: Optional[float] = 0.5,
            max_shift: Optional[float] = 1.15,
            base_image_seq_len: Optional[int] = 256,
            max_image_seq_len: Optional[int] = 4096,
            invert_sigmas: bool = False,
            shift_terminal: Optional[float] = None,
            use_karras_sigmas: Optional[bool] = False,
            use_exponential_sigmas: Optional[bool] = False,
            use_beta_sigmas: Optional[bool] = False,
            time_shift_type: str = "exponential",
            scale_factors: Optional[List[float]] = None,
            upscale_mode: Optional[str] = 'bicubic',
            dynamic_boost: float = 0.15,
            contrast_factor: float = 0.95,
    ):
        if self.config.use_beta_sigmas and not is_scipy_available():
            raise ImportError("Make sure to install scipy if you want to use beta sigmas.")
        if sum([self.config.use_beta_sigmas, self.config.use_exponential_sigmas, self.config.use_karras_sigmas]) > 1:
            raise ValueError(
                "Only one of `config.use_beta_sigmas`, `config.use_exponential_sigmas`, `config.use_karras_sigmas` can be used."
            )

        timesteps = np.linspace(1, num_train_timesteps, num_train_timesteps, dtype=np.float32)[::-1].copy()
        timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32)

        sigmas = timesteps / num_train_timesteps
        if not use_dynamic_shifting:
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        self.timesteps = sigmas * num_train_timesteps

        self._step_index = None
        self._begin_index = None
        self._shift = shift
        self._init_size = None
        self._scale_factors = scale_factors
        self._upscale_mode = upscale_mode

        self.sigmas = sigmas.to("cpu")
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()

    @property
    def shift(self):
        return self._shift

    @property
    def step_index(self):
        return self._step_index

    @property
    def begin_index(self):
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0):
        self._begin_index = begin_index

    def set_shift(self, shift: float):
        self._shift = shift

    def set_scale_factors(self, scale_factors: list, upscale_mode):
        self._scale_factors = scale_factors
        self._upscale_mode = upscale_mode

    def scale_noise(
            self,
            sample: torch.FloatTensor,
            timestep: Union[float, torch.FloatTensor],
            noise: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        sigmas = self.sigmas.to(device=sample.device, dtype=sample.dtype)

        if sample.device.type == "mps" and torch.is_floating_point(timestep):
            schedule_timesteps = self.timesteps.to(sample.device, dtype=torch.float32)
            timestep = timestep.to(sample.device, dtype=torch.float32)
        else:
            schedule_timesteps = self.timesteps.to(sample.device)
            timestep = timestep.to(sample.device)

        if self.begin_index is None:
            step_indices = [self.index_for_timestep(t, schedule_timesteps) for t in timestep]
        elif self.step_index is not None:
            step_indices = [self.step_index] * timestep.shape[0]
        else:
            step_indices = [self.begin_index] * timestep.shape[0]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < len(sample.shape):
            sigma = sigma.unsqueeze(-1)

        sample = sigma * noise + (1.0 - sigma) * sample

        return sample

    def _sigma_to_t(self, sigma):
        return sigma * self.config.num_train_timesteps

    def set_timesteps(
            self,
            num_inference_steps: Optional[int] = None,
            device: Union[str, torch.device] = None,
            sigmas: Optional[List[float]] = None,
            mu: Optional[float] = None,
            timesteps: Optional[List[float]] = None,
    ):
        if self.config.use_dynamic_shifting and mu is None:
            raise ValueError("`mu` must be passed when `use_dynamic_shifting` is set to be `True`")

        if sigmas is not None and timesteps is not None:
            if len(sigmas) != len(timesteps):
                raise ValueError("`sigmas` and `timesteps` should have the same length")

        if num_inference_steps is not None:
            if (sigmas is not None and len(sigmas) != num_inference_steps) or (
                    timesteps is not None and len(timesteps) != num_inference_steps
            ):
                raise ValueError(
                    "`sigmas` and `timesteps` should have the same length as num_inference_steps, if `num_inference_steps` is provided"
                )
        else:
            num_inference_steps = len(sigmas) if sigmas is not None else len(timesteps)

        self.num_inference_steps = num_inference_steps

        is_timesteps_provided = timesteps is not None

        if is_timesteps_provided:
            timesteps = np.array(timesteps).astype(np.float32)

        if sigmas is None:
            if timesteps is None:
                timesteps = np.linspace(
                    self._sigma_to_t(self.sigma_max), self._sigma_to_t(self.sigma_min), num_inference_steps
                )
            sigmas = timesteps / self.config.num_train_timesteps
        else:
            sigmas = np.array(sigmas).astype(np.float32)
            num_inference_steps = len(sigmas)

        if self.config.use_dynamic_shifting:
            sigmas = self.time_shift(mu, 1.0, sigmas)
        else:
            sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)

        if self.config.shift_terminal:
            sigmas = self.stretch_shift_to_terminal(sigmas)

        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)
        if not is_timesteps_provided:
            timesteps = sigmas * self.config.num_train_timesteps
        else:
            timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32, device=device)

        if self.config.invert_sigmas:
            sigmas = 1.0 - sigmas
            timesteps = sigmas * self.config.num_train_timesteps
            sigmas = torch.cat([sigmas, torch.ones(1, device=sigmas.device)])
        else:
            sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])

        self.timesteps = timesteps
        self.sigmas = sigmas
        self._step_index = None
        self._begin_index = None

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep).nonzero()
        pos = 1 if len(indices) > 1 else 0
        return indices[pos].item()

    def _init_step_index(self, timestep):
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def step(
            self,
            model_output: torch.FloatTensor,
            timestep: Union[float, torch.FloatTensor],
            sample: torch.FloatTensor,
            s_churn: float = 0.0,
            s_tmin: float = 0.0,
            s_tmax: float = float("inf"),
            s_noise: float = 1.0,
            generator: Optional[torch.Generator] = None,
            per_token_timesteps: Optional[torch.Tensor] = None,
            return_dict: bool = True,
    ) -> Union[HumoLCMSchedulerOutput, Tuple]:


        if (
                isinstance(timestep, int)
                or isinstance(timestep, torch.IntTensor)
                or isinstance(timestep, torch.LongTensor)
        ):
            raise ValueError(
                "Passing integer indices as timesteps is not supported."
            )

        if self._init_size is None or self.step_index is None:
            self._init_size = model_output.size()[2:]

        if self.step_index is None:
            self._init_step_index(timestep)

        sample = sample.to(torch.float32)

        sigma = self.sigmas[self.step_index]
        sigma_next = self.sigmas[self.step_index + 1]

        x0_pred = (sample - sigma * model_output)

        step_ratio = self.step_index / max(len(self.sigmas) - 1, 1)

        dynamic_factor = self.config.dynamic_boost * (1.0 - step_ratio)
        x0_mean = x0_pred.mean()
        x0_pred = x0_mean + (x0_pred - x0_mean) * (1.0 + dynamic_factor * 0.5)

        contrast = self.config.contrast_factor
        x0_pred = x0_mean + (x0_pred - x0_mean) * contrast

        if self._scale_factors and self._upscale_mode:
            if self._step_index < len(self._scale_factors):
                size = [
                    round(self._scale_factors[self._step_index] * size)
                    for size in self._init_size
                ]
                x0_pred = torch.nn.functional.interpolate(
                    x0_pred,
                    size=size,
                    mode=self._upscale_mode
                )

        noise = randn_tensor(
            x0_pred.shape, generator=generator, device=x0_pred.device, dtype=x0_pred.dtype
        )

        prev_sample = (1 - sigma_next) * x0_pred + sigma_next * noise

        self._step_index += 1
        if per_token_timesteps is None:
            prev_sample = prev_sample.to(model_output.dtype)

        if not return_dict:
            return (prev_sample,)

        return HumoLCMSchedulerOutput(prev_sample=prev_sample)

    def time_shift(self, mu: float, sigma: float, t: torch.Tensor):
        if self.config.time_shift_type == "exponential":
            return self._time_shift_exponential(mu, sigma, t)
        elif self.config.time_shift_type == "linear":
            return self._time_shift_linear(mu, sigma, t)

    def stretch_shift_to_terminal(self, t: torch.Tensor) -> torch.Tensor:
        one_minus_z = 1 - t
        scale_factor = one_minus_z[-1] / (1 - self.config.shift_terminal)
        stretched_t = 1 - (one_minus_z / scale_factor)
        return stretched_t

    def _time_shift_exponential(self, mu, sigma, t):
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

    def _time_shift_linear(self, mu, sigma, t):
        return mu / (mu + (1 / t - 1) ** sigma)

    def __len__(self):
        return self.config.num_train_timesteps


def create_humo_lcm_scheduler(
    num_train_timesteps: int = 1000,
    shift: float = 1.0,
    dynamic_boost: float = 0.15,
    contrast_factor: float = 0.95,
    **kwargs
) -> HumoLCMScheduler:
    return HumoLCMScheduler(
        num_train_timesteps=num_train_timesteps,
        shift=shift,
        dynamic_boost=dynamic_boost,
        contrast_factor=contrast_factor,
        **kwargs
    )


if __name__ == "__main__":
    print("HUMO LCM Scheduler - Standalone Version")
    print("Based on lcm+/contrast_normal configuration")
    print("Author: eddy")
    print()

    scheduler = create_humo_lcm_scheduler()

    print(f"Configuration Parameters:")
    print(f"  num_train_timesteps: {scheduler.config.num_train_timesteps}")
    print(f"  shift: {scheduler.config.shift}")
    print(f"  dynamic_boost: {scheduler.config.dynamic_boost}")
    print(f"  contrast_factor: {scheduler.config.contrast_factor}")
    print()
    print("Scheduler created successfully!")
