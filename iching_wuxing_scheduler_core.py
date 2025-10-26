import torch
import numpy as np
from typing import List, Optional, Tuple, Union
from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_utils import SchedulerMixin, SchedulerOutput
from diffusers.configuration_utils import ConfigMixin, register_to_config
from dataclasses import dataclass

@dataclass
class IChingSchedulerOutput(SchedulerOutput):
    prev_sample: torch.FloatTensor

class WuxingDynamics:
    def __init__(self, growth_rates: np.ndarray, coupling_strength: float = 0.15):
        self.state = np.ones(5) * 0.5
        self.growth_rates = growth_rates
        self.coupling_strength = coupling_strength
        
        self.generative_matrix = np.array([
            [0.0, 0.8, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.8, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.8, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.8],
            [0.8, 0.0, 0.0, 0.0, 0.0]
        ])
        
        self.inhibitory_matrix = np.array([
            [0.0, 0.0, 0.6, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.6, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.6],
            [0.6, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.6, 0.0, 0.0, 0.0]
        ])
    
    def compute_derivatives(self, state: np.ndarray) -> np.ndarray:
        generative_effect = self.generative_matrix @ state
        inhibitory_effect = self.inhibitory_matrix @ state
        
        derivatives = self.growth_rates * state * (
            1.0 + self.coupling_strength * generative_effect 
            - self.coupling_strength * inhibitory_effect
        )
        
        return derivatives
    
    def step_rk4(self, dt: float):
        k1 = self.compute_derivatives(self.state)
        k2 = self.compute_derivatives(self.state + 0.5 * dt * k1)
        k3 = self.compute_derivatives(self.state + 0.5 * dt * k2)
        k4 = self.compute_derivatives(self.state + dt * k3)
        
        self.state = self.state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        self.state = np.clip(self.state, 0.1, 2.0)
        
        return self.state
    
    def get_parameters(self) -> dict:
        return {
            'adaptive_order_bias': self.state[0],
            'smoothing_strength': self.state[1],
            'damping_modulation': self.state[2],
            'stabilization_blend': self.state[3],
            'threshold_shift': self.state[4]
        }

class IChingWuxingScheduler(SchedulerMixin, ConfigMixin):
    def _resolve_mode_preset(self, mode: str) -> dict:
        m = (mode or "iching/wuxing").lower()
        if "/" in m:
            m = m.split("/")[-1]

        preset = {
            "coupling_strength": 0.15,
            "solver_order": 4,
            "final_damping_threshold": 0.10,
            "final_stabilize": False,
            "force_pure_ode": True,
            "eta": 0.0,
            "s_noise": 0.03,
            "sde_sigma_threshold": 0.60,
            "flat_suppress": 0.90,
            "sde_only": False,
        }

        table = {
            "wuxing": {
                "coupling_strength": 0.15,
                "solver_order": 4,
                "final_damping_threshold": 0.10,
                "final_stabilize": False
            },
            "wuxing-strong": {
                "coupling_strength": 0.30,
                "solver_order": 4,
                "final_damping_threshold": 0.08,
                "final_stabilize": False
            },
            "wuxing-stable": {
                "coupling_strength": 0.12,
                "solver_order": 4,
                "final_damping_threshold": 0.12,
                "final_stabilize": False
            },
            "wuxing-smooth": {
                "coupling_strength": 0.10,
                "solver_order": 4,
                "final_damping_threshold": 0.10,
                "final_stabilize": False
            },
            "wuxing-clean": {
                "coupling_strength": 0.12,
                "solver_order": 4,
                "final_damping_threshold": 0.15,
                "final_stabilize": False
            },
            "wuxing-sharp": {
                "coupling_strength": 0.18,
                "solver_order": 4,
                "final_damping_threshold": 0.08,
                "final_stabilize": False
            },
            "wuxing-film": {
                "coupling_strength": 0.12,
                "solver_order": 4,
                "final_damping_threshold": 0.10,
                "final_stabilize": False,
                "force_pure_ode": False,
                "eta": 0.12,
                "s_noise": 0.04,
                "sde_sigma_threshold": 0.70,
                "flat_suppress": 0.65,
                "sde_only": False,
            },
            "wuxing-sde-only": {
                "coupling_strength": 0.15,
                "solver_order": 4,
                "final_damping_threshold": 0.10,
                "final_stabilize": False,
                "force_pure_ode": False,
                "eta": 0.35,
                "s_noise": 0.07,
                "sde_sigma_threshold": 0.70,
                "flat_suppress": 0.50,
                "sde_only": True,
            },
            "wuxing-lowstep": {
                "coupling_strength": 0.35,
                "solver_order": 1,
                "final_damping_threshold": 0.01,
                "final_stabilize": False
            },
            "lowstep": {
                "coupling_strength": 0.35,
                "solver_order": 1,
                "final_damping_threshold": 0.01,
                "final_stabilize": False
            },
        }

        if m in table:
            preset = table[m]

        return preset

    @register_to_config
    def __init__(
        self,
        mode: str = "iching/wuxing",
    ):
        self.mode = mode
        self.shift = 3.0
        self.num_train_timesteps = 1000
        self.wuxing_dt = 0.1
        self.use_latent_postprocessing = True

        preset = self._resolve_mode_preset(self.mode)
        self.coupling_strength = preset["coupling_strength"]
        self.solver_order = preset["solver_order"]
        self.final_damping_threshold = preset["final_damping_threshold"]
        self.final_stabilize = preset["final_stabilize"]

        m = (mode or "iching/wuxing").lower()
        if "/" in m:
            m = m.split("/")[-1]
        self.is_lowstep_mode = ("lowstep" in m)

        self.edge_suppress = 0.60
        self.edge_ode_boost = 0.20
        self.ghost_suppress = 0.70
        self.smoothness = 0.65
        self.detail_preserve = 0.15
        self.sde_highpass = 1.00
        self.flat_suppress = preset.get("flat_suppress", 0.90)

        self.force_pure_ode = preset.get("force_pure_ode", True)
        self.eta = preset.get("eta", 0.0)
        self.s_noise = preset.get("s_noise", 0.03)
        self.sde_sigma_threshold = preset.get("sde_sigma_threshold", 0.60)
        self.sde_only = preset.get("sde_only", False)
        self.force_sde_ratio = 0.0

        self._color_ref_mean = None
        self._color_ref_std = None
        self._color_ref_set = False

        self.smoothed_velocity = None
        self.convergence_threshold = 0.12
        self.smoothing_factor = 0.85
        
        growth_rates = np.array([0.5, 0.7, 0.3, 0.6, 0.4])
        self.wuxing = WuxingDynamics(growth_rates, self.coupling_strength)
        
        self.sigmas = None
        self.timesteps = None
        self._step_index = None
        self.velocity_buffer = []
        
    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        self.num_inference_steps = num_inference_steps
        
        timesteps = np.linspace(1, 0, num_inference_steps + 1)

        if num_inference_steps <= 10 or self.is_lowstep_mode:
            sigmas_base = timesteps.copy()
        else:
            sigmas_base = 0.5 * (1 + np.cos(np.pi * timesteps))

        sigmas_shifted = self.shift * sigmas_base / (1 + (self.shift - 1) * sigmas_base)
        
        self.sigmas = torch.from_numpy(sigmas_shifted).to(dtype=torch.float32, device=device)
        self.timesteps = (self.sigmas[:-1] * 1000).to(torch.int64)
        if device is not None:
            self.timesteps = self.timesteps.to(device)

        self._step_index = 0
        self.velocity_buffer = []
        self.smoothed_velocity = None

        self._color_ref_mean = None
        self._color_ref_std = None
        self._color_ref_set = False

        wuxing_reset_rates = np.array([0.5, 0.7, 0.3, 0.6, 0.4])
        self.wuxing = WuxingDynamics(wuxing_reset_rates, self.coupling_strength)
    
    def scale_model_input(self, sample: torch.FloatTensor, timestep: Union[float, torch.FloatTensor]) -> torch.FloatTensor:
        return sample

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[IChingSchedulerOutput, Tuple]:
        step_index = self._step_index
        sigma = self.sigmas[step_index]
        sigma_next = self.sigmas[step_index + 1]
        sigma_val = float(sigma)

        self.wuxing.step_rk4(self.wuxing_dt)
        params = self.wuxing.get_parameters()

        self.velocity_buffer.append(model_output)
        if len(self.velocity_buffer) > self.solver_order + 1:
            self.velocity_buffer.pop(0)

        adaptive_order = self.solver_order

        if self.num_inference_steps <= 8:
            adaptive_order = min(3, self.solver_order)
        elif self.num_inference_steps <= 15:
            if sigma_val > 0.8:
                adaptive_order = min(3, self.solver_order)
            else:
                adaptive_order = self.solver_order
        else:
            adaptive_order = self.solver_order

        if len(self.velocity_buffer) >= 4 and adaptive_order >= 4:
            velocity = (
                (55/24) * self.velocity_buffer[-1] -
                (59/24) * self.velocity_buffer[-2] +
                (37/24) * self.velocity_buffer[-3] -
                (9/24) * self.velocity_buffer[-4]
            )
        elif len(self.velocity_buffer) >= 3 and adaptive_order >= 3:
            velocity = (
                (23/12) * self.velocity_buffer[-1] -
                (16/12) * self.velocity_buffer[-2] +
                (5/12) * self.velocity_buffer[-3]
            )
        elif len(self.velocity_buffer) >= 2 and adaptive_order >= 2:
            velocity = 1.5 * self.velocity_buffer[-1] - 0.5 * self.velocity_buffer[-2]
        else:
            velocity = model_output

        if sigma_val < self.convergence_threshold:
            if self.smoothed_velocity is None:
                self.smoothed_velocity = velocity
            else:
                alpha = self.smoothing_factor
                self.smoothed_velocity = alpha * self.smoothed_velocity + (1 - alpha) * velocity
            velocity = self.smoothed_velocity
        elif sigma_val < 0.25:
            if self.smoothed_velocity is None:
                self.smoothed_velocity = velocity
            else:
                alpha = 0.65
                self.smoothed_velocity = alpha * self.smoothed_velocity + (1 - alpha) * velocity
            velocity = self.smoothed_velocity
        else:
            self.smoothed_velocity = velocity

        dt = sigma_next - sigma
        if self.num_inference_steps > 8:
            damping_mod = params['damping_modulation']
            threshold = self.convergence_threshold * (damping_mod / 1.5)
            if sigma_val < threshold:
                base_damping = 0.5 + 0.5 * (sigma_val / threshold)
                dt = dt * base_damping
        elif self.num_inference_steps >= 4:
            if sigma_val < 0.05:
                damping = 0.5 + 0.5 * (sigma_val / 0.05)
                dt = dt * damping

        edge_soft = None
        flat_soft = None
        if sample.dim() == 4:
            if sample.shape[2] >= 3 and sample.shape[3] >= 3:
                gx = torch.abs(sample[:, :, :, 1:] - sample[:, :, :, :-1])
                gy = torch.abs(sample[:, :, 1:, :] - sample[:, :, :-1, :])
                gx = torch.cat([gx, gx[:, :, :, -1:]], dim=3)
                gy = torch.cat([gy, gy[:, :, -1:, :]], dim=2)
                grad_mag = gx + gy
                grad_mag = grad_mag.mean(dim=1, keepdim=True)
                eps = 1e-6
                gmin = grad_mag.amin(dim=(2, 3), keepdim=True)
                gmax = grad_mag.amax(dim=(2, 3), keepdim=True)
                edge_soft = (grad_mag - gmin) / (gmax - gmin + eps)
                edge_soft = torch.nn.functional.avg_pool2d(edge_soft, kernel_size=3, stride=1, padding=1)
                flat_soft = 1.0 - edge_soft
        elif sample.dim() == 5:
            if sample.shape[3] >= 3 and sample.shape[4] >= 3:
                gx = torch.abs(sample[:, :, :, :, 1:] - sample[:, :, :, :, :-1])
                gy = torch.abs(sample[:, :, :, 1:, :] - sample[:, :, :, :-1, :])
                gx = torch.cat([gx, gx[:, :, :, :, -1:]], dim=4)
                gy = torch.cat([gy, gy[:, :, :, -1:, :]], dim=3)
                grad_mag = gx + gy
                grad_mag = grad_mag.mean(dim=1, keepdim=True)
                eps = 1e-6
                gmin = grad_mag.amin(dim=(2, 3, 4), keepdim=True)
                gmax = grad_mag.amax(dim=(2, 3, 4), keepdim=True)
                edge_soft = (grad_mag - gmin) / (gmax - gmin + eps)
                edge_soft = torch.nn.functional.avg_pool3d(
                    edge_soft, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)
                )
                flat_soft = 1.0 - edge_soft

        ode_sample = sample + velocity * dt

        allow_sde = (not self.force_pure_ode) and (self.eta > 0) and (sigma_val > self.sde_sigma_threshold)
        if allow_sde:
            noise = randn_tensor(sample.shape, generator=generator, device=sample.device, dtype=sample.dtype)
            
            noise_scale = self.s_noise
            
            sigma_noise = (sigma_val ** 2) * self.eta * noise_scale
            
            if edge_soft is not None:
                edge_suppress_factor = 0.5
                noise = noise * (1.0 - edge_suppress_factor * edge_soft)
                if flat_soft is not None:
                    noise = noise * (1.0 - self.flat_suppress * flat_soft)
            
            sde_sample = ode_sample + sigma_noise * noise
        else:
            sde_sample = ode_sample

        if self.sde_only:
            ode_weight = 0.0 if allow_sde else 1.0
        else:
            ode_weight = 0.97 if allow_sde else 1.0
        
        prev_sample = ode_weight * ode_sample + (1.0 - ode_weight) * sde_sample

        if self.smoothness > 0 and sigma_val < 0.5 and prev_sample.dim() >= 4:
            if prev_sample.dim() == 4:
                k = 3
                pad = k // 2
                p = torch.nn.functional.pad(prev_sample, (pad, pad, pad, pad), mode='replicate')
                B, C, H, W = prev_sample.shape
                u = torch.nn.functional.unfold(p, kernel_size=k, stride=1)
                u = u.view(B, C, k * k, H * W)
                med = u.median(dim=2).values
                smoothed = med.view(B, C, H, W)
            elif prev_sample.dim() == 5:
                k = 3
                pad = k // 2
                B, C, T, H, W = prev_sample.shape
                x = prev_sample.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
                p = torch.nn.functional.pad(x, (pad, pad, pad, pad), mode='replicate')
                u = torch.nn.functional.unfold(p, kernel_size=k, stride=1)
                u = u.view(B * T, C, k * k, H * W)
                med = u.median(dim=2).values
                smoothed = med.view(B * T, C, H, W).view(B, T, C, H, W).permute(0, 2, 1, 3, 4).contiguous()
            else:
                smoothed = prev_sample

            if sigma_val < self.convergence_threshold:
                smooth_strength = self.smoothness * 0.3
            else:
                smooth_strength = 0.0
            smooth_strength = max(0.0, min(1.0, smooth_strength))
            if flat_soft is not None:
                mask = smooth_strength * flat_soft
            else:
                mask = smooth_strength
            prev_sample = (1 - mask) * prev_sample + mask * smoothed

        if self.final_stabilize and self.num_inference_steps > 8 and len(self.velocity_buffer) >= 3:
            blend_mod = params['stabilization_blend']
            threshold_water = 0.05 * (2.0 - params['threshold_shift'] * 0.3)
            if sigma_val < threshold_water:
                avg_velocity = sum(self.velocity_buffer[-3:]) / 3
                stabilized = sample + avg_velocity * dt
                blend_strength = (sigma_val / threshold_water) * (blend_mod / 1.5)
                blend_strength = max(0.0, min(1.0, blend_strength))
                prev_sample = blend_strength * prev_sample + (1 - blend_strength) * stabilized

        if self.use_latent_postprocessing:
            if sigma_val < 0.20:
                prev_sample = self._apply_latent_postprocessing(prev_sample)
            elif sigma_val < 0.35:
                prev_sample = self._vae_compatible_postprocess(prev_sample)

        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return IChingSchedulerOutput(prev_sample=prev_sample)

    def _sigma_to_alpha_sigma_t(self, sigma):
        alpha_t = 1 / torch.sqrt(1 + sigma**2)
        sigma_t = sigma * alpha_t
        return alpha_t, sigma_t

    def _vae_compatible_postprocess(self, latent: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(latent, nan=0.0, posinf=4.0, neginf=-4.0)
        x = torch.clamp(x, -10.0, 10.0)
        n = torch.linalg.norm(x)
        if torch.isfinite(n) and n > 50.0:
            x = x * (20.0 / n)
        return x

    def _apply_latent_postprocessing(self, latent: torch.Tensor) -> torch.Tensor:
        processed = latent.clone()

        processed = torch.clamp(processed, -5.0, 5.0)

        q01 = torch.quantile(processed, 0.02)
        q99 = torch.quantile(processed, 0.98)
        processed = torch.clamp(processed, q01, q99)

        if processed.dim() == 4 and processed.shape[2] > 3 and processed.shape[3] > 3:
            k = 3
            pad = k // 2
            p = torch.nn.functional.pad(processed, (pad, pad, pad, pad), mode='replicate')
            B, C, H, W = processed.shape
            u = torch.nn.functional.unfold(p, kernel_size=k, stride=1)
            u = u.view(B, C, k * k, H * W)
            med = u.median(dim=2).values
            processed = med.view(B, C, H, W)
        elif processed.dim() == 5 and processed.shape[3] > 3 and processed.shape[4] > 3:
            k = 3
            pad = k // 2
            B, C, T, H, W = processed.shape
            x = processed.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
            p = torch.nn.functional.pad(x, (pad, pad, pad, pad), mode='replicate')
            u = torch.nn.functional.unfold(p, kernel_size=k, stride=1)
            u = u.view(B * T, C, k * k, H * W)
            med = u.median(dim=2).values
            x2d = med.view(B * T, C, H, W).view(B, T, C, H, W).permute(0, 2, 1, 3, 4).contiguous()
            if T >= 3:
                pt = torch.nn.functional.pad(x2d, (0, 0, 0, 0, 1, 1), mode='replicate')
                t0 = pt[:, :, 0:T, :, :]
                t1 = pt[:, :, 1:1+T, :, :]
                t2 = pt[:, :, 2:2+T, :, :]
                stacked = torch.stack([t0, t1, t2], dim=2)
                processed = stacked.median(dim=2).values
            else:
                processed = x2d

        current_std = torch.std(processed)
        if current_std > 1e-6:
            target_std = min(1.0, current_std.item())
            processed = processed * (target_std / current_std)

        processed = processed - torch.mean(processed)

        return processed

    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.FloatTensor,
    ) -> torch.FloatTensor:
        sigmas = timesteps.to(dtype=torch.float32)
        if sigmas.max() > 1.0:
            sigmas = sigmas / 1000.0

        while len(sigmas.shape) < len(original_samples.shape):
            sigmas = sigmas.unsqueeze(-1)

        sigma_mix = sigmas * sigmas
        if getattr(self, 'force_pure_ode', False):
            sigma_mix = torch.where(sigmas > 0.60, sigma_mix, torch.zeros_like(sigma_mix))

        flat_soft = None
        with torch.no_grad():
            if original_samples.dim() == 4 and original_samples.shape[2] >= 3 and original_samples.shape[3] >= 3:
                gx = torch.abs(original_samples[:, :, :, 1:] - original_samples[:, :, :, :-1])
                gy = torch.abs(original_samples[:, :, 1:, :] - original_samples[:, :, :-1, :])
                gx = torch.cat([gx, gx[:, :, :, -1:]], dim=3)
                gy = torch.cat([gy, gy[:, :, -1:, :]], dim=2)
                grad_mag = (gx + gy).mean(dim=1, keepdim=True)
                eps = 1e-6
                gmin = grad_mag.amin(dim=(2, 3), keepdim=True)
                gmax = grad_mag.amax(dim=(2, 3), keepdim=True)
                edge_soft = (grad_mag - gmin) / (gmax - gmin + eps)
                flat_soft = 1.0 - torch.nn.functional.avg_pool2d(edge_soft, kernel_size=3, stride=1, padding=1)
            elif original_samples.dim() == 5 and original_samples.shape[3] >= 3 and original_samples.shape[4] >= 3:
                gx = torch.abs(original_samples[:, :, :, :, 1:] - original_samples[:, :, :, :, :-1])
                gy = torch.abs(original_samples[:, :, :, 1:, :] - original_samples[:, :, :, :-1, :])
                gx = torch.cat([gx, gx[:, :, :, :, -1:]], dim=4)
                gy = torch.cat([gy, gy[:, :, :, -1:, :]], dim=3)
                grad_mag = (gx + gy).mean(dim=1, keepdim=True)
                eps = 1e-6
                gmin = grad_mag.amin(dim=(2, 3, 4), keepdim=True)
                gmax = grad_mag.amax(dim=(2, 3, 4), keepdim=True)
                edge_soft = (grad_mag - gmin) / (gmax - gmin + eps)
                flat_soft = 1.0 - torch.nn.functional.avg_pool3d(edge_soft, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))

        n = noise
        if n.dim() == 4 and n.shape[2] > 3 and n.shape[3] > 3:
            k = 3; pad = k // 2
            p = torch.nn.functional.pad(n, (pad, pad, pad, pad), mode='replicate')
            n_blur = torch.nn.functional.avg_pool2d(p, kernel_size=k, stride=1)
            p2 = torch.nn.functional.pad(n_blur, (pad, pad, pad, pad), mode='replicate')
            n_blur = torch.nn.functional.avg_pool2d(p2, kernel_size=k, stride=1)
            n = 0.5 * n + 0.5 * n_blur
        elif n.dim() == 5 and n.shape[3] > 3 and n.shape[4] > 3:
            k = 3; pad = k // 2
            p = torch.nn.functional.pad(n, (pad, pad, pad, pad, 0, 0), mode='replicate')
            n_blur = torch.nn.functional.avg_pool3d(p, kernel_size=(1, k, k), stride=1)
            p2 = torch.nn.functional.pad(n_blur, (pad, pad, pad, pad, 0, 0), mode='replicate')
            n_blur = torch.nn.functional.avg_pool3d(p2, kernel_size=(1, k, k), stride=1)
            n = 0.5 * n + 0.5 * n_blur

        if flat_soft is not None:
            n = n * (1.0 - self.flat_suppress * flat_soft)

        noisy_samples = (1 - sigma_mix) * original_samples + sigma_mix * n
        return noisy_samples

    def __len__(self):
        return self.config.num_train_timesteps
