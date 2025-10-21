import torch
import math
from typing import Optional, Union
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin, SchedulerOutput


class FlowMatchSAODEStableScheduler(SchedulerMixin, ConfigMixin):
    
    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 3.0,
        solver_order: int = 3,
        use_adaptive_order: bool = True,
        use_velocity_smoothing: bool = True,
        convergence_threshold: float = 0.15,
        smoothing_factor: float = 0.8,
        eta: float = 0.0,
        use_pece: bool = False,
        predictor_order: int = 3,
        corrector_order: int = 4,
    ):
        self.solver_order = solver_order
        self.use_adaptive_order = use_adaptive_order
        self.use_velocity_smoothing = use_velocity_smoothing
        self.convergence_threshold = convergence_threshold
        self.smoothing_factor = smoothing_factor
        

        self.velocity_buffer = []
        self.smoothed_velocity = None
        self.step_count = 0
        
    def set_timesteps(
        self, 
        num_inference_steps: int, 
        device: torch.device = None,
        sigmas: Optional[torch.Tensor] = None,
        denoising_strength: float = 1.0
    ):

        self.num_inference_steps = num_inference_steps
        
        if sigmas is not None:
            self.sigmas = sigmas.to(device)
        else:

            t = torch.linspace(0, 1, num_inference_steps + 1)
            
            if num_inference_steps <= 10:
                sigmas = 1 - t
            else:
                sigmas = 0.5 * (1 + torch.cos(math.pi * t))
            

            sigmas = self.config.shift * sigmas / (1 + (self.config.shift - 1) * sigmas)
            self.sigmas = sigmas.to(device)
        

        self.timesteps = self.sigmas[:-1] * self.config.num_train_timesteps
        

        self._reset_state()
        
    def _reset_state(self):
        self.velocity_buffer = []
        self.smoothed_velocity = None
        self.step_count = 0
        
    def _get_adaptive_order(self, sigma: float) -> int:
        if not self.use_adaptive_order:
            return self.solver_order
        

        if self.num_inference_steps <= 8:
            return min(2, self.solver_order)

        if sigma > 0.7:
            return min(2, self.solver_order)
        elif sigma > self.convergence_threshold:
            return self.solver_order
        else:
            return max(1, self.solver_order - 1)
    
    def _compute_multistep_velocity(self, order: int) -> torch.Tensor:
        if not self.velocity_buffer:
            raise RuntimeError("velocity_buffer is empty")

        if len(self.velocity_buffer) < order:
            order = len(self.velocity_buffer)


        if order >= 3 and len(self.velocity_buffer) >= 3:
            v = (
                (23/12) * self.velocity_buffer[-1] -
                (16/12) * self.velocity_buffer[-2] +
                (5/12) * self.velocity_buffer[-3]
            )
        elif order >= 2 and len(self.velocity_buffer) >= 2:
            v = 1.5 * self.velocity_buffer[-1] - 0.5 * self.velocity_buffer[-2]
        elif len(self.velocity_buffer) >= 1:
            v = self.velocity_buffer[-1]
        else:
            raise RuntimeError("No velocity data available")

        return v
    
    def _apply_velocity_smoothing(self, velocity: torch.Tensor, sigma: float) -> torch.Tensor:
        if not self.use_velocity_smoothing:
            return velocity
        

        if self.num_inference_steps <= 8:
            return velocity
        

        if sigma < self.convergence_threshold:
            if self.smoothed_velocity is None:
                self.smoothed_velocity = velocity
            else:

                alpha = self.smoothing_factor
                self.smoothed_velocity = alpha * self.smoothed_velocity + (1 - alpha) * velocity
            return self.smoothed_velocity
        else:
            self.smoothed_velocity = velocity
            return velocity
    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[torch.Tensor, float],
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[SchedulerOutput, tuple]:

        if isinstance(timestep, torch.Tensor) and timestep.ndim == 2:
            timestep = timestep.flatten(0, 1)
        

        self.sigmas = self.sigmas.to(model_output.device)
        self.timesteps = self.timesteps.to(model_output.device)
        

        if timestep.ndim == 0:
            timestep_idx = torch.argmin((self.timesteps - timestep).abs())
        else:
            timestep_idx = torch.argmin((self.timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        

        if timestep_idx >= len(self.sigmas):
            raise IndexError(f"timestep_idx {timestep_idx} out of range for sigmas length {len(self.sigmas)}")

        sigma = self.sigmas[timestep_idx]
        if timestep_idx + 1 < len(self.sigmas):
            sigma_next = self.sigmas[timestep_idx + 1]
        else:

            if len(self.sigmas) > 0:
                sigma_next = self.sigmas[-1]
            else:
                raise RuntimeError("sigmas array is empty")
        

        if sigma.ndim == 0:
            sigma = sigma.reshape(-1, 1, 1, 1)
            sigma_next = sigma_next.reshape(-1, 1, 1, 1)
            sigma_val = sigma.item()
        else:
            sigma = sigma.reshape(-1, 1, 1, 1)
            sigma_next = sigma_next.reshape(-1, 1, 1, 1)
            sigma_val = sigma[0].item()
        

        if model_output is not None:
            self.velocity_buffer.append(model_output)

            while len(self.velocity_buffer) > self.solver_order + 1:
                self.velocity_buffer.pop(0)
        else:
            raise ValueError("model_output cannot be None")
        

        current_order = self._get_adaptive_order(sigma_val)
        

        if len(self.velocity_buffer) >= 2:
            velocity = self._compute_multistep_velocity(current_order)
        else:
            velocity = model_output
        

        velocity = self._apply_velocity_smoothing(velocity, sigma_val)
        

        dt = sigma_next - sigma
        

        if self.num_inference_steps > 8 and sigma_val < self.convergence_threshold:

            damping = 0.5 + 0.5 * (sigma_val / self.convergence_threshold)
            dt = dt * damping
        

        prev_sample = sample + velocity * dt
        

        if self.num_inference_steps > 8 and sigma_val < 0.05 and len(self.velocity_buffer) >= 3:
            avg_velocity = sum(self.velocity_buffer[-3:]) / 3
            stabilized = sample + avg_velocity * dt
            blend_factor = sigma_val / 0.05
            prev_sample = blend_factor * prev_sample + (1 - blend_factor) * stabilized
        
        self.step_count += 1
        
        if not return_dict:
            return (prev_sample,)
        
        return SchedulerOutput(prev_sample=prev_sample)
    
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timestep: Union[torch.Tensor, float]
    ) -> torch.Tensor:
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.flatten()
        
        timestep_idx = torch.argmin(
            torch.abs(self.timesteps.unsqueeze(0) - timestep.unsqueeze(1)), dim=1
        )
        sigma = self.sigmas[timestep_idx].reshape(-1, 1, 1, 1)
        
        noisy_samples = (1 - sigma) * original_samples + sigma * noise
        return noisy_samples