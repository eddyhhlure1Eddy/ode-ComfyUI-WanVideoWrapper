"""
SA-ODE Stable - 优化收敛稳定性的SA-Solver ODE版本
基于成功的sa_solver/ode，进一步提升收敛稳定性
"""

import torch
import math
from typing import Optional, Union
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin, SchedulerOutput


class FlowMatchSAODEStableScheduler(SchedulerMixin, ConfigMixin):
    """
    SA-ODE Stable - 稳定收敛版本
    
    核心优化：
    1. 纯确定性ODE（eta=0）
    2. 自适应多步预测
    3. 收敛阶段稳定化
    4. 历史速度平滑
    """
    
    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 3.0,
        solver_order: int = 3,  # 默认三阶
        # 稳定性参数
        use_adaptive_order: bool = True,  # 自适应阶数
        use_velocity_smoothing: bool = True,  # 速度平滑
        convergence_threshold: float = 0.15,  # 收敛阈值（15%时开始稳定化）
        smoothing_factor: float = 0.8,  # 平滑因子
        # 兼容参数
        eta: float = 0.0,  # 保持为0，纯ODE
        use_pece: bool = False,  # 不使用PECE
        predictor_order: int = 3,
        corrector_order: int = 4,
    ):
        self.solver_order = solver_order
        self.use_adaptive_order = use_adaptive_order
        self.use_velocity_smoothing = use_velocity_smoothing
        self.convergence_threshold = convergence_threshold
        self.smoothing_factor = smoothing_factor
        
        # 状态
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
        """设置时间步"""
        self.num_inference_steps = num_inference_steps
        
        if sigmas is not None:
            self.sigmas = sigmas.to(device)
        else:
            # 根据步数选择调度策略
            t = torch.linspace(0, 1, num_inference_steps + 1)
            
            if num_inference_steps <= 10:
                # 低步数：使用简单线性调度，避免复杂变换
                sigmas = 1 - t
            else:
                # 高步数：可以使用更复杂的调度
                # 使用平滑的余弦调度，避免分段不连续
                sigmas = 0.5 * (1 + torch.cos(math.pi * t))
            
            # 应用shift
            sigmas = self.config.shift * sigmas / (1 + (self.config.shift - 1) * sigmas)
            self.sigmas = sigmas.to(device)
        
        # 时间步
        self.timesteps = self.sigmas[:-1] * self.config.num_train_timesteps
        
        # 重置状态
        self._reset_state()
        
    def _reset_state(self):
        """重置内部状态"""
        self.velocity_buffer = []
        self.smoothed_velocity = None
        self.step_count = 0
        
    def _get_adaptive_order(self, sigma: float) -> int:
        """根据当前位置自适应选择阶数"""
        if not self.use_adaptive_order:
            return self.solver_order
        
        # 低步数特殊处理
        if self.num_inference_steps <= 8:
            # 低步数时避免高阶方法
            return min(2, self.solver_order)
        
        # 正常步数的自适应策略
        # 早期：使用低阶（稳定）
        if sigma > 0.7:
            return min(2, self.solver_order)
        # 中期：使用高阶（精确）
        elif sigma > self.convergence_threshold:
            return self.solver_order
        # 后期：降低阶数（稳定收敛）
        else:
            return max(1, self.solver_order - 1)
    
    def _compute_multistep_velocity(self, order: int) -> torch.Tensor:
        """多步预测速度"""
        # 安全检查：确保velocity_buffer不为空
        if not self.velocity_buffer:
            raise RuntimeError("velocity_buffer is empty")

        if len(self.velocity_buffer) < order:
            order = len(self.velocity_buffer)

        # 安全的数组访问
        if order >= 3 and len(self.velocity_buffer) >= 3:
            # 三阶 Adams-Bashforth
            v = (
                (23/12) * self.velocity_buffer[-1] -
                (16/12) * self.velocity_buffer[-2] +
                (5/12) * self.velocity_buffer[-3]
            )
        elif order >= 2 and len(self.velocity_buffer) >= 2:
            # 二阶 Adams-Bashforth
            v = 1.5 * self.velocity_buffer[-1] - 0.5 * self.velocity_buffer[-2]
        elif len(self.velocity_buffer) >= 1:
            # 一阶（直接使用最新速度）
            v = self.velocity_buffer[-1]
        else:
            raise RuntimeError("No velocity data available")

        return v
    
    def _apply_velocity_smoothing(self, velocity: torch.Tensor, sigma: float) -> torch.Tensor:
        """应用速度平滑（稳定收敛）"""
        if not self.use_velocity_smoothing:
            return velocity
        
        # 低步数时禁用平滑
        if self.num_inference_steps <= 8:
            return velocity
        
        # 在收敛阶段应用平滑
        if sigma < self.convergence_threshold:
            if self.smoothed_velocity is None:
                self.smoothed_velocity = velocity
            else:
                # 指数移动平均
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
        """
        执行SA-ODE Stable步骤
        """
        # 处理时间步
        if isinstance(timestep, torch.Tensor) and timestep.ndim == 2:
            timestep = timestep.flatten(0, 1)
        
        # 移动到设备
        self.sigmas = self.sigmas.to(model_output.device)
        self.timesteps = self.timesteps.to(model_output.device)
        
        # 找到索引
        if timestep.ndim == 0:
            timestep_idx = torch.argmin((self.timesteps - timestep).abs())
        else:
            timestep_idx = torch.argmin((self.timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        
        # 获取 sigma - 添加安全检查
        if timestep_idx >= len(self.sigmas):
            raise IndexError(f"timestep_idx {timestep_idx} out of range for sigmas length {len(self.sigmas)}")

        sigma = self.sigmas[timestep_idx]
        if timestep_idx + 1 < len(self.sigmas):
            sigma_next = self.sigmas[timestep_idx + 1]
        else:
            # 安全访问最后一个元素
            if len(self.sigmas) > 0:
                sigma_next = self.sigmas[-1]
            else:
                raise RuntimeError("sigmas array is empty")
        
        # 重塑
        if sigma.ndim == 0:
            sigma = sigma.reshape(-1, 1, 1, 1)
            sigma_next = sigma_next.reshape(-1, 1, 1, 1)
            sigma_val = sigma.item()
        else:
            sigma = sigma.reshape(-1, 1, 1, 1)
            sigma_next = sigma_next.reshape(-1, 1, 1, 1)
            sigma_val = sigma[0].item()
        
        # 存储速度历史 - 添加安全检查
        if model_output is not None:
            self.velocity_buffer.append(model_output)
            # 安全的pop操作
            while len(self.velocity_buffer) > self.solver_order + 1:
                self.velocity_buffer.pop(0)
        else:
            raise ValueError("model_output cannot be None")
        
        # 自适应选择阶数
        current_order = self._get_adaptive_order(sigma_val)
        
        # 多步预测
        if len(self.velocity_buffer) >= 2:
            velocity = self._compute_multistep_velocity(current_order)
        else:
            velocity = model_output
        
        # 收敛阶段的速度平滑
        velocity = self._apply_velocity_smoothing(velocity, sigma_val)
        
        # 步长
        dt = sigma_next - sigma
        
        # 收敛阶段的步长调整（低步数时禁用）
        if self.num_inference_steps > 8 and sigma_val < self.convergence_threshold:
            # 后期使用更小的步长确保稳定
            damping = 0.5 + 0.5 * (sigma_val / self.convergence_threshold)
            dt = dt * damping
        
        # Flow Matching 更新（纯ODE）
        prev_sample = sample + velocity * dt
        
        # 后期稳定化处理（低步数时禁用）
        if self.num_inference_steps > 8 and sigma_val < 0.05 and len(self.velocity_buffer) >= 3:
            # 使用历史平均进行最终收敛
            avg_velocity = sum(self.velocity_buffer[-3:]) / 3
            stabilized = sample + avg_velocity * dt
            # 混合原始和稳定化结果
            blend_factor = sigma_val / 0.05  # 0到1
            prev_sample = blend_factor * prev_sample + (1 - blend_factor) * stabilized
        
        # 更新步数
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
        """添加噪声 - Flow Matching 前向过程"""
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.flatten()
        
        timestep_idx = torch.argmin(
            torch.abs(self.timesteps.unsqueeze(0) - timestep.unsqueeze(1)), dim=1
        )
        sigma = self.sigmas[timestep_idx].reshape(-1, 1, 1, 1)
        
        # Flow Matching: x_t = (1 - σ) * x_0 + σ * noise
        noisy_samples = (1 - sigma) * original_samples + sigma * noise
        return noisy_samples