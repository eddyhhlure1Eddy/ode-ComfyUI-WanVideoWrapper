# Original source: https://github.com/kijai/ComfyUI-WanVideoWrapper/tree/main/wanvideo/schedulers
# Modified and optimized by eddy

import torch
import math
import importlib
from .fm_solvers import (FlowDPMSolverMultistepScheduler, get_sampling_sigmas, retrieve_timesteps)
from .fm_solvers_unipc import FlowUniPCMultistepScheduler
from .basic_flowmatch import FlowMatchScheduler
from .flowmatch_pusa import FlowMatchSchedulerPusa
from .flowmatch_res_multistep import FlowMatchSchedulerResMultistep
from .scheduling_flow_match_lcm import FlowMatchLCMScheduler
from .flowmatch_sa_ode_stable import FlowMatchSAODEStableScheduler
from .humo_lcm_integration import get_humo_lcm_scheduler
from .fm_rcm import rCMFlowMatchScheduler
from .flowmatch_lowstep_d import FlowMatchLowStepScheduler
try:
    from .iching_wuxing_scheduler_core import IChingWuxingScheduler
except ImportError:
    IChingWuxingScheduler = None
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler, DEISMultistepScheduler
try:
    from ...utils import log
except ImportError:
    class SimpleLog:
        def info(self, msg): print(f"INFO: {msg}")
    log = SimpleLog()

scheduler_list = [
    "unipc", "unipc/beta",
    "dpm++", "dpm++/beta",
    "dpm++_sde", "dpm++_sde/beta",
    "euler", "euler/beta",
    "deis",
    "lcm", "lcm/beta",
    "res_multistep",
    "flowmatch_causvid",
    "flowmatch_distill",
    "flowmatch_pusa",
    "flowmatch_lowstep_d",
    "flowmatch_sa_ode_stable",
    "sa_ode_stable/lowstep",
    "ode/+",
    "humo_lcm",
    "multitalk",
    "iching/wuxing",
    "iching/wuxing-strong",
    "iching/wuxing-stable",
    "iching/wuxing-smooth",
    "iching/wuxing-clean",
    "iching/wuxing-sharp",
    "iching/wuxing-lowstep",
    "rcm"
]

def get_scheduler(scheduler, steps, start_step, end_step, shift, device, transformer_dim=5120, flowedit_args=None, denoise_strength=1.0, sigmas=None, log_timesteps=False):
    timesteps = None
    if 'unipc' in scheduler:
        sample_scheduler = FlowUniPCMultistepScheduler(shift=shift)
        if sigmas is None:
            sample_scheduler.set_timesteps(steps, device=device, shift=shift, use_beta_sigmas=('beta' in scheduler))
        else:
            sample_scheduler.sigmas = sigmas.to(device)
            sample_scheduler.timesteps = (sample_scheduler.sigmas[:-1] * 1000).to(torch.int64).to(device)
            sample_scheduler.num_inference_steps = len(sample_scheduler.timesteps)

    elif scheduler in ['euler/beta', 'euler']:
        sample_scheduler = FlowMatchEulerDiscreteScheduler(shift=shift, use_beta_sigmas=(scheduler == 'euler/beta'))
        if flowedit_args:
            timesteps, _ = retrieve_timesteps(sample_scheduler, device=device, sigmas=get_sampling_sigmas(steps, shift))
        else:
            sample_scheduler.set_timesteps(steps, device=device, sigmas=sigmas[:-1].tolist() if sigmas is not None else None)
    elif 'dpm' in scheduler:
        if 'sde' in scheduler:
            algorithm_type = "sde-dpmsolver++"
        else:
            algorithm_type = "dpmsolver++"
        sample_scheduler = FlowDPMSolverMultistepScheduler(shift=shift, algorithm_type=algorithm_type)
        if sigmas is None:
            sample_scheduler.set_timesteps(steps, device=device, use_beta_sigmas=('beta' in scheduler))
        else:
            sample_scheduler.sigmas = sigmas.to(device)
            sample_scheduler.timesteps = (sample_scheduler.sigmas[:-1] * 1000).to(torch.int64).to(device)
            sample_scheduler.num_inference_steps = len(sample_scheduler.timesteps)
    elif scheduler == 'deis':
        sample_scheduler = DEISMultistepScheduler(use_flow_sigmas=True, prediction_type="flow_prediction", flow_shift=shift)
        sample_scheduler.set_timesteps(steps, device=device)
        sample_scheduler.sigmas[-1] = 1e-6
    elif 'lcm' in scheduler:
        sample_scheduler = FlowMatchLCMScheduler(shift=shift, use_beta_sigmas=(scheduler == 'lcm/beta'))
        sample_scheduler.set_timesteps(steps, device=device, sigmas=sigmas[:-1].tolist() if sigmas is not None else None)
    elif 'flowmatch_causvid' in scheduler:
        if sigmas is not None:
            raise NotImplementedError("This scheduler does not support custom sigmas")
        if transformer_dim == 5120:
            denoising_list = [999, 934, 862, 756, 603, 410, 250, 140, 74]
        else:
            if steps != 4:
                raise ValueError("CausVid 1.3B schedule is only for 4 steps")
            denoising_list = [1000, 750, 500, 250]
        sample_scheduler = FlowMatchScheduler(num_inference_steps=steps, shift=shift, sigma_min=0, extra_one_step=True)
        sample_scheduler.timesteps = torch.tensor(denoising_list)[:steps].to(device)
        sample_scheduler.sigmas = torch.cat([sample_scheduler.timesteps / 1000, torch.tensor([0.0], device=device)])
    elif 'flowmatch_distill' in scheduler:
        if sigmas is not None:
            raise NotImplementedError("This scheduler does not support custom sigmas")
        sample_scheduler = FlowMatchScheduler(
            shift=shift, sigma_min=0.0, extra_one_step=True
        )
        sample_scheduler.set_timesteps(1000, training=True)
    
        denoising_step_list = torch.tensor([999, 750, 500, 250] , dtype=torch.long)
        temp_timesteps = torch.cat((sample_scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
        denoising_step_list = temp_timesteps[1000 - denoising_step_list]

        
        if steps != 4:
            raise ValueError("This scheduler is only for 4 steps")
        
        sample_scheduler.timesteps = denoising_step_list[:steps].clone().detach().to(device)
        sample_scheduler.sigmas = torch.cat([sample_scheduler.timesteps / 1000, torch.tensor([0.0], device=device)])
    elif 'flowmatch_pusa' in scheduler:
        sample_scheduler = FlowMatchSchedulerPusa(shift=shift, sigma_min=0.0, extra_one_step=True)
        sample_scheduler.set_timesteps(steps+1, denoising_strength=denoise_strength, shift=shift,
                                       sigmas=sigmas[:-1].tolist() if sigmas is not None else None)
    elif scheduler == 'res_multistep':
        sample_scheduler = FlowMatchSchedulerResMultistep(shift=shift)
        sample_scheduler.set_timesteps(steps, denoising_strength=denoise_strength, sigmas=sigmas[:-1].tolist() if sigmas is not None else None)
    elif scheduler == 'flowmatch_lowstep_d':
        sample_scheduler = FlowMatchLowStepScheduler(shift=shift)
        sample_scheduler.set_timesteps(steps, device=device, sigmas=sigmas[:-1].tolist() if sigmas is not None else None)
    elif scheduler in ['flowmatch_sa_ode_stable', 'sa_ode_stable/lowstep']:
        sample_scheduler = FlowMatchSAODEStableScheduler(shift=shift)
        sample_scheduler.set_timesteps(steps, device=device, sigmas=sigmas[:-1].tolist() if sigmas is not None else None)
    elif scheduler == 'ode/+':
        # Tuned ODE+ profile (8-step optimized): linear sigmas + mild jitter at last 2 steps
        order = 2  # AB2 at <=8 steps
        sample_scheduler = FlowMatchLowStepScheduler(
            shift=shift,
            solver_order=order,
        )
        # build linear sigmas with small jitter on last 2 steps
        t = torch.linspace(0, 1, steps + 1, device=device, dtype=torch.float32)
        custom_sigmas = 1 - t
        s = shift
        if abs(s - 1.0) > 1e-6:
            custom_sigmas = (s * custom_sigmas) / (1 + (s - 1) * custom_sigmas)
        # jitter: amplitude 0.006 on last 2 steps
        sj_amp = 0.006
        sj_fw = 2
        if sj_amp > 0 and steps >= 2:
            N = steps + 1
            idx = torch.arange(N, device=device, dtype=custom_sigmas.dtype)
            base = torch.sin(math.pi * idx / max(1, N - 1))
            start = max(0, N - sj_fw - 1)
            ramp = torch.zeros_like(base)
            ramp[start:] = torch.linspace(0.0, 1.0, N - start, device=device, dtype=custom_sigmas.dtype)
            delta = sj_amp * base * ramp
            custom_sigmas = custom_sigmas + delta
            custom_sigmas[-1] = torch.tensor(0.0, device=device, dtype=custom_sigmas.dtype)
            custom_sigmas = torch.clamp(custom_sigmas, 0.0, 1.0)
            for i in range(1, N):
                prev = custom_sigmas[i-1]
                custom_sigmas[i] = torch.minimum(custom_sigmas[i], prev)
        try:
            log.info(f"[ode/+] order={order} sj_amp=0.006 sj_fw=2 sigmas[0]={float(custom_sigmas[0])} sigmas[-1]={float(custom_sigmas[-1])}")
        except Exception:
            pass
        sample_scheduler.set_timesteps(steps, device=device, sigmas=custom_sigmas)
    elif scheduler == 'humo_lcm':
        sample_scheduler = get_humo_lcm_scheduler(
            steps=steps,
            shift=shift,
            device=device,
            sigmas=sigmas
        )
    elif scheduler.startswith('iching/'):
        if IChingWuxingScheduler is None:
            raise ImportError("IChingWuxingScheduler is not available. The compiled module may be missing.")
        sample_scheduler = IChingWuxingScheduler(mode=scheduler)
        sample_scheduler.set_timesteps(steps, device=device)
    elif scheduler == 'rcm':
        sample_scheduler = rCMFlowMatchScheduler(num_inference_steps=steps)
        sample_scheduler.timesteps = sample_scheduler.timesteps.to(device)
        sample_scheduler.sigmas = sample_scheduler.sigmas.to(device)
    if timesteps is None:
        timesteps = sample_scheduler.timesteps

    steps = len(timesteps)
    if (isinstance(start_step, int) and end_step != -1 and start_step >= end_step) or (not isinstance(start_step, int) and start_step != -1 and end_step >= start_step):
        raise ValueError("start_step must be less than end_step")


    start_idx = 0
    end_idx = len(timesteps) - 1

    if log_timesteps:
        log.info(f"------- Scheduler info -------")
        log.info(f"Total timesteps: {timesteps}")

    if isinstance(start_step, float):
        idxs = (sample_scheduler.sigmas <= start_step).nonzero(as_tuple=True)[0]
        if len(idxs) > 0:
            start_idx = idxs[0].item()
    elif isinstance(start_step, int):
        if start_step > 0:
            start_idx = start_step

    if isinstance(end_step, float):
        idxs = (sample_scheduler.sigmas >= end_step).nonzero(as_tuple=True)[0]
        if len(idxs) > 0:
            end_idx = idxs[-1].item()
    elif isinstance(end_step, int):
        if end_step != -1:
            end_idx = end_step - 1


    timesteps = timesteps[start_idx:end_idx+1]
    sample_scheduler.full_sigmas = sample_scheduler.sigmas.clone()
    sample_scheduler.sigmas = sample_scheduler.sigmas[start_idx:start_idx+len(timesteps)+1]
    
    if log_timesteps:
        log.info(f"Using timesteps: {timesteps}")
        log.info(f"Using sigmas: {sample_scheduler.sigmas}")
        log.info(f"------------------------------")

    if hasattr(sample_scheduler, 'timesteps'):
        sample_scheduler.timesteps = timesteps

    return sample_scheduler, timesteps, start_idx, end_idx