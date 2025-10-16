"""
Original source from kijai's ComfyUI-WanVideoWrapper
https://github.com/kijai/ComfyUI-WanVideoWrapper/tree/main/wanvideo/schedulers

Modified and optimized by eddy
"""

try:
    from .humo_lcm_scheduler import HumoLCMScheduler, create_humo_lcm_scheduler
except ImportError:
    from humo_lcm_scheduler import HumoLCMScheduler, create_humo_lcm_scheduler


def get_humo_lcm_scheduler(
    steps: int = 4,
    shift: float = 1.0,
    device=None,
    sigmas=None,
    **kwargs
):
    scheduler = create_humo_lcm_scheduler(
        shift=shift,
        **kwargs
    )
    scheduler.set_timesteps(
        steps, 
        device=device, 
        sigmas=sigmas[:-1].tolist() if sigmas is not None else None
    )
    return scheduler


def register_humo_lcm_in_comfyui():
    scheduler_list_addition = [
        "humo_lcm"
    ]
    
    def handle_humo_lcm_scheduler(scheduler, steps, shift, device, sigmas):
        if scheduler == 'humo_lcm':
            return get_humo_lcm_scheduler(
                steps=steps,
                shift=shift,
                device=device,
                sigmas=sigmas
            )
        return None
    
    return scheduler_list_addition, handle_humo_lcm_scheduler


def standalone_usage_example():
    import torch
    
    print("HUMO LCM Scheduler Standalone Usage Example")
    print("=" * 50)
    
    scheduler = get_humo_lcm_scheduler(steps=4)
    
    print(f"Scheduler Type: {type(scheduler).__name__}")
    print(f"Inference Steps: {scheduler.num_inference_steps}")
    print(f"Shift: {scheduler.shift}")
    print(f"Dynamic Boost: {scheduler.config.dynamic_boost}")
    print(f"Contrast Factor: {scheduler.config.contrast_factor}")
    
    print("\nSimulating Sampling Process:")
    device = torch.device("cpu")
    
    sample = torch.randn(1, 4, 64, 64, device=device)
    
    for step_idx in range(scheduler.num_inference_steps):
        timestep = scheduler.timesteps[step_idx]
        model_output = torch.randn_like(sample) * 0.1
        output = scheduler.step(
            model_output=model_output,
            timestep=timestep,
            sample=sample,
            return_dict=True
        )
        sample = output.prev_sample
        print(f"  Step {step_idx}: timestep={timestep:.3f}, "
              f"sample_range=[{sample.min():.3f}, {sample.max():.3f}]")
    
    print(f"\nFinal Output Range: [{sample.min():.3f}, {sample.max():.3f}]")
    print("Sampling Complete!")


HUMO_LCM_CONFIG = {
    "name": "humo_lcm",
    "display_name": "HUMO LCM",
    "description": "Optimized sampler based on LCM+ V2 contrast version",
    "author": "eddy",
    "version": "1.0",
    "base_algorithm": "lcm+/contrast_normal",
    "parameters": {
        "dynamic_boost": 0.15,
        "contrast_factor": 0.95,
        "shift": 1.0,
        "num_train_timesteps": 1000
    },
    "recommended_steps": [4, 6, 8],
    "compatible_models": ["flux", "sd3", "diffusion_models"],
    "features": [
        "Dynamic Range Enhancement",
        "Contrast Optimization",
        "Flow Matching Algorithm",
        "Low-Step Sampling",
        "High Quality Output"
    ]
}


def print_config_info():
    config = HUMO_LCM_CONFIG
    
    print(f"Scheduler Name: {config['display_name']}")
    print(f"Internal Name: {config['name']}")
    print(f"Description: {config['description']}")
    print(f"Author: {config['author']}")
    print(f"Version: {config['version']}")
    print(f"Base Algorithm: {config['base_algorithm']}")
    print()
    
    print("Core Parameters:")
    for key, value in config['parameters'].items():
        print(f"  {key}: {value}")
    print()
    
    print(f"Recommended Steps: {', '.join(map(str, config['recommended_steps']))}")
    print(f"Compatible Models: {', '.join(config['compatible_models'])}")
    print()
    
    print("Features:")
    for feature in config['features']:
        print(f"  - {feature}")


MIGRATION_GUIDE = """
HUMO LCM Scheduler Migration Guide
=====================

File List:
1. humo_lcm_scheduler.py - Core scheduler implementation
2. humo_lcm_integration.py - Integration and usage examples

Migration Steps:

1. Copy Files
   - Copy both files to the target environment's scheduler directory

2. Install Dependencies
   - torch
   - numpy  
   - diffusers

3. Register Scheduler (ComfyUI)
   - Add "humo_lcm" to scheduler_list
   - Add handling logic in get_scheduler function:
   
   elif scheduler == 'humo_lcm':
       from .humo_lcm_integration import get_humo_lcm_scheduler
       sample_scheduler = get_humo_lcm_scheduler(
           steps=steps,
           shift=shift,
           device=device,
           sigmas=sigmas
       )

4. Standalone Usage
   - Import HumoLCMScheduler class directly
   - Use create_humo_lcm_scheduler function to create instance

5. Testing
   - Run standalone_usage_example() to verify functionality
   - Test sampling effects in target environment

Configuration:
- Recommended Steps: 4-8 steps
- Compatible Models: Flux, SD3, other Flow Matching models
- Core Features: Dynamic Range Enhancement + Contrast Optimization

Technical Support:
- Based on lcm+/contrast_normal configuration
- Uses Flow Matching algorithm
- Optimized parameter combination for general scenarios
"""


def print_migration_guide():
    print(MIGRATION_GUIDE)


if __name__ == "__main__":
    print_config_info()
    print("\n" + "=" * 50)
    standalone_usage_example()
    print("\n" + "=" * 50)
    print_migration_guide()
