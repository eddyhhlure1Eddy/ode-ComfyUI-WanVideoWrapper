# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math
import torch
import torch.nn as nn
from einops import repeat, rearrange
from ...enhance_a_video.enhance import get_feta_scores
import time
from contextlib import nullcontext

try:
    from ..radial_attention.attn_mask import RadialSpargeSageAttn, RadialSpargeSageAttnDense, MaskMap
except:
    pass

from .attention import attention
import numpy as np
from tqdm import tqdm
import gc

from ...utils import log, get_module_memory_mb
from ...cache_methods.cache_methods import TeaCacheState, MagCacheState, EasyCacheState, relative_l1_distance
from ...multitalk.multitalk import get_attn_map_with_target
from ...echoshot.echoshot import rope_apply_z, rope_apply_c, rope_apply_echoshot

from ...MTV.mtv import apply_rotary_emb

class FramePackMotioner(nn.Module):#from comfy.ldm.wan.model
    def __init__(
            self,
            inner_dim=1024,
            num_heads=16,  # Used to indicate the number of heads in the backbone network; unrelated to this module's design
            zip_frame_buckets=[1, 2, 16],  # Three numbers representing the number of frames sampled for patch operations from the nearest to the farthest frames
            drop_mode="drop",  # If not "drop", it will use "padd", meaning padding instead of deletion
            ):
        super().__init__()
        self.proj = nn.Conv3d(16, inner_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.proj_2x = nn.Conv3d(16, inner_dim, kernel_size=(2, 4, 4), stride=(2, 4, 4))
        self.proj_4x = nn.Conv3d(16, inner_dim, kernel_size=(4, 8, 8), stride=(4, 8, 8))
        self.zip_frame_buckets = zip_frame_buckets

        self.inner_dim = inner_dim
        self.num_heads = num_heads
        self.drop_mode = drop_mode

    def forward(self, motion_latents, rope_embedder, add_last_motion=2):
        lat_height, lat_width = motion_latents.shape[3], motion_latents.shape[4]
        padd_lat = torch.zeros(motion_latents.shape[0], 16, sum(self.zip_frame_buckets), lat_height, lat_width).to(device=motion_latents.device, dtype=motion_latents.dtype)
        overlap_frame = min(padd_lat.shape[2], motion_latents.shape[2])
        if overlap_frame > 0:
            padd_lat[:, :, -overlap_frame:] = motion_latents[:, :, -overlap_frame:]

        if add_last_motion < 2 and self.drop_mode != "drop":
            zero_end_frame = sum(self.zip_frame_buckets[:len(self.zip_frame_buckets) - add_last_motion - 1])
            padd_lat[:, :, -zero_end_frame:] = 0

        clean_latents_4x, clean_latents_2x, clean_latents_post = padd_lat[:, :, -sum(self.zip_frame_buckets):, :, :].split(self.zip_frame_buckets[::-1], dim=2)  # 16, 2 ,1

        # patchfy
        clean_latents_post = self.proj(clean_latents_post).flatten(2).transpose(1, 2)
        clean_latents_2x = self.proj_2x(clean_latents_2x)
        l_2x_shape = clean_latents_2x.shape
        clean_latents_2x = clean_latents_2x.flatten(2).transpose(1, 2)
        clean_latents_4x = self.proj_4x(clean_latents_4x)
        l_4x_shape = clean_latents_4x.shape
        clean_latents_4x = clean_latents_4x.flatten(2).transpose(1, 2)

        if add_last_motion < 2 and self.drop_mode == "drop":
            clean_latents_post = clean_latents_post[:, :0] if add_last_motion < 2 else clean_latents_post
            clean_latents_2x = clean_latents_2x[:, :0] if add_last_motion < 1 else clean_latents_2x

        motion_lat = torch.cat([clean_latents_post, clean_latents_2x, clean_latents_4x], dim=1)

        rope_post = rope_embedder.rope_encode_comfy(1, lat_height, lat_width, t_start=-1, device=motion_latents.device, dtype=motion_latents.dtype)
        rope_2x = rope_embedder.rope_encode_comfy(1, lat_height, lat_width, t_start=-3, steps_h=l_2x_shape[-2], steps_w=l_2x_shape[-1], device=motion_latents.device, dtype=motion_latents.dtype)
        rope_4x = rope_embedder.rope_encode_comfy(4, lat_height, lat_width, t_start=-19, steps_h=l_4x_shape[-2], steps_w=l_4x_shape[-1], device=motion_latents.device, dtype=motion_latents.dtype)

        rope = torch.cat([rope_post, rope_2x, rope_4x], dim=1)
        return motion_lat, rope

from diffusers.models.attention import AdaLayerNorm

__all__ = ['WanModel']

from comfy import model_management as mm


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def torch_dfs(model: nn.Module, parent_name='root'):
    module_names, modules = [], []
    current_name = parent_name if parent_name else 'root'
    module_names.append(current_name)
    modules.append(model)

    for name, child in model.named_children():
        if parent_name:
            child_name = f'{parent_name}.{name}'
        else:
            child_name = name
        child_modules, child_names = torch_dfs(child, child_name)
        module_names += child_names
        modules += child_modules
    return modules, module_names

#from comfy.ldm.flux.math import apply_rope as apply_rope_comfy
def apply_rope_comfy(xq, xk, freqs_cis):    
    xq_ = xq.to(dtype=freqs_cis.dtype).reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.to(dtype=freqs_cis.dtype).reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)

def apply_rope_comfy_chunked(xq, xk, freqs_cis, num_chunks=4):
    seq_dim = 1
    
    # Initialize output tensors
    xq_out = torch.empty_like(xq)
    xk_out = torch.empty_like(xk)
    
    # Calculate chunks
    seq_len = xq.shape[seq_dim]
    chunk_sizes = [seq_len // num_chunks + (1 if i < seq_len % num_chunks else 0) 
                  for i in range(num_chunks)]
    
    # First pass: process xq completely
    start_idx = 0
    for size in chunk_sizes:
        end_idx = start_idx + size
        
        slices = [slice(None)] * len(xq.shape)
        slices[seq_dim] = slice(start_idx, end_idx)
        
        freq_slices = [slice(None)] * len(freqs_cis.shape)
        if seq_dim < len(freqs_cis.shape):
            freq_slices[seq_dim] = slice(start_idx, end_idx)
        freqs_chunk = freqs_cis[tuple(freq_slices)]
        
        xq_chunk = xq[tuple(slices)]
        xq_chunk_ = xq_chunk.to(dtype=freqs_cis.dtype).reshape(*xq_chunk.shape[:-1], -1, 1, 2)
        xq_out[tuple(slices)] = (freqs_chunk[..., 0] * xq_chunk_[..., 0] + 
                                freqs_chunk[..., 1] * xq_chunk_[..., 1]).reshape(*xq_chunk.shape).type_as(xq)
        
        del xq_chunk, xq_chunk_, freqs_chunk
        start_idx = end_idx
    
    # Second pass: process xk completely
    start_idx = 0
    for size in chunk_sizes:
        end_idx = start_idx + size
        
        slices = [slice(None)] * len(xk.shape)
        slices[seq_dim] = slice(start_idx, end_idx)
        
        freq_slices = [slice(None)] * len(freqs_cis.shape)
        if seq_dim < len(freqs_cis.shape):
            freq_slices[seq_dim] = slice(start_idx, end_idx)
        freqs_chunk = freqs_cis[tuple(freq_slices)]
        
        xk_chunk = xk[tuple(slices)]
        xk_chunk_ = xk_chunk.to(dtype=freqs_cis.dtype).reshape(*xk_chunk.shape[:-1], -1, 1, 2)
        xk_out[tuple(slices)] = (freqs_chunk[..., 0] * xk_chunk_[..., 0] + 
                                freqs_chunk[..., 1] * xk_chunk_[..., 1]).reshape(*xk_chunk.shape).type_as(xk)
        
        del xk_chunk, xk_chunk_, freqs_chunk
        start_idx = end_idx
    
    return xq_out, xk_out

def rope_riflex(pos, dim, i, theta, L_test, k, ntk_factor=1.0):
    assert dim % 2 == 0
    if mm.is_device_mps(pos.device) or mm.is_intel_xpu() or mm.is_directml_enabled():
        device = torch.device("cpu")
    else:
        device = pos.device

    if ntk_factor != 1.0:
        theta *= ntk_factor

    scale = torch.linspace(0, (dim - 2) / dim, steps=dim//2, dtype=torch.float64, device=device)
    omega = 1.0 / (theta**scale)

    # RIFLEX modification - adjust last frequency component if L_test and k are provided
    if i==0 and k > 0 and L_test:
        omega[k-1] = 0.9 * 2 * torch.pi / L_test

    out = torch.einsum("...n,d->...nd", pos.to(dtype=torch.float32, device=device), omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.to(dtype=torch.float32, device=pos.device)

class EmbedND_RifleX(nn.Module):
    def __init__(self, dim, theta, axes_dim, num_frames, k):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim
        self.num_frames = num_frames
        self.k = k

    def forward(self, ids, ntk_factor=[1.0,1.0,1.0]):
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope_riflex(
                ids[..., i], 
                self.axes_dim[i], 
                i, #f h w
                self.theta, 
                self.num_frames, 
                self.k,
                ntk_factor[i])
            for i in range(n_axes)],
            dim=-3,
        )
        return emb.unsqueeze(1)

def poly1d(coefficients, x):
    result = torch.zeros_like(x)
    for i, coeff in enumerate(coefficients):
        result += coeff * (x ** (len(coefficients) - 1 - i))
    return result.abs()

def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float32)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x

def rope_params(max_seq_len, dim, theta=10000, L_test=25, k=0):
    assert dim % 2 == 0
    exponents = torch.arange(0, dim, 2, dtype=torch.float64).div(dim)
    inv_theta_pow = 1.0 / torch.pow(theta, exponents)
    
    if k > 0:
        print(f"RifleX: Using {k}th freq")
        inv_theta_pow[k-1] = 0.9 * 2 * torch.pi / L_test
        
    freqs = torch.outer(torch.arange(max_seq_len), inv_theta_pow)
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs

@torch.autocast(device_type=mm.get_autocast_device(mm.get_torch_device()), enabled=False)
@torch.compiler.disable()
def rope_apply(x, grid_sizes, freqs, reverse_time=False):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        if reverse_time:
            time_freqs = freqs[0][:f].view(f, 1, 1, -1)
            time_freqs = torch.flip(time_freqs, dims=[0])
            time_freqs = time_freqs.expand(f, h, w, -1)
            
            spatial_freqs = torch.cat([
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
            ], dim=-1)
            
            freqs_i = torch.cat([time_freqs, spatial_freqs], dim=-1).reshape(seq_len, 1, -1)
        else:
            freqs_i = torch.cat([
                freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
            ],
                                dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).to(x.dtype)


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x, num_chunks=1):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        use_chunked = num_chunks > 1
        if use_chunked:
            return self.forward_chunked(x, num_chunks)
        else:
            return self._norm(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps).to(x.dtype)

    def forward_chunked(self, x, num_chunks=4):
        output = torch.empty_like(x)
        
        chunk_sizes = [x.shape[1] // num_chunks + (1 if i < x.shape[1] % num_chunks else 0) 
                    for i in range(num_chunks)]
        
        start_idx = 0
        for size in chunk_sizes:
            end_idx = start_idx + size
            
            chunk = x[:, start_idx:end_idx, :]
            
            norm_factor = torch.rsqrt(chunk.pow(2).mean(dim=-1, keepdim=True) + self.eps)
            output[:, start_idx:end_idx, :] = chunk * norm_factor.to(chunk.dtype) * self.weight

            start_idx = end_idx
            
        return output
    
class WanFusedRMSNorm(nn.RMSNorm):
    def forward(self, x, num_chunks=1):
        use_chunked = num_chunks > 1
        if use_chunked:
            return self.forward_chunked(x, num_chunks)
        else:
            return super().forward(x)

    def forward_chunked(self, x, num_chunks=4):
        output = torch.empty_like(x)
        
        chunk_sizes = [x.shape[1] // num_chunks + (1 if i < x.shape[1] % num_chunks else 0) 
                    for i in range(num_chunks)]
        
        start_idx = 0
        for size in chunk_sizes:
            end_idx = start_idx + size
            chunk = x[:, start_idx:end_idx, :]
            output[:, start_idx:end_idx, :] = super().forward(chunk)
            start_idx = end_idx
            
        return output


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x)


class WanSelfAttention(nn.Module):

    def __init__(self,
                 in_features,
                 out_features,
                 num_heads,
                 qk_norm=True,
                 eps=1e-6,
                 attention_mode="sdpa",
                 rms_norm_function="default",
                 kv_dim=None):
        assert out_features % num_heads == 0
        super().__init__()
        self.dim = out_features
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads
        self.qk_norm = qk_norm
        self.eps = eps
        self.attention_mode = attention_mode

        #radial attention
        self.mask_map = None
        self.decay_factor = 0.2
        self.cond_size = None

        # layers
        self.q = nn.Linear(in_features, out_features)
        if kv_dim is not None:
            self.k = nn.Linear(kv_dim, out_features)
            self.v = nn.Linear(kv_dim, out_features)
        else:
            self.k = nn.Linear(in_features, out_features)
            self.v = nn.Linear(in_features, out_features)
        self.o = nn.Linear(in_features, out_features)

        if rms_norm_function=="pytorch":
            self.norm_q = WanFusedRMSNorm(out_features, eps=eps) if qk_norm else nn.Identity()
            self.norm_k = WanFusedRMSNorm(out_features, eps=eps) if qk_norm else nn.Identity()
        else:
            self.norm_q = WanRMSNorm(out_features, eps=eps) if qk_norm else nn.Identity()
            self.norm_k = WanRMSNorm(out_features, eps=eps) if qk_norm else nn.Identity()

    def qkv_fn(self, x):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        return q, k, v
    
    def qkv_fn_ip(self, x):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        q = self.norm_q(self.q(x) + self.q_loras(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x) + self.k_loras(x)).view(b, s, n, d)
        v = (self.v(x) + self.v_loras(x)).view(b, s, n, d)
        return q, k, v

    def forward(self, q, k, v, seq_lens, attention_mode_override=None):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        attention_mode = self.attention_mode
        if attention_mode_override is not None:
            attention_mode = attention_mode_override

        x = attention(q, k, v, k_lens=seq_lens, attention_mode=attention_mode)
        return self.o(x.flatten(2))
    
    def forward_ip(self, q, k, v, q_ip, k_ip, v_ip, seq_lens, attention_mode_override=None):
        attention_mode = self.attention_mode
        if attention_mode_override is not None:
            attention_mode = attention_mode_override
        
        # Concatenate main and IP keys/values for main attention
        full_k = torch.cat([k, k_ip], dim=1)
        full_v = torch.cat([v, v_ip], dim=1)
        main_out = attention(q, full_k, full_v, k_lens=seq_lens, attention_mode=attention_mode)
        
        cond_out = attention(q_ip, k_ip, v_ip, k_lens=seq_lens, attention_mode=attention_mode)
        x = torch.cat([main_out, cond_out], dim=1)

        return self.o(x.flatten(2))
   
    
    def forward_radial(self, q, k, v, dense_step=False):
        if dense_step:
            x = RadialSpargeSageAttnDense(q, k, v, self.mask_map)
        else:
            x = RadialSpargeSageAttn(q, k, v, self.mask_map, decay_factor=self.decay_factor)
        return self.o(x.flatten(2))
    
    
    def forward_multitalk(self, q, k, v, seq_lens, grid_sizes, ref_target_masks):
        x = attention(
            q, k, v,
            k_lens=seq_lens,
            attention_mode=self.attention_mode
            )

        # output
        x = x.flatten(2)
        x = self.o(x)

        x_ref_attn_map = get_attn_map_with_target(q.type_as(x), k.type_as(x), grid_sizes[0], ref_target_masks=ref_target_masks)

        return x, x_ref_attn_map
    
    
    def forward_split(self, q, k, v, seq_lens, grid_sizes, freqs, seq_chunks=1,current_step=0, video_attention_split_steps = []):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """

        # Split by frames if multiple prompts are provided
        if seq_chunks > 1 and current_step in video_attention_split_steps:
            outputs = []
            # Extract frame, height, width from grid_sizes
            frames = grid_sizes[0][0]
            height = grid_sizes[0][1]
            width = grid_sizes[0][2]
            tokens_per_frame = height * width
            
            actual_chunks = torch.min(torch.tensor(seq_chunks, device=frames.device), frames)
            base_frames_per_chunk = frames // actual_chunks
            extra_frames = frames % actual_chunks
            
            # Calculate all chunk boundaries
            chunk_indices = torch.arange(actual_chunks, device=frames.device)
            chunk_sizes = base_frames_per_chunk + (chunk_indices < extra_frames).long()
            chunk_starts = torch.cumsum(torch.cat([torch.zeros(1, device=frames.device), chunk_sizes[:-1]]), dim=0).long()
            chunk_ends = chunk_starts + chunk_sizes
            
            # Process each chunk using tensor indexing
            for i in range(actual_chunks.item()):
                start_frame = chunk_starts[i]
                end_frame = chunk_ends[i]
                
                # Convert to token indices using tensor operations
                start_idx = start_frame * tokens_per_frame
                end_idx = end_frame * tokens_per_frame
                
                chunk_q = q[:, start_idx:end_idx, :, :]
                chunk_k = k[:, start_idx:end_idx, :, :]
                chunk_v = v[:, start_idx:end_idx, :, :]
                
                chunk_out = attention(
                    q=chunk_q,
                    k=chunk_k,
                    v=chunk_v,
                    k_lens=seq_lens,
                    attention_mode=self.attention_mode)
                
                outputs.append(chunk_out)
            
            # Concatenate outputs along the sequence dimension
            x = torch.cat(outputs, dim=1)
        else:
            # Original attention computation
            x = attention(
                q=q,
                k=k,
                v=v,
                k_lens=seq_lens,
                attention_mode=self.attention_mode)

        # output
        x = x.flatten(2)
        x = self.o(x)

        return x
    
    def normalized_attention_guidance(self, b, n, d, q, context, nag_context=None, nag_params={}):
        # NAG text attention
        context_positive = context
        context_negative = nag_context
        nag_scale = nag_params['nag_scale']
        nag_alpha = nag_params['nag_alpha']
        nag_tau = nag_params['nag_tau']

        k_positive = self.norm_k(self.k(context_positive)).view(b, -1, n, d)
        v_positive = self.v(context_positive).view(b, -1, n, d)
        k_negative = self.norm_k(self.k(context_negative)).view(b, -1, n, d)
        v_negative = self.v(context_negative).view(b, -1, n, d)

        x_positive = attention(q, k_positive, v_positive, attention_mode=self.attention_mode)
        x_positive = x_positive.flatten(2)

        x_negative = attention(q, k_negative, v_negative, attention_mode=self.attention_mode)
        x_negative = x_negative.flatten(2)

        nag_guidance = x_positive * nag_scale - x_negative * (nag_scale - 1)
        
        norm_positive = torch.norm(x_positive, p=1, dim=-1, keepdim=True)
        norm_guidance = torch.norm(nag_guidance, p=1, dim=-1, keepdim=True)
        
        scale = norm_guidance / norm_positive
        scale = torch.nan_to_num(scale, nan=10.0)
        
        mask = scale > nag_tau
        adjustment = (norm_positive * nag_tau) / (norm_guidance + 1e-7)
        nag_guidance = torch.where(mask, nag_guidance * adjustment, nag_guidance)
        del mask, adjustment
        
        return nag_guidance * nag_alpha + x_positive * (1 - nag_alpha)

class LoRALinearLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 128,
        device=torch.device("cuda"),
        dtype=torch.float32,
        strength: float = 1.0
    ):
        super().__init__()
        self.down = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self.up = nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)
        self.rank = rank
        self.out_features = out_features
        self.in_features = in_features
        self.strength = strength

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states) * self.strength
        return up_hidden_states.to(orig_dtype)

#region crossattn
class WanT2VCrossAttention(WanSelfAttention):

    def __init__(self, in_features, out_features, num_heads, kv_dim=None, qk_norm=True, eps=1e-6, attention_mode='sdpa', rms_norm_function="default"):
        super().__init__(in_features, out_features, num_heads, qk_norm, eps, kv_dim=kv_dim, rms_norm_function=rms_norm_function)
        self.attention_mode = attention_mode

    def forward(self, x, context, grid_sizes=None, clip_embed=None, audio_proj=None, audio_scale=1.0, 
                num_latent_frames=21, nag_params={}, nag_context=None, is_uncond=False, rope_func="comfy", 
                inner_t=None, inner_c=None, cross_freqs=None,
                adapter_proj=None, adapter_attn_mask=None, ip_scale=1.0, orig_seq_len=None, **kwargs):
        b, n, d = x.size(0), self.num_heads, self.head_dim
        # compute query
        q = self.norm_q(self.q(x),num_chunks=2 if rope_func == "comfy_chunked" else 1).view(b, -1, n, d)

        if nag_context is not None and not is_uncond:
            x_text = self.normalized_attention_guidance(b, n, d, q, context, nag_context, nag_params)
        else:
            k = self.norm_k(self.k(context)).view(b, -1, n, d)
            v = self.v(context).view(b, -1, n, d)

            #EchoShot rope
            if inner_t is not None and cross_freqs is not None and not is_uncond:
                q = rope_apply_z(q, grid_sizes, cross_freqs, inner_t).to(q)
                k = rope_apply_c(k, cross_freqs, inner_c).to(q)

            x_text = attention(q, k, v, attention_mode=self.attention_mode)
            x_text = x_text.flatten(2)

        x = x_text

        # FantasyTalking audio attention
        if audio_proj is not None:
            if len(audio_proj.shape) == 4:
                audio_q = q.view(b * num_latent_frames, -1, n, d)
                ip_key = self.k_proj(audio_proj).view(b * num_latent_frames, -1, n, d)
                ip_value = self.v_proj(audio_proj).view(b * num_latent_frames, -1, n, d)
                audio_x = attention(audio_q, ip_key, ip_value, attention_mode=self.attention_mode)
                audio_x = audio_x.view(b, q.size(1), n, d).flatten(2)
            elif len(audio_proj.shape) == 3:
                ip_key = self.k_proj(audio_proj).view(b, -1, n, d)
                ip_value = self.v_proj(audio_proj).view(b, -1, n, d)
                audio_x = attention(q, ip_key, ip_value, attention_mode=self.attention_mode).flatten(2)
            x = x + audio_x * audio_scale

        # FantasyPortrait adapter attention
        if adapter_proj is not None:
            if len(adapter_proj.shape) == 4:
                q_in = q[:, :orig_seq_len]                
                adapter_q = q_in.view(b * num_latent_frames, -1, n, d)
                ip_key = self.ip_adapter_single_stream_k_proj(adapter_proj).view(b * num_latent_frames, -1, n, d)
                ip_value = self.ip_adapter_single_stream_v_proj(adapter_proj).view(b * num_latent_frames, -1, n, d)

                adapter_x = attention(adapter_q, ip_key, ip_value, attention_mode=self.attention_mode)
                adapter_x = adapter_x.view(b, q_in.size(1), n, d)
                adapter_x = adapter_x.flatten(2)
            elif len(adapter_proj.shape) == 3:
                ip_key = self.ip_adapter_single_stream_k_proj(adapter_proj).view(b, -1, n, d)
                ip_value = self.ip_adapter_single_stream_v_proj(adapter_proj).view(b, -1, n, d)
                adapter_x = attention(q_in, ip_key, ip_value, attention_mode=self.attention_mode)
                adapter_x = adapter_x.flatten(2)
            x[:, :orig_seq_len] = x[:, :orig_seq_len] + adapter_x * ip_scale

        return self.o(x)


class WanI2VCrossAttention(WanSelfAttention):

    def __init__(self, in_features, out_features, num_heads, qk_norm=True, eps=1e-6, attention_mode='sdpa', rms_norm_function="default"):
        super().__init__(in_features, out_features, num_heads, qk_norm, eps, rms_norm_function=rms_norm_function)
        self.k_img = nn.Linear(in_features, out_features)
        self.v_img = nn.Linear(in_features, out_features)
        self.norm_k_img = WanRMSNorm(out_features, eps=eps) if qk_norm else nn.Identity()
        self.attention_mode = attention_mode

    def forward(self, x, context, grid_sizes=None, clip_embed=None, audio_proj=None, 
                audio_scale=1.0, num_latent_frames=21, nag_params={}, nag_context=None, is_uncond=False, rope_func="comfy", 
                adapter_proj=None, adapter_attn_mask=None, ip_scale=1.0, orig_seq_len=None, **kwargs):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim
        # compute query
        q = self.norm_q(self.q(x),num_chunks=2 if rope_func == "comfy_chunked" else 1).view(b, -1, n, d)

        if nag_context is not None and not is_uncond:
            x_text = self.normalized_attention_guidance(b, n, d, q, context, nag_context, nag_params)
        else:
            # text attention
            k = self.norm_k(self.k(context)).view(b, -1, n, d)
            v = self.v(context).view(b, -1, n, d)
            x_text = attention(q, k, v, attention_mode=self.attention_mode).flatten(2)

        #img attention
        if clip_embed is not None:
            k_img = self.norm_k_img(self.k_img(clip_embed)).view(b, -1, n, d)
            v_img = self.v_img(clip_embed).view(b, -1, n, d)
            img_x = attention(q, k_img, v_img, attention_mode=self.attention_mode).flatten(2)
            x = x_text + img_x
        else:
            x = x_text

        # FantasyTalking audio attention
        if audio_proj is not None:
            if len(audio_proj.shape) == 4:
                audio_q = q.view(b * num_latent_frames, -1, n, d)
                ip_key = self.k_proj(audio_proj).view(b * num_latent_frames, -1, n, d)
                ip_value = self.v_proj(audio_proj).view(b * num_latent_frames, -1, n, d)

                audio_x = attention(audio_q, ip_key, ip_value, attention_mode=self.attention_mode)
                audio_x = audio_x.view(b, q.size(1), n, d).flatten(2)
            elif len(audio_proj.shape) == 3:
                ip_key = self.k_proj(audio_proj).view(b, -1, n, d)
                ip_value = self.v_proj(audio_proj).view(b, -1, n, d)
                audio_x = attention(q, ip_key, ip_value, attention_mode=self.attention_mode).flatten(2)
            x = x + audio_x * audio_scale

        # FantasyPortrait adapter attention
        if adapter_proj is not None:
            if len(adapter_proj.shape) == 4:
                adapter_q = q.view(b * num_latent_frames, -1, n, d)
                ip_key = self.ip_adapter_single_stream_k_proj(adapter_proj).view(b * num_latent_frames, -1, n, d)
                ip_value = self.ip_adapter_single_stream_v_proj(adapter_proj).view(b * num_latent_frames, -1, n, d)

                adapter_x = attention(adapter_q, ip_key, ip_value, attention_mode=self.attention_mode)
                adapter_x = adapter_x.view(b, q.size(1), n, d)
                adapter_x = adapter_x.flatten(2)
            elif len(adapter_proj.shape) == 3:
                ip_key = self.ip_adapter_single_stream_k_proj(adapter_proj).view(b, -1, n, d)
                ip_value = self.ip_adapter_single_stream_v_proj(adapter_proj).view(b, -1, n, d)
                adapter_x = attention(q, ip_key, ip_value, attention_mode=self.attention_mode)
                adapter_x = adapter_x.flatten(2)
            x = x + adapter_x * ip_scale

        return self.o(x)
    
class WanHuMoCrossAttention(WanSelfAttention):

    def __init__(self, in_features, out_features, num_heads, kv_dim=None, qk_norm=True, eps=1e-6, attention_mode='sdpa', rms_norm_function="default"):
        super().__init__(in_features, out_features, num_heads, qk_norm, eps, kv_dim=kv_dim, rms_norm_function=rms_norm_function)
        self.attention_mode = attention_mode

    def forward(self, x, context, grid_sizes, **kwargs):
    
        b, n, d = x.size(0), self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        # Handle video spatial structure
        hlen_wlen = grid_sizes[0][1] * grid_sizes[0][2]
        q = q.reshape(-1, hlen_wlen, n, d)
        
        # Handle audio temporal structure (16 tokens per frame)
        k = k.reshape(-1, 16, n, d)
        v = v.reshape(-1, 16, n, d)

        x_text = attention(q, k, v, attention_mode=self.attention_mode)
        x_text = x_text.view(b, -1, n, d).flatten(2)

        x = x_text

        return self.o(x)
    
class AudioCrossAttentionWrapper(nn.Module):
    def __init__(self, in_features, out_features, num_heads, qk_norm=True, eps=1e-6, kv_dim=None):
        super().__init__()

        self.audio_cross_attn = WanHuMoCrossAttention(in_features, out_features, num_heads, kv_dim=kv_dim)
        self.norm1_audio = WanLayerNorm(out_features, eps, elementwise_affine=True)

    def forward(self, x, audio, grid_sizes, humo_audio_scale=1.0):
        x = x + self.audio_cross_attn(self.norm1_audio(x), audio, grid_sizes) * humo_audio_scale
        return x

class MTVCrafterMotionAttention(WanSelfAttention):

    def forward(self, x, mo, pe, grid_sizes, freqs):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            mo: Motion tokens
            pe: 4D RoPE
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(mo)).view(b, n, -1, d)
        v = self.v(mo).view(b, -1, n, d)

        # compute attention
        x = attention(
            q=rope_apply(q, grid_sizes, freqs),
            k=apply_rotary_emb(k, pe).transpose(1, 2),
            v=v
        )

        return self.o(x.flatten(2))
    

WAN_CROSSATTENTION_CLASSES = {
    't2v_cross_attn': WanT2VCrossAttention,
    'i2v_cross_attn': WanI2VCrossAttention,
}


class WanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 in_features,
                 out_features,
                 ffn_dim,
                 ffn2_dim,
                 num_heads,
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 attention_mode="sdpa",
                 rope_func="comfy",
                 rms_norm_function="default",
                 use_motion_attn=False,
                 use_humo_audio_attn=False,
                 face_fuser_block=False
                 ):
        super().__init__()
        self.dim = out_features
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.attention_mode = attention_mode
        self.rope_func = rope_func
        #radial attn
        self.dense_timesteps = 10
        self.dense_block = False
        self.dense_attention_mode = "sageattn"

        self.kv_cache = None
        self.use_motion_attn = use_motion_attn
        self.has_face_fuser_block = face_fuser_block

        # layers
        self.norm1 = WanLayerNorm(out_features, eps)
        self.self_attn = WanSelfAttention(in_features, out_features, num_heads, qk_norm, eps, self.attention_mode, rms_norm_function=rms_norm_function)

        # MTV Crafter motion attn
        if self.use_motion_attn:
            self.norm4 = WanLayerNorm(out_features, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
            self.motion_attn = MTVCrafterMotionAttention(in_features, out_features, num_heads, qk_norm, eps, self.attention_mode)

        if cross_attn_type != "no_cross_attn":
            self.norm3 = WanLayerNorm(out_features, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
            self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](in_features, out_features, num_heads, qk_norm, eps, rms_norm_function=rms_norm_function)
        self.norm2 = WanLayerNorm(out_features, eps)
        self.ffn = nn.Sequential(
            nn.Linear(in_features, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn2_dim, out_features))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, out_features) / in_features**0.5)
        self.seg_idx = None

        # HuMo audio cross-attn
        if use_humo_audio_attn:
            self.audio_cross_attn_wrapper = AudioCrossAttentionWrapper(in_features, out_features, num_heads, qk_norm, eps, kv_dim=1536)

        if face_fuser_block:
            from .wananimate.face_blocks import FaceBlock
            self.fuser_block = FaceBlock(self.dim, num_heads)

    #@torch.compiler.disable()
    def get_mod(self, e):
        if e.dim() == 3:
            return (self.modulation  + e).chunk(6, dim=1) # 1, 6, dim
        elif e.dim() == 4:
            e_mod = self.modulation.unsqueeze(2) + e
            return [ei.squeeze(1) for ei in e_mod.unbind(dim=1)]

    def modulate(self, x, shift_msa, scale_msa, seg_idx=None):
        """
        Modulate x with shift and scale. If seg_idx is provided, apply segmented modulation.
        """
        norm_x = self.norm1(x)
        if seg_idx is not None:
            parts = []
            for i in range(2):
                part = torch.addcmul(
                    shift_msa[:, i:i + 1],
                    norm_x[:, seg_idx[i]:seg_idx[i + 1]],
                    1 + scale_msa[:, i:i + 1]
                )
                parts.append(part)
            norm_x = torch.cat(parts, dim=1)
            return norm_x
        else:
            return torch.addcmul(shift_msa, norm_x, 1 + scale_msa)
    
    def ffn_chunked(self, x, shift_mlp, scale_mlp, num_chunks=4):
        modulated_input = torch.addcmul(shift_mlp, self.norm2(x), 1 + scale_mlp)
        
        result = torch.empty_like(x)
        seq_len = modulated_input.shape[1]
        
        chunk_sizes = [seq_len // num_chunks + (1 if i < seq_len % num_chunks else 0) 
                    for i in range(num_chunks)]
        
        start_idx = 0
        for size in chunk_sizes:
            end_idx = start_idx + size
            chunk = modulated_input[:, start_idx:end_idx, :]
            result[:, start_idx:end_idx, :] = self.ffn(chunk)
            start_idx = end_idx
        
        return result

    #region attention forward
    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        current_step,
        last_step=False,
        video_attention_split_steps=[],
        clip_embed=None,
        camera_embed=None,
        audio_proj=None,
        audio_scale=1.0,
        num_latent_frames=21,
        original_seq_len=None,
        enhance_enabled=False,
        nag_params={},
        nag_context=None,
        is_uncond=False,
        multitalk_audio_embedding=None,
        ref_target_masks=None,
        human_num=0,
        inner_t=None, inner_c=None,
        cross_freqs=None,
        x_ip=None, e_ip=None,
        freqs_ip=None,
        adapter_proj=None,
        ip_scale=1.0,
        reverse_time=False,
        mtv_motion_tokens=None, mtv_motion_rotary_emb=None, mtv_strength=1.0, mtv_freqs=None,
        humo_audio_input=None, humo_audio_scale=1.0,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        self.original_seq_len = original_seq_len
        self.zero_timestep = len(e) == 2
        if self.zero_timestep: #s2v zero timestep
            self.seg_idx = e[1]
            self.seg_idx = min(max(0, self.seg_idx), x.size(1))
            self.seg_idx = [0, self.seg_idx, x.size(1)]
            e = e[0]

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.get_mod(e.to(x.device))
        del e
        input_x = self.modulate(x, shift_msa, scale_msa, seg_idx=self.seg_idx)
        del shift_msa, scale_msa

        if x_ip is not None:
            shift_msa_ip, scale_msa_ip, gate_msa_ip, shift_mlp_ip, scale_mlp_ip, gate_mlp_ip = self.get_mod(e_ip.to(x.device))
            input_x_ip = self.modulate(self.norm1(x_ip), shift_msa_ip, scale_msa_ip)
            self.cond_size = input_x_ip.shape[1]
            input_x = torch.concat([input_x, input_x_ip], dim=1)
            self.kv_cache = None

        if camera_embed is not None:
            # encode ReCamMaster camera
            camera_embed = self.cam_encoder(camera_embed.to(x))
            camera_embed = camera_embed.repeat(1, 2, 1)
            camera_embed = camera_embed.unsqueeze(2).unsqueeze(3).repeat(1, 1, grid_sizes[0][1], grid_sizes[0][2], 1)
            camera_embed = rearrange(camera_embed, 'b f h w d -> b (f h w) d')
            input_x += camera_embed

        # self-attention
        x_ref_attn_map = None

        # self-attention variables
        q_ip = k_ip = v_ip = None

        #RoPE and QKV computation
        if inner_t is not None:
            #query, key, value
            q, k, v = self.self_attn.qkv_fn(input_x)
            q=rope_apply_echoshot(q, grid_sizes, freqs, inner_t).to(q)
            k=rope_apply_echoshot(k, grid_sizes, freqs, inner_t).to(k)
        elif x_ip is not None and self.kv_cache is None:
            # First pass - separate main and IP components
            x_main, x_ip_input = input_x[:, : -self.cond_size], input_x[:, -self.cond_size :]
            # Compute QKV for main content
            q, k, v = self.self_attn.qkv_fn(x_main)
            if self.rope_func == "comfy":
                q, k = apply_rope_comfy(q, k, freqs)
            elif self.rope_func == "comfy_chunked":
                q, k = apply_rope_comfy_chunked(q, k, freqs)
            # Compute QKV for IP content
            q_ip, k_ip, v_ip = self.self_attn.qkv_fn_ip(x_ip_input)
            if self.rope_func == "comfy":
                q_ip, k_ip = apply_rope_comfy(q_ip, k_ip, freqs_ip)
            elif self.rope_func == "comfy_chunked":
                q_ip, k_ip = apply_rope_comfy_chunked(q_ip, k_ip, freqs_ip)
        else:
            q, k, v = self.self_attn.qkv_fn(input_x)
            if self.rope_func == "comfy":
                q, k = apply_rope_comfy(q, k, freqs)
            elif self.rope_func == "comfy_chunked":
                q, k = apply_rope_comfy_chunked(q, k, freqs)
            else:
                q=rope_apply(q, grid_sizes, freqs, reverse_time=reverse_time)
                k=rope_apply(k, grid_sizes, freqs, reverse_time=reverse_time)

        # FETA
        if enhance_enabled:
            feta_scores = get_feta_scores(q, k)
        
        #self-attention
        split_attn = (context is not None 
                      and (context.shape[0] > 1 or (clip_embed is not None and clip_embed.shape[0] > 1)) 
                      and x.shape[0] == 1
                      and inner_t is None
                      and x_ip is None  # Don't split when using IP-Adapter
                      )
        if split_attn:
            y = self.self_attn.forward_split(
            q, k, v, 
            seq_lens, grid_sizes, freqs, 
            seq_chunks=max(context.shape[0], clip_embed.shape[0] if clip_embed is not None else 0),
            current_step=current_step,
            video_attention_split_steps=video_attention_split_steps
            )
        elif ref_target_masks is not None:
            y, x_ref_attn_map = self.self_attn.forward_multitalk(q, k, v, seq_lens, grid_sizes, ref_target_masks)
        elif self.attention_mode == "radial_sage_attention":
            if self.dense_block or self.dense_timesteps is not None and current_step < self.dense_timesteps:
                if self.dense_attention_mode == "sparse_sage_attn":
                    y = self.self_attn.forward_radial(q, k, v, dense_step=True)
                else:
                    y = self.self_attn.forward(q, k, v, seq_lens)
            else:
                y = self.self_attn.forward_radial(q, k, v, dense_step=False)
        elif self.attention_mode == "sageattn_3":
            if current_step != 0 and not last_step:
                y = self.self_attn.forward(q, k, v, seq_lens, attention_mode_override="sageattn_3")
            else:
                y = self.self_attn.forward(q, k, v, seq_lens, attention_mode_override="sageattn")
        elif x_ip is not None and self.kv_cache is None:
            # First pass: cache IP keys/values and compute attention
            self.kv_cache = {"k_ip": k_ip.detach(), "v_ip": v_ip.detach()}
            y = self.self_attn.forward_ip(q, k, v, q_ip, k_ip, v_ip, seq_lens)
        elif self.kv_cache is not None:
            # Subsequent passes: use cached IP keys/values
            k_ip = self.kv_cache["k_ip"]
            v_ip = self.kv_cache["v_ip"]
            full_k = torch.cat([k, k_ip], dim=1)
            full_v = torch.cat([v, v_ip], dim=1)
            y = self.self_attn.forward(q, full_k, full_v, seq_lens)
        else:
            y = self.self_attn.forward(q, k, v, seq_lens)

        # FETA
        if enhance_enabled:
            y.mul_(feta_scores)

        #ReCamMaster
        if camera_embed is not None:
            y = self.projector(y)        

        if x_ip is not None:
            y, y_ip = (
                y[:, : -self.cond_size],
                y[:, -self.cond_size :],
            )

        if self.zero_timestep:
            z = []
            for i in range(2):
                z.append(y[:, self.seg_idx[i]:self.seg_idx[i + 1]] * gate_msa[:, i:i + 1])
            y = torch.cat(z, dim=1)
            x = x.add(y)
        else:
            x = x.addcmul(y, gate_msa)
        del y, gate_msa

        # cross-attention & ffn function
        if context is not None:
            if split_attn:
                if nag_context is not None:
                    raise NotImplementedError("nag_context is not supported in split_cross_attn_ffn")
                x = self.split_cross_attn_ffn(x, context, shift_mlp, scale_mlp, gate_mlp, clip_embed, grid_sizes)
            else:
                x = self.cross_attn_ffn(x, context, grid_sizes, shift_mlp, scale_mlp, gate_mlp, clip_embed, 
                                        audio_proj, audio_scale, num_latent_frames, nag_params, nag_context, is_uncond, 
                                        multitalk_audio_embedding, x_ref_attn_map, human_num, inner_t, inner_c, cross_freqs,
                                        adapter_proj=adapter_proj, ip_scale=ip_scale, 
                                        mtv_freqs=mtv_freqs, mtv_motion_tokens=mtv_motion_tokens, mtv_motion_rotary_emb=mtv_motion_rotary_emb, mtv_strength=mtv_strength,
                                        humo_audio_input=humo_audio_input, humo_audio_scale=humo_audio_scale
                                        )
        else:
            if self.rope_func == "comfy_chunked":
                y = self.ffn_chunked(x, shift_mlp, scale_mlp)
            else:
                y = self.ffn(torch.addcmul(shift_mlp, self.norm2(x), 1 + scale_mlp))
            x = x.addcmul(y, gate_mlp)
        del gate_mlp

        if x_ip is not None:
            x_ip = x_ip.addcmul(y_ip, gate_msa_ip)
            y_ip = self.ffn(torch.addcmul(shift_mlp_ip, self.norm2(x_ip), 1 + scale_mlp_ip))
            x_ip = x_ip.addcmul(y_ip, gate_mlp_ip)

        return x, x_ip

    
    def cross_attn_ffn(self, x, context, grid_sizes, shift_mlp, scale_mlp, gate_mlp, clip_embed, 
                       audio_proj, audio_scale, num_latent_frames, nag_params, 
                       nag_context, is_uncond, multitalk_audio_embedding, x_ref_attn_map, human_num, 
                       inner_t, inner_c, cross_freqs, adapter_proj, ip_scale, mtv_freqs, mtv_motion_tokens, mtv_motion_rotary_emb, mtv_strength,
                       humo_audio_input, humo_audio_scale):

            x = x + self.cross_attn(self.norm3(x), context, grid_sizes, clip_embed=clip_embed,
                                    audio_proj=audio_proj, audio_scale=audio_scale,
                                    num_latent_frames=num_latent_frames, nag_params=nag_params, nag_context=nag_context, is_uncond=is_uncond,
                                    rope_func=self.rope_func, inner_t=inner_t, inner_c=inner_c, cross_freqs=cross_freqs,
                                    adapter_proj=adapter_proj, ip_scale=ip_scale, orig_seq_len=self.original_seq_len)
            # MultiTalk
            if multitalk_audio_embedding is not None and not isinstance(self, VaceWanAttentionBlock):
                x_audio = self.audio_cross_attn(self.norm_x(x), encoder_hidden_states=multitalk_audio_embedding,
                                            shape=grid_sizes[0], x_ref_attn_map=x_ref_attn_map, human_num=human_num)
                x = x + x_audio * audio_scale

            # MTV-Crafter Motion Attention
            if self.use_motion_attn and mtv_motion_tokens is not None and mtv_motion_rotary_emb is not None:                
                x_motion = self.motion_attn(self.norm4(x), mtv_motion_tokens, mtv_motion_rotary_emb, grid_sizes, mtv_freqs)
                x = x + x_motion * mtv_strength

            # HuMo Audio Cross-Attention
            if humo_audio_input is not None:
                x = self.audio_cross_attn_wrapper(x, humo_audio_input, grid_sizes, humo_audio_scale)

            if self.rope_func == "comfy_chunked" and not self.zero_timestep:
                y = self.ffn_chunked(x, shift_mlp, scale_mlp)
            else:
                norm2_x = self.norm2(x)
                if self.zero_timestep:
                    parts = []
                    for i in range(2):
                        parts.append(norm2_x[:, self.seg_idx[i]:self.seg_idx[i + 1]] *
                                    (1 + scale_mlp[:, i:i + 1]) + shift_mlp[:, i:i + 1])
                    norm2_x = torch.cat(parts, dim=1)
                    y = self.ffn(norm2_x)
                else:
                    input_x = torch.addcmul(shift_mlp, norm2_x, 1 + scale_mlp)
                    del shift_mlp, scale_mlp, norm2_x
                    y = self.ffn(input_x)
            if self.zero_timestep:
                z = []
                for i in range(2):
                    z.append(y[:, self.seg_idx[i]:self.seg_idx[i + 1]] * gate_mlp[:, i:i + 1])
                y = torch.cat(z, dim=1)
                x = x.add(y)
            else:
                x = x.addcmul(y, gate_mlp)
            return x
    
    @torch.compiler.disable()
    def split_cross_attn_ffn(self, x, context, shift_mlp, scale_mlp, gate_mlp, clip_embed=None, grid_sizes=None):
        # Get number of prompts
        num_prompts = context.shape[0]
        num_clip_embeds = 0 if clip_embed is None else clip_embed.shape[0]
        num_segments = max(num_prompts, num_clip_embeds)
        
        # Extract spatial dimensions
        frames, height, width = grid_sizes[0]  # Assuming batch size 1
        tokens_per_frame = height * width
        
        # Distribute frames across prompts
        frames_per_segment = max(1, frames // num_segments)
        
        # Process each prompt segment
        x_combined = torch.zeros_like(x)
        
        for i in range(num_segments):
            # Calculate frame boundaries for this segment
            start_frame = i * frames_per_segment
            end_frame = min((i+1) * frames_per_segment, frames) if i < num_segments-1 else frames
            
            # Convert frame indices to token indices
            start_idx = start_frame * tokens_per_frame
            end_idx = end_frame * tokens_per_frame
            segment_indices = torch.arange(start_idx, end_idx, device=x.device, dtype=torch.long)
            
            # Get prompt segment (cycle through available prompts if needed)
            prompt_idx = i % num_prompts
            segment_context = context[prompt_idx:prompt_idx+1]
            
            # Handle clip_embed for this segment (cycle through available embeddings)
            segment_clip_embed = None
            if clip_embed is not None:
                clip_idx = i % num_clip_embeds
                segment_clip_embed = clip_embed[clip_idx:clip_idx+1]
            
            # Get tensor segment
            x_segment = x[:, segment_indices, :]
            
            # Process segment with its prompt and clip embedding
            processed_segment = self.cross_attn(self.norm3(x_segment), segment_context, clip_embed=segment_clip_embed)
            processed_segment = processed_segment.to(x.dtype)
            
            # Add to combined result
            x_combined[:, segment_indices, :] = processed_segment
        
        # Continue with FFN
        x = x + x_combined
        y = self.ffn_chunked(x, shift_mlp, scale_mlp)
        x = x.addcmul(y, gate_mlp)
        return x

class VaceWanAttentionBlock(WanAttentionBlock):
    def __init__(
            self,
            cross_attn_type,
            in_features,
            out_features,
            ffn_dim,
            ffn2_dim,
            num_heads,
            qk_norm=True,
            cross_attn_norm=False,
            eps=1e-6,
            block_id=0,
            attention_mode='sdpa',
            rope_func="comfy",
            rms_norm_function="default"
    ):
        super().__init__(cross_attn_type, in_features, out_features, ffn_dim, ffn2_dim, num_heads, qk_norm, cross_attn_norm, eps, attention_mode, rope_func, rms_norm_function=rms_norm_function)
        self.block_id = block_id
        if block_id == 0:
            self.before_proj = nn.Linear(in_features, out_features)
        self.after_proj = nn.Linear(in_features, out_features)

    def forward(self, c, **kwargs):
        return super().forward(c, **kwargs)

class BaseWanAttentionBlock(WanAttentionBlock):
    def __init__(
        self,
        cross_attn_type,
        in_features,
        out_features,
        ffn_dim,
        ffn2_dim,
        num_heads,
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        block_id=None,
        attention_mode='sdpa',
        rope_func="comfy",
        rms_norm_function="default"
    ):
        super().__init__(cross_attn_type, in_features, out_features, ffn_dim, ffn2_dim, num_heads, qk_norm, cross_attn_norm, eps, attention_mode, rope_func, rms_norm_function=rms_norm_function)
        self.block_id = block_id

    def forward(self, x, vace_hints=None, vace_context_scale=[1.0], **kwargs):
        x, x_ip = super().forward(x, **kwargs)
        if vace_hints is None:
            return x, x_ip
        
        if self.block_id is not None:
            for i in range(len(vace_hints)):
                x.add_(vace_hints[i][self.block_id].to(x.device), alpha=vace_context_scale[i])
        return x, x_ip

class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def get_mod(self, e):
        if e.dim() == 2:
            return (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
        elif e.dim() == 3:
            e = (self.modulation.unsqueeze(2) + e.unsqueeze(1)).chunk(2, dim=1)
            return [ei.squeeze(1) for ei in e]

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """

        e = self.get_mod(e.to(x.device))
        x = self.head(self.norm(x).mul_(1 + e[1]).add_(e[0]))
        return x


class MLPProj(torch.nn.Module):

    def __init__(self, in_dim, out_dim, fl_pos_emb=False):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim), torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(), torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim))
        if fl_pos_emb:  # NOTE: we only use this for `fl2v`
            self.emb_pos = nn.Parameter(torch.zeros(1, 257 * 2, 1280))

    def forward(self, image_embeds):
        if hasattr(self, 'emb_pos'):
            image_embeds = image_embeds + self.emb_pos.to(image_embeds.device)
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens

from .s2v.auxi_blocks import MotionEncoder_tc


class CausalAudioEncoder(nn.Module):

    def __init__(self,
                 dim=5120,
                 num_layers=25,
                 out_dim=2048,
                 video_rate=8,
                 num_token=4,
                 need_global=False):
        super().__init__()
        self.encoder = MotionEncoder_tc(
            in_dim=dim,
            hidden_dim=out_dim,
            num_heads=num_token,
            need_global=need_global)
        weight = torch.ones((1, num_layers, 1, 1)) * 0.01

        self.weights = torch.nn.Parameter(weight)
        self.act = torch.nn.SiLU()

    def forward(self, features):
        # features B * num_layers * dim * video_length
        weights = self.act(self.weights)
        weights_sum = weights.sum(dim=1, keepdims=True)
        weighted_feat = ((features * weights) / weights_sum).sum(
            dim=1)  # b dim f
        weighted_feat = weighted_feat.permute(0, 2, 1)  # b f dim
        res = self.encoder(weighted_feat)  # b f n dim

        return res  # b f n dim


class AudioCrossAttention(WanT2VCrossAttention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class AudioInjector_WAN(nn.Module):

    def __init__(self,
                 all_modules,
                 all_modules_names,
                 dim=2048,
                 num_heads=32,
                 inject_layer=[0, 27],
                 root_net=None,
                 enable_adain=False,
                 adain_dim=2048,
                 need_adain_ont=False,
                 attention_mode='sdpa'):
        super().__init__()
        self.injected_block_id = {}
        audio_injector_id = 0
        for mod_name, mod in zip(all_modules_names, all_modules):
            if isinstance(mod, WanAttentionBlock):
                for inject_id in inject_layer:
                    if f'transformer_blocks.{inject_id}' in mod_name:
                        self.injected_block_id[inject_id] = audio_injector_id
                        audio_injector_id += 1

        self.injector = nn.ModuleList([
            AudioCrossAttention(
                in_features=dim,
                out_features=dim,
                num_heads=num_heads,
                qk_norm=True,
                attention_mode=attention_mode
            ) for _ in range(audio_injector_id)
        ])
        self.injector_pre_norm_feat = nn.ModuleList([
            nn.LayerNorm(
                dim,
                elementwise_affine=False,
                eps=1e-6,
            ) for _ in range(audio_injector_id)
        ])
        self.injector_pre_norm_vec = nn.ModuleList([
            nn.LayerNorm(
                dim,
                elementwise_affine=False,
                eps=1e-6,
            ) for _ in range(audio_injector_id)
        ])
        if enable_adain:
            self.injector_adain_layers = nn.ModuleList([
                AdaLayerNorm(
                    output_dim=dim * 2, embedding_dim=adain_dim, chunk_dim=1)
                for _ in range(audio_injector_id)
            ])
            if need_adain_ont:
                self.injector_adain_output_layers = nn.ModuleList(
                    [nn.Linear(dim, dim) for _ in range(audio_injector_id)])

class WanModel(torch.nn.Module):
    def __init__(self,
                model_type='t2v',
                patch_size=(1, 2, 2),
                text_len=512,
                in_dim=16,
                dim=2048,
                in_features=5120,
                out_features=5120,
                ffn_dim=8192,
                ffn2_dim=8192,
                freq_dim=256,
                text_dim=4096,
                out_dim=16,
                num_heads=16,
                num_layers=32,
                qk_norm=True,
                cross_attn_norm=True,
                eps=1e-6,
                attention_mode='sdpa',
                rope_func='comfy',
                rms_norm_function='default',
                main_device=torch.device('cuda'),
                offload_device=torch.device('cpu'),
                dtype=torch.float16,
                teacache_coefficients=[],
                magcache_ratios=[],
                vace_layers=None,
                vace_in_dim=None,
                inject_sample_info=False,
                add_ref_conv=False,
                in_dim_ref_conv=16,
                add_control_adapter=False,
                in_dim_control_adapter=24,
                use_motion_attn=False,
                #s2v
                cond_dim=0,
                audio_dim=1024,
                num_audio_token=4,
                enable_adain=False,
                adain_mode="attn_norm",
                audio_inject_layers=[0, 4, 8, 12, 16, 20, 24, 27, 30, 33, 36, 39],
                zero_timestep=False,
                humo_audio=False,
                # WanAnimate
                is_wananimate=False,
                motion_encoder_dim=512,
                ):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.in_features = in_features
        self.out_features = out_features
        self.ffn_dim = ffn_dim
        self.ffn2_dim = ffn2_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.attention_mode = attention_mode
        self.rope_func = rope_func
        self.main_device = main_device
        self.offload_device = offload_device
        self.vace_layers = vace_layers
        self.device = main_device
        self.patched_linear = False

        self.blocks_to_swap = -1
        self.offload_txt_emb = False
        self.offload_img_emb = False
        self.vace_blocks_to_swap = -1

        self.cache_device = offload_device

        #init TeaCache variables
        self.enable_teacache = False
        self.rel_l1_thresh = 0.15
        self.teacache_start_step= 0
        self.teacache_end_step = -1
        self.teacache_state = TeaCacheState(cache_device=self.cache_device)
        self.teacache_coefficients = teacache_coefficients
        self.teacache_use_coefficients = False
        self.teacache_mode = 'e'

        #init MagCache variables
        self.enable_magcache = False
        self.magcache_state = MagCacheState(cache_device=self.cache_device)
        self.magcache_thresh = 0.24
        self.magcache_K = 4
        self.magcache_start_step = 0
        self.magcache_end_step = -1
        self.magcache_ratios = magcache_ratios

        #init EasyCache variables
        self.enable_easycache = False
        self.easycache_thresh = 0.1
        self.easycache_start_step = 0
        self.easycache_end_step = -1
        self.easycache_state = EasyCacheState(cache_device=self.cache_device)

        self.slg_blocks = None
        self.slg_start_percent = 0.0
        self.slg_end_percent = 1.0

        self.use_non_blocking = False
        self.prefetch_blocks = 0
        self.block_swap_debug = False

        self.video_attention_split_steps = []
        self.lora_scheduling_enabled = False

        self.multitalk_model_type = "none"

        self.humo_audio = humo_audio

        self.motion_encoder_dim = motion_encoder_dim

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        
        self.original_patch_embedding = self.patch_embedding
        self.expanded_patch_embedding = self.patch_embedding

        if model_type != 'no_cross_attn':
            self.text_embedding = nn.Sequential(
                nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
                nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        if vace_layers is not None:
            self.vace_layers = [i for i in range(0, self.num_layers, 2)] if vace_layers is None else vace_layers
            self.vace_in_dim = self.in_dim if vace_in_dim is None else vace_in_dim

            self.vace_layers_mapping = {i: n for n, i in enumerate(self.vace_layers)}

            # vace blocks
            self.vace_blocks = nn.ModuleList([
                VaceWanAttentionBlock('t2v_cross_attn', self.in_features, self.out_features, self.ffn_dim, self.ffn2_dim,self.num_heads, self.qk_norm,
                                        self.cross_attn_norm, self.eps, block_id=i, attention_mode=self.attention_mode, rope_func=self.rope_func, rms_norm_function=rms_norm_function)
                for i in self.vace_layers
            ])

            # vace patch embeddings
            self.vace_patch_embedding = nn.Conv3d(
                self.vace_in_dim, self.dim, kernel_size=self.patch_size, stride=self.patch_size
            )
            self.blocks = nn.ModuleList([
            BaseWanAttentionBlock('t2v_cross_attn', self.in_features, self.out_features, ffn_dim, self.ffn2_dim, num_heads,
                              qk_norm, cross_attn_norm, eps,
                              attention_mode=self.attention_mode, rope_func=self.rope_func, rms_norm_function=rms_norm_function,
                              block_id=self.vace_layers_mapping[i] if i in self.vace_layers else None)
            for i in range(num_layers)
            ])
        else:
            # blocks
            if model_type == 't2v' or model_type == 's2v':
                cross_attn_type = 't2v_cross_attn'
            elif model_type == 'i2v' or model_type == 'fl2v':
                cross_attn_type = 'i2v_cross_attn'
            else:
                cross_attn_type = 'no_cross_attn'

            self.blocks = nn.ModuleList([
                WanAttentionBlock(cross_attn_type, self.in_features, self.out_features, ffn_dim, ffn2_dim, num_heads,
                                qk_norm, cross_attn_norm, eps,
                                attention_mode=self.attention_mode, rope_func=self.rope_func, rms_norm_function=rms_norm_function, 
                                use_motion_attn=(i % 4 == 0 and use_motion_attn), use_humo_audio_attn=self.humo_audio,
                                face_fuser_block = (i % 5 == 0 and is_wananimate))
                for i in range(num_layers)
            ])
        #MTV Crafter
        if use_motion_attn:
            self.pad_motion_tokens = torch.zeros(1, 1, 2048)

        # head
        self.head = Head(dim, out_dim, patch_size, eps)
        

        d = self.dim // self.num_heads
        self.rope_embedder = EmbedND_RifleX(
            d, 
            10000.0, 
            [d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)],
            num_frames=None,
            k=None,
            )
        self.cached_freqs = self.cached_shape = self.cached_cond = None

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        
        if model_type == 'i2v' or model_type == 'fl2v':
            self.img_emb = MLPProj(1280, dim, fl_pos_emb=model_type == 'fl2v')

        #skyreels v2
        if inject_sample_info:
            self.fps_embedding = nn.Embedding(2, dim)
            self.fps_projection = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim * 6))
        #fun 1.1
        if add_ref_conv:
            self.ref_conv = nn.Conv2d(in_dim_ref_conv, dim, kernel_size=patch_size[1:], stride=patch_size[1:])
        else:
            self.ref_conv = None

        if add_control_adapter:
            from .wan_camera_adapter import SimpleAdapter
            self.control_adapter = SimpleAdapter(in_dim_control_adapter, dim, kernel_size=patch_size[1:], stride=patch_size[1:])
        else:
            self.control_adapter = None

        #S2V
        self.zero_timestep = self.audio_injector = self.trainable_cond_mask =None
        if cond_dim > 0:
            self.cond_encoder = nn.Conv3d(
                cond_dim,
                self.dim,
                kernel_size=self.patch_size,
                stride=self.patch_size)
        if self.model_type == 's2v':
            self.enable_adain = enable_adain
            self.casual_audio_encoder = CausalAudioEncoder(
                dim=audio_dim,
                out_dim=self.dim,
                num_token=num_audio_token,
                need_global=enable_adain)
            all_modules, all_modules_names = torch_dfs(
                self.blocks, parent_name="root.transformer_blocks")
            self.audio_injector = AudioInjector_WAN(
                all_modules,
                all_modules_names,
                dim=self.dim,
                num_heads=self.num_heads,
                inject_layer=audio_inject_layers,
                root_net=self,
                enable_adain=enable_adain,
                adain_dim=self.dim,
                need_adain_ont=adain_mode != "attn_norm",
                attention_mode=attention_mode
            )
            self.trainable_cond_mask = nn.Embedding(3, self.dim)

            self.frame_packer = FramePackMotioner(
                inner_dim=self.dim,
                num_heads=self.num_heads,
                zip_frame_buckets=[1, 2, 16],
                drop_mode='padd')
        self.adain_mode = adain_mode
        self.zero_timestep = zero_timestep

        # HuMo Audio
        if self.humo_audio:
            from ...HuMo.audio_proj import AudioProjModel
            self.audio_proj = AudioProjModel(seq_len=8, blocks=5, channels=1280, 
                intermediate_dim=512, output_dim=1536, context_tokens=16)
        # WanAnimate
        self.motion_encoder = self.pose_patch_embedding = self.face_encoder = self.face_adapter = None
        if is_wananimate:
            from .wananimate.motion_encoder import MotionExtractor
            from .wananimate.face_blocks import FaceEncoder
            self.pose_patch_embedding = nn.Conv3d(16, dim, kernel_size=patch_size, stride=patch_size)
            self.motion_encoder = MotionExtractor()

            self.face_encoder = FaceEncoder(
                in_dim=motion_encoder_dim,
                out_dim=self.dim,
                num_heads=4,
                dtype=dtype
            )

    def block_swap(self, blocks_to_swap, offload_txt_emb=False, offload_img_emb=False, vace_blocks_to_swap=None, prefetch_blocks=0, block_swap_debug=False):
        # Clamp blocks_to_swap to valid range
        blocks_to_swap = max(0, min(blocks_to_swap, len(self.blocks)))
        
        log.info(f"Swapping {blocks_to_swap} transformer blocks")
        self.blocks_to_swap = blocks_to_swap
        self.prefetch_blocks = prefetch_blocks
        self.block_swap_debug = block_swap_debug
        
        self.offload_img_emb = offload_img_emb
        self.offload_txt_emb = offload_txt_emb

        total_offload_memory = 0
        total_main_memory = 0
        
        # Calculate the index where swapping starts
        swap_start_idx = len(self.blocks) - blocks_to_swap
       
        for b, block in tqdm(enumerate(self.blocks), total=len(self.blocks), desc="Initializing block swap"):
            block_memory = get_module_memory_mb(block)
            
            if b < swap_start_idx:
                block.to(self.main_device)
                total_main_memory += block_memory
            else:
                block.to(self.offload_device, non_blocking=self.use_non_blocking)
                total_offload_memory += block_memory

        if blocks_to_swap != -1 and vace_blocks_to_swap == 0:
            vace_blocks_to_swap = 1

        if vace_blocks_to_swap > 0 and self.vace_layers is not None:
            # Clamp vace_blocks_to_swap to valid range
            vace_blocks_to_swap = max(0, min(vace_blocks_to_swap, len(self.vace_blocks)))
            self.vace_blocks_to_swap = vace_blocks_to_swap
            
            # Calculate the index where VACE swapping starts
            vace_swap_start_idx = len(self.vace_blocks) - vace_blocks_to_swap

            for b, block in tqdm(enumerate(self.vace_blocks), total=len(self.vace_blocks), desc="Initializing vace block swap"):
                block_memory = get_module_memory_mb(block)
                
                if b < vace_swap_start_idx:
                    block.to(self.main_device)
                    total_main_memory += block_memory
                else:
                    block.to(self.offload_device, non_blocking=self.use_non_blocking)
                    total_offload_memory += block_memory

        mm.soft_empty_cache()
        gc.collect()

        log.info("----------------------")
        log.info(f"Block swap memory summary:")
        log.info(f"Transformer blocks on {self.offload_device}: {total_offload_memory:.2f}MB")
        log.info(f"Transformer blocks on {self.main_device}: {total_main_memory:.2f}MB")
        log.info(f"Total memory used by transformer blocks: {(total_offload_memory + total_main_memory):.2f}MB")
        log.info(f"Non-blocking memory transfer: {self.use_non_blocking}")
        log.info("----------------------")

    def forward_vace(
        self,
        x,
        vace_context,
        seq_len,
        kwargs
    ):
        # embeddings
        c = [self.vace_patch_embedding(u.unsqueeze(0).float()).to(x.dtype) for u in vace_context]
        c = [u.flatten(2).transpose(1, 2) for u in c]
        c = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in c
        ])

        if x.shape[1] > c.shape[1]:
            c = torch.cat([c.new_zeros(x.shape[0], x.shape[1] - c.shape[1], c.shape[2]), c], dim=1)
        if c.shape[1] > x.shape[1]:
            c = c[:, :x.shape[1]]
        
        hints = []
        current_c = c
        vace_swap_start_idx = len(self.vace_blocks) - self.vace_blocks_to_swap if self.vace_blocks_to_swap > 0 else len(self.vace_blocks)
        
        for b, block in enumerate(self.vace_blocks):
            if b >= vace_swap_start_idx and self.vace_blocks_to_swap > 0:
                block.to(self.main_device)
                
            if b == 0:
                c_processed = block.before_proj(current_c) + x
            else:
                c_processed = current_c
                
            c_processed, _ = block.forward(c_processed, **kwargs)
            
            # Store skip connection
            c_skip = block.after_proj(c_processed)
            hints.append(c_skip.to(
                self.offload_device if self.vace_blocks_to_swap > 0 else self.main_device, 
                non_blocking=self.use_non_blocking
            ))
            
            current_c = c_processed
            
            if b >= vace_swap_start_idx and self.vace_blocks_to_swap > 0:
                block.to(self.offload_device, non_blocking=self.use_non_blocking)

        return hints
    
    def audio_injector_forward(self, block_idx, x, audio_emb, scale=1.0):
        if block_idx in self.audio_injector.injected_block_id.keys():
            audio_attn_id = self.audio_injector.injected_block_id[block_idx]
            num_frames = audio_emb.shape[1]# b f n c

            input_x = x[:, :self.original_seq_len].clone()  # b (f h w) c
            input_x = rearrange(input_x, "b (t n) c -> (b t) n c", t=num_frames)

            if self.enable_adain and self.adain_mode == "attn_norm":
                audio_emb_global = self.audio_emb_global
                audio_emb_global = rearrange(audio_emb_global,"b t n c -> (b t) n c")
                attn_x = self.audio_injector.injector_adain_layers[audio_attn_id](input_x, temb=audio_emb_global[:, 0])
            else:
                attn_x = self.audio_injector.injector_pre_norm_feat[audio_attn_id](input_x)

            attn_audio_emb = rearrange(audio_emb, "b t n c -> (b t) n c", t=num_frames)
            residual_out = self.audio_injector.injector[audio_attn_id](
                x=attn_x ,
                context=attn_audio_emb * scale,
            )
            residual_out = rearrange(residual_out, "(b t) n c -> b (t n) c", t=num_frames)
            x[:, :self.original_seq_len].add_(residual_out)

        return x

    def wananimate_pose_embedding(self, x, pose_latents, strength=1.0):
        pose_latents = [self.pose_patch_embedding(u.unsqueeze(0).to(torch.float32)).to(x[0].dtype) for u in pose_latents]
        for x_, pose_latents_ in zip(x, pose_latents):
            x_[:, :, 1:].add_(pose_latents_, alpha=strength)
        return x


    def wananimate_face_embedding(self, face_pixel_values):
        b,c,T,h,w = face_pixel_values.shape
        face_pixel_values = rearrange(face_pixel_values, "b c t h w -> (b t) c h w")

        encode_bs = 8
        face_pixel_values_tmp = []
        self.motion_encoder.to(self.main_device)
        for i in range(math.ceil(face_pixel_values.shape[0]/encode_bs)):
            face_pixel_values_tmp.append(self.motion_encoder(face_pixel_values[i*encode_bs:(i+1)*encode_bs]))
        del face_pixel_values
        self.motion_encoder.to(self.offload_device)

        motion_vec = rearrange(torch.cat(face_pixel_values_tmp), "(b t) c -> b t c", t=T)
        del face_pixel_values_tmp
        self.face_encoder.to(self.main_device)
        motion_vec = self.face_encoder(motion_vec.to(self.face_encoder.dtype))
        self.face_encoder.to(self.offload_device)

        B, L, H, C = motion_vec.shape
        pad_face = torch.zeros(B, 1, H, C, device=motion_vec.device, dtype=motion_vec.dtype)
        return torch.cat([pad_face, motion_vec], dim=1)


    def wananimate_forward(self, block, x, motion_vec, strength=1.0, motion_masks=None):
            adapter_args = [x, motion_vec, motion_masks]
            residual_out = block.fuser_block(*adapter_args)
            return x.add(residual_out, alpha=strength)


    def rope_encode_comfy(self, t, h, w, freq_offset=0, t_start=0, attn_cond_shape=None, steps_t=None, steps_h=None, steps_w=None, ntk_alphas=[1,1,1], device=None, dtype=None):
        patch_size = self.patch_size
        t_len = ((t + (patch_size[0] // 2)) // patch_size[0])
        h_len = ((h + (patch_size[1] // 2)) // patch_size[1])
        w_len = ((w + (patch_size[2] // 2)) // patch_size[2])

        if steps_t is None:
            steps_t = t_len
        if steps_h is None:
            steps_h = h_len
        if steps_w is None:
            steps_w = w_len

        img_ids = torch.zeros((steps_t, steps_h, steps_w, 3), device=device, dtype=dtype)
        img_ids[:, :, :, 0] = img_ids[:, :, :, 0] + torch.linspace(t_start+freq_offset, t_start + (t_len - 1), steps=steps_t, device=device, dtype=dtype).reshape(-1, 1, 1)
        img_ids[:, :, :, 1] = img_ids[:, :, :, 1] + torch.linspace(freq_offset, h_len - 1, steps=steps_h, device=device, dtype=dtype).reshape(1, -1, 1)
        img_ids[:, :, :, 2] = img_ids[:, :, :, 2] + torch.linspace(freq_offset, w_len - 1, steps=steps_w, device=device, dtype=dtype).reshape(1, 1, -1)
        img_ids = img_ids.reshape(1, -1, img_ids.shape[-1])
        if attn_cond_shape is not None:
            F_cond, H_cond, W_cond = attn_cond_shape[2], attn_cond_shape[3], attn_cond_shape[4]
            cond_f_len = ((F_cond + (self.patch_size[0] // 2)) // self.patch_size[0])
            cond_h_len = ((H_cond + (self.patch_size[1] // 2)) // self.patch_size[1])
            cond_w_len = ((W_cond + (self.patch_size[2] // 2)) // self.patch_size[2])
            cond_img_ids = torch.zeros((cond_f_len, cond_h_len, cond_w_len, 3), device=device, dtype=dtype)
            
            #shift
            shift_f_size = 81 # Default value
            shift_f = False
            if shift_f:
                cond_img_ids[:, :, :, 0] = cond_img_ids[:, :, :, 0] + torch.linspace(shift_f_size, shift_f_size + cond_f_len - 1,steps=cond_f_len, device=device, dtype=dtype).reshape(-1, 1, 1)
            else:
                cond_img_ids[:, :, :, 0] = cond_img_ids[:, :, :, 0] + torch.linspace(0, cond_f_len - 1, steps=cond_f_len, device=device, dtype=dtype).reshape(-1, 1, 1)
            cond_img_ids[:, :, :, 1] = cond_img_ids[:, :, :, 1] + torch.linspace(h_len, h_len + cond_h_len - 1, steps=cond_h_len, device=device, dtype=dtype).reshape(1, -1, 1)
            cond_img_ids[:, :, :, 2] = cond_img_ids[:, :, :, 2] + torch.linspace(w_len, w_len + cond_w_len - 1, steps=cond_w_len, device=device, dtype=dtype).reshape(1, 1, -1)

            # Combine original and conditional position ids
            #img_ids = repeat(img_ids, "t h w c -> b (t h w) c", b=1)
            #cond_img_ids = repeat(cond_img_ids, "t h w c -> b (t h w) c", b=1)
            cond_img_ids = cond_img_ids.reshape(1, -1, cond_img_ids.shape[-1])
            combined_img_ids = torch.cat([img_ids, cond_img_ids], dim=1)
            
            # Generate RoPE frequencies for the combined positions
            freqs = self.rope_embedder(combined_img_ids, ntk_alphas).movedim(1, 2)
        else:
            freqs = self.rope_embedder(img_ids, ntk_alphas).movedim(1, 2)
        return freqs

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        is_uncond=False,
        current_step_percentage=0.0,
        current_step=0,
        last_step=0,
        total_steps=50,
        clip_fea=None,
        y=None,
        device=torch.device('cuda'),
        freqs=None,
        enhance_enabled=False,
        pred_id=None,
        control_lora_enabled=False,
        vace_data=None,
        camera_embed=None,
        unianim_data=None,
        fps_embeds=None,
        fun_ref=None,
        fun_camera=None,
        audio_proj=None,
        audio_scale=1.0,
        uni3c_data=None,
        controlnet=None,
        add_cond=None,
        attn_cond=None,
        nag_params={},
        nag_context=None,
        multitalk_audio=None,
        ref_target_masks=None,
        inner_t=None,
        standin_input=None,
        fantasy_portrait_input=None,
        phantom_ref=None,
        reverse_time=False,
        ntk_alphas = [1.0, 1.0, 1.0],
        mtv_motion_tokens=None,
        mtv_motion_rotary_emb=None,
        mtv_freqs=None,
        mtv_strength=1.0,
        s2v_audio_input=None,
        s2v_ref_latent=None,
        s2v_audio_scale=1.0,
        s2v_ref_motion=None,
        s2v_pose=None,
        s2v_motion_frames=[1, 0],
        humo_audio=None,
        humo_audio_scale=1.0,
        wananim_pose_latents=None, 
        wananim_face_pixel_values=None,
        wananim_pose_strength=1.0,
        wananim_face_strength=1.0,

    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        # Stand-In only used on first positive pass, then cached in kv_cache
        if is_uncond or current_step > 0: 
            standin_input = None
        
        # MTV Crafter motion projection
        if mtv_motion_tokens is not None:
            bs, motion_seq_len =  mtv_motion_tokens.shape[0], mtv_motion_tokens.shape[1]
            mtv_motion_tokens = torch.cat([mtv_motion_tokens, self.pad_motion_tokens.to(mtv_motion_tokens).expand(bs, motion_seq_len, -1)], dim=-1)

        # Fantasy Portrait
        adapter_proj = ip_scale = None
        if fantasy_portrait_input is not None:
            if fantasy_portrait_input['start_percent'] <= current_step_percentage <= fantasy_portrait_input['end_percent']:
                adapter_proj = fantasy_portrait_input.get("adapter_proj", None)
                ip_scale = fantasy_portrait_input.get("strength", 1.0)

        if self.lora_scheduling_enabled:
            for name, submodule in self.named_modules():
                if isinstance(submodule, nn.Linear):
                    if hasattr(submodule, 'step'):
                        submodule.step = current_step

        #s2v
        if self.model_type == 's2v' and s2v_audio_input is not None:
            if is_uncond:
                s2v_audio_input = s2v_audio_input * 0 # to match original code
            s2v_audio_input = torch.cat([s2v_audio_input[..., 0:1].repeat(1, 1, 1, s2v_motion_frames[0]), s2v_audio_input], dim=-1)
            
            audio_emb_res = self.casual_audio_encoder(s2v_audio_input)
            if self.enable_adain:
                audio_emb_global, audio_emb = audio_emb_res
                self.audio_emb_global = audio_emb_global[:, s2v_motion_frames[1]:].clone()
            else:
                audio_emb = audio_emb_res
            merged_audio_emb = audio_emb[:, s2v_motion_frames[1]:, :]

        # params
        device = self.patch_embedding.weight.device

        if freqs is not None and freqs.device != device:
           freqs = freqs.to(device)

        _, F, H, W = x[0].shape
            
        if y is not None:
            if hasattr(self, "randomref_embedding_pose") and unianim_data is not None:
                if unianim_data['start_percent'] <= current_step_percentage <= unianim_data['end_percent']:
                    random_ref_emb = unianim_data["random_ref"]
                    if random_ref_emb is not None:
                        y[0].add_(random_ref_emb, alpha=unianim_data["strength"])
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]
        
        #uni3c controlnet
        if uni3c_data is not None:
            render_latent = uni3c_data["render_latent"].to(x[0].dtype)
            hidden_states = x[0].unsqueeze(0).clone().float()
            if hidden_states.shape[1] == 16: #T2V work around
                hidden_states = torch.cat([hidden_states, torch.zeros_like(hidden_states[:, :4])], dim=1)
            render_latent = torch.cat([hidden_states[:, :20], render_latent], dim=1)

        # embeddings
        if control_lora_enabled:
            self.expanded_patch_embedding.to(device)
            x = [
            self.expanded_patch_embedding(u.unsqueeze(0).to(torch.float32)).to(x[0].dtype)
            for u in x
            ]
        else:
            self.original_patch_embedding.to(self.main_device)
            x = [
            self.original_patch_embedding(u.unsqueeze(0).to(torch.float32)).to(x[0].dtype)
            for u in x
            ]

        # WanAnimate
        motion_vec = None
        if wananim_face_pixel_values is not None:
            motion_vec = self.wananimate_face_embedding(wananim_face_pixel_values).to(x[0].dtype)

        if wananim_pose_latents is not None:
            x = self.wananimate_pose_embedding(x, wananim_pose_latents, strength=wananim_pose_strength)

        # s2v pose embedding
        if s2v_pose is not None:
            x[0] = x[0] + self.cond_encoder(s2v_pose.to(self.cond_encoder.weight.dtype)).to(x[0].dtype)

        # Fun camera
        if self.control_adapter is not None and fun_camera is not None:
            fun_camera = self.control_adapter(fun_camera)
            x = [u + v for u, v in zip(x, fun_camera)]

        grid_sizes = torch.stack([torch.tensor(u.shape[2:], device=device, dtype=torch.long) for u in x])
        original_grid_sizes = grid_sizes.clone()
        x = [u.flatten(2).transpose(1, 2) for u in x]

        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.int32)
        assert seq_lens.max() <= seq_len

        cond_mask_weight = None
        if self.trainable_cond_mask is not None:
            cond_mask_weight = self.trainable_cond_mask.weight.to(x[0]).unsqueeze(1).unsqueeze(1)

        self.original_seq_len = x[0].shape[1]

        if add_cond is not None:
            add_cond = self.add_conv_in(add_cond.to(self.add_conv_in.weight.dtype)).to(x[0].dtype)
            add_cond = add_cond.flatten(2).transpose(1, 2)
            x[0] = x[0] + self.add_proj(add_cond)
        attn_cond_shape = None
        if attn_cond is not None:
            attn_cond_shape = attn_cond.shape
            grid_sizes = torch.stack([torch.tensor([u[0] + 1, u[1], u[2]]) for u in grid_sizes]).to(grid_sizes.device)
            attn_cond = self.attn_conv_in(attn_cond.to(self.attn_conv_in.weight.dtype)).to(x[0].dtype)
            attn_cond = attn_cond.flatten(2).transpose(1, 2)
            x[0] = torch.cat([x[0], attn_cond], dim=1)
            seq_len += attn_cond.size(1)
            for block in self.blocks:
                block.self_attn.mask_map = MaskMap(video_token_num=seq_len, num_frame=F+1)

        if self.ref_conv is not None and fun_ref is not None:
            fun_ref = self.ref_conv(fun_ref).flatten(2).transpose(1, 2)
            grid_sizes = torch.stack([torch.tensor([u[0] + 1, u[1], u[2]]) for u in grid_sizes]).to(grid_sizes.device)
            seq_len += fun_ref.size(1)
            F += 1
            x = [torch.cat([_fun_ref.unsqueeze(0), u], dim=1) for _fun_ref, u in zip(fun_ref, x)]

        end_ref_latent=None
        if s2v_ref_latent is not None:
            end_ref_latent = s2v_ref_latent.squeeze(0)
        elif phantom_ref is not None:
            end_ref_latent = phantom_ref
            F += end_ref_latent.size(1)
        if end_ref_latent is not None:
            end_ref_latent_frames = end_ref_latent.size(1)
            end_ref_latent = self.original_patch_embedding(end_ref_latent.unsqueeze(0).to(torch.float32)).to(x[0].dtype)
            end_ref_latent = end_ref_latent.flatten(2).transpose(1, 2)
            if cond_mask_weight is not None:
                end_ref_latent = end_ref_latent + cond_mask_weight[1]
            grid_sizes = torch.stack([torch.tensor([u[0] + end_ref_latent_frames, u[1], u[2]]) for u in grid_sizes]).to(grid_sizes.device)
            end_ref_latent_seq_len = end_ref_latent.size(1)
            seq_len += end_ref_latent_seq_len
            x = [torch.cat([u, end_ref_latent.unsqueeze(0)], dim=1) for end_ref_latent, u in zip(end_ref_latent, x)]

        
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                    dim=1) for u in x
        ])

        if self.trainable_cond_mask is not None:
            x = x + cond_mask_weight[0]

        # StandIn LoRA input
        x_ip = None
        freq_offset = 0
        if standin_input is not None:
            ip_image = standin_input["ip_image_latent"]

            if ip_image.dim() == 6 and ip_image.shape[3] == 1:
                ip_image = ip_image.squeeze(1)

            ip_image_patch = self.original_patch_embedding(ip_image.float()).to(x.dtype)
            f_ip, h_ip, w_ip = ip_image_patch.shape[2:]
            x_ip = ip_image_patch.flatten(2).transpose(1, 2)  # [B, N, D]
            freq_offset = standin_input["freq_offset"]

        if freqs is None: #comfy rope
            current_shape = (F, H, W)
           
            has_cond = attn_cond is not None

            if (self.cached_freqs is not None and 
                self.cached_shape == current_shape and 
                self.cached_cond == has_cond and
                self.cached_rope_k == self.rope_embedder.k and
                self.cached_ntk_alphas == ntk_alphas
                ):
                freqs = self.cached_freqs
            else:
                freqs = self.rope_encode_comfy(F, H, W, freq_offset=freq_offset, ntk_alphas=ntk_alphas, attn_cond_shape=attn_cond_shape, device=x.device, dtype=x.dtype)
                if s2v_ref_latent is not None:
                    freqs_ref = self.rope_encode_comfy(
                        s2v_ref_latent.shape[2], 
                        s2v_ref_latent.shape[3], 
                        s2v_ref_latent.shape[4], 
                        t_start=max(30, F + 9), device=x.device, dtype=x.dtype)
                    freqs = torch.cat([freqs, freqs_ref], dim=1)

                self.cached_freqs = freqs
                self.cached_shape = current_shape
                self.cached_cond = has_cond
                self.cached_rope_k = self.rope_embedder.k
                self.cached_ntk_alphas = ntk_alphas

        # Stand-In RoPE frequencies
        if x_ip is not None:
            # Generate RoPE frequencies for x_ip
            h_len = (H + 1) // 2
            w_len = (W + 1) // 2
            ip_img_ids = torch.zeros((f_ip, h_ip, w_ip, 3), device=x.device, dtype=x.dtype)
            ip_img_ids[:, :, :, 0] = ip_img_ids[:, :, :, 0] + torch.linspace(0, f_ip - 1, steps=f_ip, device=x.device, dtype=x.dtype).reshape(-1, 1, 1)
            ip_img_ids[:, :, :, 1] = ip_img_ids[:, :, :, 1] + torch.linspace(h_len + freq_offset, h_len + freq_offset + h_ip - 1, steps=h_ip, device=x.device, dtype=x.dtype).reshape(1, -1, 1)
            ip_img_ids[:, :, :, 2] = ip_img_ids[:, :, :, 2] + torch.linspace(w_len + freq_offset, w_len + freq_offset + w_ip - 1, steps=w_ip, device=x.device, dtype=x.dtype).reshape(1, 1, -1)
            ip_img_ids = repeat(ip_img_ids, "t h w c -> b (t h w) c", b=1)
            freqs_ip = self.rope_embedder(ip_img_ids).movedim(1, 2)

        # EchoShot cross attn freqs
        inner_c = None
        if inner_t is not None:
            d = self.dim // self.num_heads
            self.cross_freqs = rope_params(100, d).to(device=x.device)

        if s2v_ref_motion is not None:
            motion_encoded, freqs_motion = self.frame_packer(s2v_ref_motion, self)
            motion_encoded = motion_encoded + cond_mask_weight[2]
            x = torch.cat([x, motion_encoded], dim=1)
            freqs = torch.cat([freqs, freqs_motion], dim=1)

        # time embeddings
        if t.dim() == 2:
            b, f = t.shape
            expanded_timesteps = True
        else:
            expanded_timesteps = False

        if self.zero_timestep:
            t = torch.cat([t, torch.zeros([1], dtype=t.dtype, device=t.device)])

        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t.flatten()).to(x.dtype))  # b, dim
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))  # b, 6, dim

        #S2V zero timestep
        if self.zero_timestep:
            e = e[:-1]
            zero_e0 = e0[-1:]
            e0 = e0[:-1]
            e0 = torch.cat([
                e0.unsqueeze(2),
                zero_e0.unsqueeze(2).repeat(e0.size(0), 1, 1, 1)
            ], dim=2)
            e0 = [e0, self.original_seq_len]

        if x_ip is not None:
            timestep_ip = torch.zeros_like(t)  # [B] with 0s
            t_ip = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep_ip.flatten()).to(x.dtype))  # b, dim )
            e0_ip = self.time_projection(t_ip).unflatten(1, (6, self.dim))

        if fps_embeds is not None:
            fps_embeds = torch.tensor(fps_embeds, dtype=torch.long, device=device)

            fps_emb = self.fps_embedding(fps_embeds).to(e0.dtype)
            if expanded_timesteps:
                e0 = e0 + self.fps_projection(fps_emb).unflatten(1, (6, self.dim)).repeat(t.shape[1], 1, 1)
            else:
                e0 = e0 + self.fps_projection(fps_emb).unflatten(1, (6, self.dim))

        if expanded_timesteps:
            e = e.view(b, f, 1, 1, self.dim).expand(b, f, grid_sizes[0][1], grid_sizes[0][2], self.dim)
            e0 = e0.view(b, f, 1, 1, 6, self.dim).expand(b, f, grid_sizes[0][1], grid_sizes[0][2], 6, self.dim)
            
            e = e.flatten(1, 3)
            e0 = e0.flatten(1, 3)
            
            e0 = e0.transpose(1, 2)
            if not e0.is_contiguous():
                e0 = e0.contiguous()
            
            e = e.to(self.offload_device, non_blocking=self.use_non_blocking)

        
        #context (text embedding)
        if hasattr(self, "text_embedding") and context != []:
            if self.offload_txt_emb:
                self.text_embedding.to(self.main_device)

            if inner_t is not None:
                if nag_context is not None:
                    raise NotImplementedError("nag_context is not supported with EchoShot")
                inner_c = [[u.shape[0] for u in context]]

            context = self.text_embedding(
                torch.stack([
                    torch.cat(
                        [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                    for u in context
                ]).to(x.dtype))
            
            # NAG
            if nag_context is not None:
                nag_context = self.text_embedding(
                torch.stack([
                    torch.cat(
                        [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                    for u in nag_context
                ]).to(x.dtype))
            
            if self.offload_txt_emb:
                self.text_embedding.to(self.offload_device, non_blocking=self.use_non_blocking)
        else:
            context = None

        clip_embed = None
        if clip_fea is not None and hasattr(self, "img_emb"):
            clip_fea = clip_fea.to(self.main_device)
            if self.offload_img_emb:
                self.img_emb.to(self.main_device)
            clip_embed = self.img_emb(clip_fea)  # bs x 257 x dim
            #context = torch.concat([context_clip, context], dim=1)
            if self.offload_img_emb:
                self.img_emb.to(self.offload_device, non_blocking=self.use_non_blocking)

        # MultiTalk
        if multitalk_audio is not None:
            self.multitalk_audio_proj.to(self.main_device)
            audio_cond = multitalk_audio.to(device=x.device, dtype=x.dtype)
            first_frame_audio_emb_s = audio_cond[:, :1, ...] 
            latter_frame_audio_emb = audio_cond[:, 1:, ...] 
            latter_frame_audio_emb = rearrange(latter_frame_audio_emb, "b (n_t n) w s c -> b n_t n w s c", n=4) 
            middle_index = self.multitalk_audio_proj.seq_len // 2
            latter_first_frame_audio_emb = latter_frame_audio_emb[:, :, :1, :middle_index+1, ...] 
            latter_first_frame_audio_emb = rearrange(latter_first_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c") 
            latter_last_frame_audio_emb = latter_frame_audio_emb[:, :, -1:, middle_index:, ...] 
            latter_last_frame_audio_emb = rearrange(latter_last_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c") 
            latter_middle_frame_audio_emb = latter_frame_audio_emb[:, :, 1:-1, middle_index:middle_index+1, ...] 
            latter_middle_frame_audio_emb = rearrange(latter_middle_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c") 
            latter_frame_audio_emb_s = torch.concat([latter_first_frame_audio_emb, latter_middle_frame_audio_emb, latter_last_frame_audio_emb], dim=2) 
            multitalk_audio_embedding = self.multitalk_audio_proj(first_frame_audio_emb_s, latter_frame_audio_emb_s) 
            human_num = len(multitalk_audio_embedding)
            multitalk_audio_embedding = torch.concat(multitalk_audio_embedding.split(1), dim=2).to(x.dtype)
            self.multitalk_audio_proj.to(self.offload_device)

        # convert ref_target_masks to token_ref_target_masks
        token_ref_target_masks = None
        if ref_target_masks is not None:
            ref_target_masks = ref_target_masks.unsqueeze(0).to(torch.float32) 
            token_ref_target_masks = nn.functional.interpolate(ref_target_masks, size=(H // 2, W // 2), mode='nearest') 
            token_ref_target_masks = token_ref_target_masks.squeeze(0)
            token_ref_target_masks = (token_ref_target_masks > 0)
            token_ref_target_masks = token_ref_target_masks.view(token_ref_target_masks.shape[0], -1) 
            token_ref_target_masks = token_ref_target_masks.to(x.dtype).to(device)

        humo_audio_input = None
        if humo_audio is not None:
            humo_audio_input = self.audio_proj(humo_audio.unsqueeze(0)).permute(0, 3, 1, 2)

            humo_audio_seq_len = torch.tensor(humo_audio.shape[2] * humo_audio_input.shape[3], device=device)
            humo_audio_input = humo_audio_input.flatten(2).transpose(1, 2) # 1, t*32, 1536
            pad_len = int(humo_audio_seq_len - humo_audio_input.size(1))
            if pad_len > 0:
                humo_audio_input = torch.nn.functional.pad(humo_audio_input, (0, 0, 0, pad_len))

        should_calc = True
        #TeaCache
        if self.enable_teacache and self.teacache_start_step <= current_step <= self.teacache_end_step:
            accumulated_rel_l1_distance = torch.tensor(0.0, dtype=torch.float32, device=device)
            if pred_id is None:
                pred_id = self.teacache_state.new_prediction(cache_device=self.cache_device)
                should_calc = True                
            else:
                previous_modulated_input = self.teacache_state.get(pred_id)['previous_modulated_input']
                previous_modulated_input = previous_modulated_input.to(device)
                previous_residual = self.teacache_state.get(pred_id)['previous_residual']
                accumulated_rel_l1_distance = self.teacache_state.get(pred_id)['accumulated_rel_l1_distance']

                if self.teacache_use_coefficients:
                    rescale_func = np.poly1d(self.teacache_coefficients[self.teacache_mode])
                    temb = e if self.teacache_mode == 'e' else e0
                    accumulated_rel_l1_distance += rescale_func((
                        (temb.to(device) - previous_modulated_input).abs().mean() / previous_modulated_input.abs().mean()
                        ).cpu().item())
                    del temb
                else:
                    temb_relative_l1 = relative_l1_distance(previous_modulated_input, e0)
                    accumulated_rel_l1_distance = accumulated_rel_l1_distance.to(e0.device) + temb_relative_l1
                    del temb_relative_l1


                if accumulated_rel_l1_distance < self.rel_l1_thresh:
                    should_calc = False
                else:
                    should_calc = True
                    accumulated_rel_l1_distance = torch.tensor(0.0, dtype=torch.float32, device=device)
                accumulated_rel_l1_distance = accumulated_rel_l1_distance.to(self.cache_device)

            previous_modulated_input = e.to(self.cache_device).clone() if (self.teacache_use_coefficients and self.teacache_mode == 'e') else e0.to(self.cache_device).clone()
           
            if not should_calc:
                x = x.to(previous_residual.dtype) + previous_residual.to(x.device)
                self.teacache_state.update(
                    pred_id,
                    accumulated_rel_l1_distance=accumulated_rel_l1_distance,
                )
                self.teacache_state.get(pred_id)['skipped_steps'].append(current_step)

        # MagCache
        if self.enable_magcache and self.magcache_start_step <= current_step <= self.magcache_end_step:
            if pred_id is None:
                pred_id = self.magcache_state.new_prediction(cache_device=self.cache_device)
                should_calc = True
            else:
                accumulated_ratio = self.magcache_state.get(pred_id)['accumulated_ratio']
                accumulated_err = self.magcache_state.get(pred_id)['accumulated_err']
                accumulated_steps = self.magcache_state.get(pred_id)['accumulated_steps']

                calibration_len = len(self.magcache_ratios) // 2
                cur_mag_ratio = self.magcache_ratios[int((current_step*(calibration_len/total_steps)))]

                accumulated_ratio *= cur_mag_ratio
                accumulated_err += np.abs(1-accumulated_ratio)
                accumulated_steps += 1

                self.magcache_state.update(
                    pred_id,
                    accumulated_ratio=accumulated_ratio,
                    accumulated_steps=accumulated_steps,
                    accumulated_err=accumulated_err
                )

                if accumulated_err<=self.magcache_thresh and accumulated_steps<=self.magcache_K:
                    should_calc = False
                    x += self.magcache_state.get(pred_id)['residual_cache'].to(x.device)
                    self.magcache_state.get(pred_id)['skipped_steps'].append(current_step)
                else:
                    should_calc = True
                    self.magcache_state.update(
                        pred_id,
                        accumulated_ratio=1.0,
                        accumulated_steps=0,
                        accumulated_err=0
                    )

        # EasyCache
        if self.enable_easycache and self.easycache_start_step <= current_step <= self.easycache_end_step:
            if pred_id is None:
                pred_id = self.easycache_state.new_prediction(cache_device=self.cache_device)
                should_calc = True
            else:
                state = self.easycache_state.get(pred_id)
                previous_raw_input = state.get('previous_raw_input')
                previous_raw_output = state.get('previous_raw_output')
                cache = state.get('cache')
                accumulated_error = state.get('accumulated_error')
                k = state.get('k', 1)

                if previous_raw_input is not None and previous_raw_output is not None:
                    raw_input = x.clone()
                    # Calculate input change
                    raw_input_change = (raw_input - previous_raw_input.to(raw_input.device)).abs().mean()

                    output_norm = (previous_raw_output.to(x.device)).abs().mean()

                    combined_pred_change = (raw_input_change / output_norm) * k

                    accumulated_error += combined_pred_change

                    # Predict output change
                    if accumulated_error < self.easycache_thresh:
                        should_calc = False
                        x = raw_input + cache.to(x.device)
                        state['skipped_steps'].append(current_step)
                    else:
                        should_calc = True
                else:
                    should_calc = True

        if self.enable_easycache:
            original_x = x.clone().to(self.cache_device)
        if should_calc:
            if self.enable_teacache or self.enable_magcache:
                original_x = x.clone().to(self.cache_device)

            if hasattr(self, "dwpose_embedding") and unianim_data is not None:
                if unianim_data['start_percent'] <= current_step_percentage <= unianim_data['end_percent']:
                    dwpose_emb = rearrange(unianim_data['dwpose'], 'b c f h w -> b (f h w) c').contiguous()
                    x.add_(dwpose_emb, alpha=unianim_data['strength'])
            # arguments
            kwargs = dict(
                e=e0,
                seq_lens=seq_lens,
                grid_sizes=grid_sizes,
                freqs=freqs,
                context=context,
                clip_embed=clip_embed,
                current_step=current_step,
                last_step=last_step,
                video_attention_split_steps=self.video_attention_split_steps,
                camera_embed=camera_embed,
                audio_proj=audio_proj,
                num_latent_frames = F,
                original_seq_len=self.original_seq_len,
                enhance_enabled=enhance_enabled,
                audio_scale=audio_scale,
                nag_params=nag_params, nag_context=nag_context,
                is_uncond = is_uncond,
                multitalk_audio_embedding=multitalk_audio_embedding if multitalk_audio is not None else None,
                ref_target_masks=token_ref_target_masks if multitalk_audio is not None else None,
                human_num=human_num if multitalk_audio is not None else 0,
                inner_t=inner_t, inner_c=inner_c,
                cross_freqs=self.cross_freqs if inner_t is not None else None,
                freqs_ip=freqs_ip if x_ip is not None else None,
                e_ip=e0_ip if x_ip is not None else None,
                adapter_proj=adapter_proj,
                ip_scale=ip_scale,
                reverse_time=reverse_time,
                mtv_motion_tokens=mtv_motion_tokens, mtv_motion_rotary_emb=mtv_motion_rotary_emb, mtv_strength=mtv_strength, mtv_freqs=mtv_freqs,
                humo_audio_input=humo_audio_input,
                humo_audio_scale=humo_audio_scale,
            )
            
            if vace_data is not None:
                vace_hint_list = []
                vace_scale_list = []
                if isinstance(vace_data[0], dict):
                    for data in vace_data:
                        if (data["start"] <= current_step_percentage <= data["end"]) or \
                            (data["end"] > 0 and current_step == 0 and current_step_percentage >= data["start"]):

                            vace_hints = self.forward_vace(x, data["context"], data["seq_len"], kwargs)
                            vace_hint_list.append(vace_hints)
                            vace_scale_list.append(data["scale"][current_step])
                else:
                    vace_hints = self.forward_vace(x, vace_data, seq_len, kwargs)
                    vace_hint_list.append(vace_hints)
                    vace_scale_list.append(1.0)
                
                kwargs['vace_hints'] = vace_hint_list
                kwargs['vace_context_scale'] = vace_scale_list

            #uni3c controlnet
            uni3c_controlnet_states = None
            if uni3c_data is not None:
                if (uni3c_data["start"] <= current_step_percentage <= uni3c_data["end"]) or \
                            (uni3c_data["end"] > 0 and current_step == 0 and current_step_percentage >= uni3c_data["start"]):
                    self.controlnet.to(self.main_device)
                    with torch.autocast(device_type=mm.get_autocast_device(device), dtype=x.dtype, enabled=True):
                        uni3c_controlnet_states = self.controlnet(
                            render_latent=render_latent.to(self.main_device, self.controlnet.dtype), 
                            render_mask=uni3c_data["render_mask"], 
                            camera_embedding=uni3c_data["camera_embedding"], 
                            temb=e.to(self.main_device),
                            device=self.offload_device)
                    self.controlnet.to(self.offload_device)

            # Asynchronous block offloading with CUDA streams and events
            if torch.cuda.is_available():
                cuda_stream = None #torch.cuda.Stream(device=device, priority=0) # todo causes issues on some systems
                events = [torch.cuda.Event() for _ in self.blocks]
                swap_start_idx = len(self.blocks) - self.blocks_to_swap if self.blocks_to_swap > 0 else len(self.blocks)
            else:
                cuda_stream = None
                events = None
                swap_start_idx = len(self.blocks)

            for b, block in enumerate(self.blocks):
                # Prefetch blocks if enabled
                if self.prefetch_blocks > 0:
                    for prefetch_offset in range(1, self.prefetch_blocks + 1):
                        prefetch_idx = b + prefetch_offset
                        if prefetch_idx < len(self.blocks) and self.blocks_to_swap > 0 and prefetch_idx >= swap_start_idx:
                            context_mgr = torch.cuda.stream(cuda_stream) if torch.cuda.is_available() else nullcontext()
                            with context_mgr:
                                self.blocks[prefetch_idx].to(self.main_device, non_blocking=self.use_non_blocking)
                                if events is not None:
                                    events[prefetch_idx].record(cuda_stream)
                if self.block_swap_debug:
                    transfer_start = time.perf_counter()
                # Wait for block to be ready
                if b >= swap_start_idx and self.blocks_to_swap > 0:
                    if self.prefetch_blocks > 0 and events is not None:
                        if not events[b].query():
                            events[b].synchronize()
                    block.to(self.main_device)
                if self.block_swap_debug:
                    transfer_end = time.perf_counter()
                    transfer_time = transfer_end - transfer_start
                    compute_start = time.perf_counter()
                #skip layer guidance
                if self.slg_blocks is not None:
                    if b in self.slg_blocks and is_uncond:
                        if self.slg_start_percent <= current_step_percentage <= self.slg_end_percent:
                            continue
                x, x_ip = block(x, x_ip=x_ip, **kwargs) #run block
                if self.audio_injector is not None and s2v_audio_input is not None:
                    x = self.audio_injector_forward(b, x, merged_audio_emb, scale=s2v_audio_scale) #s2v
                if block.has_face_fuser_block and motion_vec is not None:
                    x = self.wananimate_forward(block, x, motion_vec, strength=wananim_face_strength)
                if self.block_swap_debug:
                    compute_end = time.perf_counter()
                    compute_time = compute_end - compute_start
                    to_cpu_transfer_start = time.perf_counter()
                if b >= swap_start_idx and self.blocks_to_swap > 0:
                    block.to(self.offload_device, non_blocking=self.use_non_blocking)
                if self.block_swap_debug:
                    to_cpu_transfer_end = time.perf_counter()
                    to_cpu_transfer_time = to_cpu_transfer_end - to_cpu_transfer_start
                    log.info(f"Block {b}: transfer_time={transfer_time:.4f}s, compute_time={compute_time:.4f}s, to_cpu_transfer_time={to_cpu_transfer_time:.4f}s")

                #uni3c controlnet
                if uni3c_controlnet_states is not None and b < len(uni3c_controlnet_states):
                    x[:, :self.original_seq_len] += uni3c_controlnet_states[b].to(x) * uni3c_data["controlnet_weight"]
                #controlnet
                if (controlnet is not None) and (b % controlnet["controlnet_stride"] == 0) and (b // controlnet["controlnet_stride"] < len(controlnet["controlnet_states"])):
                    x[:, :self.original_seq_len] += controlnet["controlnet_states"][b // controlnet["controlnet_stride"]].to(x) * controlnet["controlnet_weight"]

            if self.enable_teacache and (self.teacache_start_step <= current_step <= self.teacache_end_step) and pred_id is not None:
                self.teacache_state.update(
                    pred_id,
                    previous_residual=(x.to(original_x.device) - original_x),
                    accumulated_rel_l1_distance=accumulated_rel_l1_distance,
                    previous_modulated_input=previous_modulated_input
                )
            elif self.enable_magcache and (self.magcache_start_step <= current_step <= self.magcache_end_step) and pred_id is not None:
                self.magcache_state.update(
                    pred_id,
                    residual_cache=(x.to(original_x.device) - original_x)
                )
            elif self.enable_easycache and (self.easycache_start_step <= current_step <= self.easycache_end_step) and pred_id is not None:
                x_out = x.clone().to(original_x.device)
                output_change = (x_out - original_x).abs().mean()
                input_change = (original_x - x_out).abs().mean()
                self.easycache_state.update(
                    pred_id,
                    previous_raw_input=original_x,
                    previous_raw_output=x_out,
                    cache=x.to(original_x.device) - original_x,
                    k = output_change / input_change,
                    accumulated_error = 0.0
                )

        if self.enable_easycache and (self.easycache_start_step <= current_step <= self.easycache_end_step) and pred_id is not None:
            self.easycache_state.update(
                pred_id,
                previous_raw_output=x.clone(),
            )
                
        if self.ref_conv is not None and fun_ref is not None:
            fun_ref_length = fun_ref.size(1)
            x = x[:, fun_ref_length:]
            grid_sizes = torch.stack([torch.tensor([u[0] - 1, u[1], u[2]]) for u in grid_sizes]).to(grid_sizes.device)
        
        if end_ref_latent is not None:
            end_ref_latent_length = end_ref_latent.size(1)
            x = x[:, :-end_ref_latent_length]
            grid_sizes = torch.stack([torch.tensor([u[0] - end_ref_latent_frames, u[1], u[2]]) for u in grid_sizes]).to(grid_sizes.device)

        if attn_cond is not None:
            x = x[:, :self.original_seq_len]
            grid_sizes = torch.stack([torch.tensor([u[0] - 1, u[1], u[2]]) for u in grid_sizes]).to(grid_sizes.device)

        
        x = x[:, :self.original_seq_len]

        x = self.head(x, e.to(x.device))
        x = self.unpatchify(x, original_grid_sizes) # type: ignore[arg-type]
        x = [u.float() for u in x]
        return (x, pred_id) if pred_id is not None else (x, None)

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[: math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum("fhwpqrc->cfphqwr", u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out
