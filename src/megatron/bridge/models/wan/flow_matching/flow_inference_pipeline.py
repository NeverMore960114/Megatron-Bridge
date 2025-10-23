# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
from tqdm import tqdm

from megatron.bridge.models.wan.wan_model import WanModel
from megatron.bridge.models.wan.wan_provider import WanModelProvider
from megatron.bridge.models.wan.modules.t5 import T5EncoderModel
from megatron.bridge.models.wan.modules import WanVAE
from megatron.bridge.models.wan.inference.utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from megatron.bridge.models.wan.inference.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from megatron.core.dist_checkpointing.validation import StrictHandling
from megatron.core import dist_checkpointing, parallel_state
from torch.nn import functional as F

import math
from typing import Tuple, Union

class FlowInferencePipeline:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_cpu=False,
        tensor_parallel_size=1,
        context_parallel_size=1,
        pipeline_parallel_size=1,
        sequence_parallel=False,
        pipeline_dtype=torch.float32,
    ):
        r"""
        Initializes the FlowInferencePipeline with the given parameters.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.tensor_parallel_size = tensor_parallel_size
        self.context_parallel_size = context_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.sequence_parallel = sequence_parallel
        self.pipeline_dtype = pipeline_dtype
        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=None)

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size        
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        wan_checkpoint_dir = os.path.join(checkpoint_dir, "iter_0000000")
        self.model = self.setup_model_from_checkpoint(wan_checkpoint_dir)

        self.sp_size = 1

        if dist.is_initialized():
            dist.barrier()
        self.model.to(self.device)

        self.sample_neg_prompt = config.sample_neg_prompt


    def patchify(self, x, patch_size):
        """
        Convert a list of reconstructed video tensor into patch embeddings (inverse of `unpatchify`).

        Args:
            x (list[torch.Tensor]): list of tensors, each with shape [C, F * pF, H * pH, W * pW]
            patch_size (tuple): (pF, pH, pW)

        Returns:
            torch.Tensor: shape [num_patches, C * prod(patch_size)],
                        where num_patches = F * H * W
        """
        out = []
        for u in x:
            c, F_pF, H_pH, W_pW = u.shape
            pF, pH, pW = patch_size
            assert F_pF % pF == 0 and H_pH % pH == 0 and W_pW % pW == 0, \
                "Spatial dimensions must be divisible by patch size."

            F, H, W = F_pF // pF, H_pH // pH, W_pW // pW

            # split spatial dims into (grid, patch) and reorder to match original patch layout:
            # start: (C, F_pF, H_pW, W_pW)
            # reshape -> (C, F, pF, H, pH, W, pW)
            # permute -> (F, H, W, pF, pH, pW, C)
            # DEBUGGING
            t = u.reshape(c, F, pF, H, pH, W, pW)
            # t = u.reshape(c, F, pF, W, pW, H, pH)
            t = t.permute(1, 3, 5, 0, 2, 4, 6)

            num_patches = F * H * W
            out.append(t.reshape(num_patches, c * (pF * pH * pW)))
        return out
        

    def unpatchify(self, x: torch.Tensor, grid_sizes: torch.Tensor, out_dim: int) -> torch.Tensor:
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (Tensor):
                Tensor of patchified features, with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            Tensor:
                # Reconstructed video tensor with shape [C_out, F, H / 8, W / 8]
                # ??? list of tensors, because each sample in the batch has a different video shape, the original video shape is determined by the grid_sizes.
                list[Tensor]: list of tensors, each with shape [C_out, F, H / 8, W / 8]
        """

        c = out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        # because the video shapes are different for each sample in the batch, we cannot stack the videos into a single tensor.
        # out = torch.stack(out, dim=0)
        return out


    def setup_model_from_checkpoint(self, checkpoint_dir):

        # def init_distributed(tp_size: int = 1, pp_size: int = 1, cp_size: int = 1):
        #     rank = int(os.environ.get("LOCAL_RANK", 0))
        #     world_size = int(os.environ.get("WORLD_SIZE", 1))
        #     torch.cuda.set_device(rank % torch.cuda.device_count())
        #     torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
        #     parallel_state.initialize_model_parallel(tp_size, pp_size, context_parallel_size=cp_size)
        # init_distributed(self.tensor_parallel_size, self.pipeline_parallel_size, self.context_parallel_size)

        provider = WanModelProvider()
        provider.tensor_model_parallel_size = self.tensor_parallel_size
        provider.pipeline_model_parallel_size = self.pipeline_parallel_size
        provider.context_parallel_size = self.context_parallel_size
        provider.sequence_parallel = self.sequence_parallel
        print(f"provider.sequence_parallel: {provider.sequence_parallel}")
        provider.pipeline_dtype = self.pipeline_dtype
        # Once all overrides are set, finalize the model provider to ensure the post initialization logic is run
        provider.finalize()
        provider.initialize_model_parallel(seed=0)
        

        ## Method 1: Read from megatron checkpoint
        from megatron.bridge.training.model_load_save import load_megatron_model as _load_megatron_model
        model = _load_megatron_model(
            checkpoint_dir,
            mp_overrides={
                "tensor_model_parallel_size": self.tensor_parallel_size,
                "pipeline_model_parallel_size": self.pipeline_parallel_size,
                "context_parallel_size": self.context_parallel_size,
                "sequence_parallel": self.sequence_parallel,
                "pipeline_dtype": self.pipeline_dtype,
            },
        )
        if isinstance(model, list):
            model = model[0]
        # ## Method 2: Read from megatron checkpoint
        # model = provider.provide_distributed_model(wrap_with_ddp=False)
        ## Method 3 (not loading checkpoint)
        # model = provider.provide()

        return model


    def grid_sizes_calculation(
        self,
        input_shape: Tuple[int, int, int],  # (D_in, H_in, W_in)
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        dilation: Union[int, Tuple[int, int, int]] = 1
    ) -> Tuple[int, int, int]:
        """
        Compute the (f,h,w) output spatial/temporal dimensions of a Conv3d patch embedder.

        Args:
            input_shape: (D_in, H_in, W_in)
            kernel_size, stride, padding, dilation of the Conv3d patch embedder: either int or 3-tuple

        Returns:
            (D_out, H_out, W_out)
        """
        
        def to_tuple(x):
            return (x, x, x) if isinstance(x, int) else x
        
        kernel_size = to_tuple(kernel_size)
        stride = to_tuple(stride)
        padding = to_tuple(padding)
        dilation = to_tuple(dilation)
        
        D_in, H_in, W_in = input_shape
        
        def calc_out(in_size, k, s, p, d):
            return math.floor((in_size + 2*p - d*(k - 1) - 1) / s + 1)
        
        D_out = calc_out(D_in, kernel_size[0], stride[0], padding[0], dilation[0])
        H_out = calc_out(H_in, kernel_size[1], stride[1], padding[1], dilation[1])
        W_out = calc_out(W_in, kernel_size[2], stride[2], padding[2], dilation[2])
        
        return [D_out, H_out, W_out]


    def forward_pp_step(
        self,
        latent_model_input: torch.Tensor,
        grid_sizes: list[Tuple[int, int, int]],
        max_video_seq_len: int,
        timestep: torch.Tensor,
        arg_c: dict,        
    ) -> torch.Tensor:
        """One decode step supporting pipeline parallelism for batch_size=1.

        Returns a tensor containing the noise prediction.
        """

        from megatron.core import parallel_state
        from megatron.core.inference.communication_utils import broadcast_from_last_pipeline_stage, recv_from_prev_pipeline_rank_, send_to_next_pipeline_rank

        pp_world_size = parallel_state.get_pipeline_model_parallel_world_size()
        is_pp_first = parallel_state.is_pipeline_first_stage(ignore_virtual=True)
        is_pp_last = parallel_state.is_pipeline_last_stage(ignore_virtual=True)

        # TP-only or single-rank
        if pp_world_size == 1:
            noise_pred_pp = self.model(
                latent_model_input,
                grid_sizes=grid_sizes,
                t=timestep,
                **arg_c)
            return noise_pred_pp

        # Pipeline-parallel path
        hidden_size = self.model.config.hidden_size
        batch_size = latent_model_input.shape[1]
        noise_pred_pp_shape = list(latent_model_input.shape)
        print(f"batch_size: {batch_size}")

        # DEBUGGING
        # we should bring x unpatchify out of the model
        # x_after_patch_embedding_shape = [16, 3, 104, 60]   # ????
        # when bring unpatchified out, for pp communicate last stage to first stage, this should be
        # x_after_patch_embedding_shape = [max_video_seq_len, batch_size, (ph pw pt C)]

        if is_pp_first:
            # First stage: compute multimodal + first PP slice, send activations, then receive sampled token
            hidden_states = self.model(
                latent_model_input,
                grid_sizes=grid_sizes,
                t=timestep,
                **arg_c)
            print(f"[rank {torch.distributed.get_rank()}] Got here! - self.model")
            send_to_next_pipeline_rank(hidden_states)
            print(f"[rank {torch.distributed.get_rank()}] Got here! - hidden_states.shape: {hidden_states.shape} - hidden_states.dtype: {hidden_states.dtype}")
            print(f"[rank {torch.distributed.get_rank()}] Got here! - send_to_next_pipeline_rank")

            noise_pred_pp = broadcast_from_last_pipeline_stage(noise_pred_pp_shape, dtype=torch.float32)
            return noise_pred_pp

        if is_pp_last:
            # Last stage: recv activations, run final slice + output, sample, broadcast
            recv_buffer = torch.empty(
                (max_video_seq_len, batch_size, hidden_size),
                dtype=next(self.model.parameters()).dtype,
                device=latent_model_input[0].device,
            )
            recv_from_prev_pipeline_rank_(recv_buffer)
            # DEBUGGING
            recv_buffer = recv_buffer.to(torch.bfloat16) # ????
            self.model.set_input_tensor(recv_buffer)
            noise_pred_pp = self.model(
                latent_model_input,
                grid_sizes=grid_sizes,
                t=timestep,
                **arg_c)

            
            print("noise_pred_pp_shape: ", noise_pred_pp_shape)

            noise_pred_pp = broadcast_from_last_pipeline_stage(noise_pred_pp_shape, dtype=noise_pred_pp.dtype, tensor=noise_pred_pp.contiguous())
            return noise_pred_pp

        # Intermediate stages: recv -> run local slice -> send -> receive broadcast token
        recv_buffer = torch.empty(
            (max_video_seq_len, batch_size, hidden_size),
            dtype=next(self.model.parameters()).dtype,
            device=latent_model_input[0].device,
        )
        print(f"[rank {torch.distributed.get_rank()}] Got here! - recv_buffer.shape: {recv_buffer.shape} - recv_buffer.dtype: {recv_buffer.dtype}")
        recv_from_prev_pipeline_rank_(recv_buffer)
        print(f"[rank {torch.distributed.get_rank()}] Got here! - recv_from_prev_pipeline_rank_")
        # DEBUGGING
        recv_buffer = recv_buffer.to(torch.bfloat16) # ????
        self.model.set_input_tensor(recv_buffer)
        print(f"[rank {torch.distributed.get_rank()}] Got here! - self.model.set_input_tensor")
        hidden_states = self.model(
            latent_model_input,
            grid_sizes=grid_sizes,
            t=timestep,
            **arg_c)
        send_to_next_pipeline_rank(hidden_states)

        noise_pred_pp = broadcast_from_last_pipeline_stage(noise_pred_pp_shape, dtype=torch.float32)
        return noise_pred_pp


    def generate(self,
                 prompts,
                 sizes,
                 frame_nums,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            size (tupele[`int`], *optional*, defaults to (1280,720)):
                Controls video resolution, (width,height).
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """
    
        # DEBUGGING
        run_debug = True

        # size = sizes[0]
        # input_prompt = prompts[0]
        # frame_num = frame_nums[0]
        
        # preprocess
        target_shapes = []
        for size, frame_num in zip(sizes, frame_nums):
            target_shapes.append((self.vae.model.z_dim, (frame_num - 1) // self.vae_stride[0] + 1,
                                size[1] // self.vae_stride[1],
                                size[0] // self.vae_stride[2]))

        max_video_seq_len = 0
        seq_lens = []
        for target_shape in target_shapes:
            seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                                (self.patch_size[1] * self.patch_size[2]) *
                                target_shape[1] / self.sp_size) * self.sp_size
            seq_lens.append(seq_len)
        max_video_seq_len = max(seq_lens)

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        ## process context
        context_max_len = 512
        context_lens = []
        contexts = []
        contexts_null = []
        for prompt in prompts:
            if not self.t5_cpu:
                self.text_encoder.model.to(self.device)
                context = self.text_encoder([prompt], self.device)[0]
                context_null = self.text_encoder([n_prompt], self.device)[0]
                if offload_model:
                    self.text_encoder.model.cpu()
            else:
                context = self.text_encoder([prompt], torch.device('cpu'))[0].to(self.device)
                context_null = self.text_encoder([n_prompt], torch.device('cpu'))[0].to(self.device)
            context_lens.append(context_max_len) # all samples have the same context_max_len
            contexts.append(context)
            contexts_null.append(context_null)
        # pad to context_max_len tokens, and stack to a tensor of shape [s, b, hidden]
        contexts = [F.pad(context, (0, 0, 0, context_max_len - context.shape[0])) for context in contexts]
        contexts_null = [F.pad(context_null, (0, 0, 0, context_max_len - context_null.shape[0])) for context_null in contexts_null]
        contexts = torch.stack(contexts, dim=1)
        contexts_null = torch.stack(contexts_null, dim=1)



        ## setup noise
        noises = []
        for target_shape in target_shapes:
            noises.append(
                torch.randn(
                    target_shape[0],
                    target_shape[1],
                    target_shape[2],
                    target_shape[3],
                    dtype=torch.float32,
                    device=self.device,
                    generator=seed_g)
            )

        # DEBUGGING
        print("[DEBUG] noises[0].shape - noises[0].dtype - noises[0].mean() - noises[0].std() - noises[0].norm():", noises[0].shape, noises[0].dtype, noises[0].mean(), noises[0].std(), noises[0].norm())
        print("[DEBUG] noises[0]:", noises[0])

        # calculate grid_sizes
        grid_sizes = [self.grid_sizes_calculation(
            input_shape =u.shape[1:], 
            kernel_size=self.model.patch_size, 
            stride=self.model.patch_size,
            ) for u in noises]
        grid_sizes = torch.tensor(grid_sizes, dtype=torch.long)


        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

            if sample_solver == 'unipc':
                # Create a prototype scheduler to compute shared timesteps
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps

                # Instantiate per-sample schedulers so each sample maintains its own state
                batch_size_for_schedulers = len(noises)
                schedulers = []
                for _ in range(batch_size_for_schedulers):
                    s = FlowUniPCMultistepScheduler(
                        num_train_timesteps=self.num_train_timesteps,
                        shift=1,
                        use_dynamic_shifting=False)
                    s.set_timesteps(sampling_steps, device=self.device, shift=shift)
                    schedulers.append(s)
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latents = noises

            from megatron.core.packed_seq_params import PackedSeqParams
            cu_q = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(seq_lens), dim=0)])
            cu_q = cu_q.to(torch.int32).to(self.device)
            cu_kv_self = cu_q
            cu_kv_cross = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(context_lens), dim=0)])
            cu_kv_cross = cu_kv_cross.to(torch.int32).to(self.device)
            packed_seq_params = {
                "self_attention": PackedSeqParams(
                    cu_seqlens_q=cu_q,
                    cu_seqlens_kv=cu_kv_self,
                    qkv_format="sbhd",
                ),
                "cross_attention": PackedSeqParams(
                    cu_seqlens_q=cu_q,
                    cu_seqlens_kv=cu_kv_cross,
                    qkv_format="sbhd",
                ),
            }
            

            arg_c = {'context': contexts, 'max_seq_len': max_video_seq_len, 'packed_seq_params': packed_seq_params}
            arg_null = {'context': contexts_null, 'max_seq_len': max_video_seq_len, 'packed_seq_params': packed_seq_params}

            for _, t in enumerate(tqdm(timesteps)):

                batch_size = len(latents)

                # patchify latents
                # ??? when batch_size > 1, we need to pad to have same length
                unpatchified_latents = latents
                latents = self.patchify(latents, self.patch_size)
                # pad to have same length
                for i in range(batch_size):
                    latents[i] = F.pad(latents[i], (0, 0, 0, max_video_seq_len - latents[i].shape[0]))
                latents = torch.stack(latents, dim=1)


                latent_model_input = latents
                timestep = [t] * batch_size
                timestep = torch.stack(timestep)

                # DEBUGGING
                if run_debug and torch.distributed.get_rank()==0:
                    print(f"[DEBUG] [rank {torch.distributed.get_rank()}] contexts.shape: {contexts.shape}")
                    print(f"[DEBUG] [rank {torch.distributed.get_rank()}] max_video_seq_len: {max_video_seq_len}")
                    print(f"[DEBUG] [rank {torch.distributed.get_rank()}] grid_sizes: {grid_sizes}")
                    print(f"[DEBUG] [rank {torch.distributed.get_rank()}] latent_model_input.shape: {latent_model_input.shape}")
                    print(f"[DEBUG] [rank {torch.distributed.get_rank()}] timestep.shape: {timestep.shape}")


                self.model.to(self.device)
                noise_pred_cond = self.forward_pp_step(
                    latent_model_input, grid_sizes=grid_sizes, max_video_seq_len=max_video_seq_len, timestep=timestep, arg_c=arg_c)

                noise_pred_uncond = self.forward_pp_step(
                    latent_model_input, grid_sizes=grid_sizes, max_video_seq_len=max_video_seq_len, timestep=timestep, arg_c=arg_null)


                # noise_pred = noise_pred_uncond + guide_scale * (
                #     noise_pred_cond - noise_pred_uncond)

                # DEBUGGING
                unpatchified_noise_pred_cond = noise_pred_cond
                unpatchified_noise_pred_cond = unpatchified_noise_pred_cond.transpose(0, 1) # bring sbhd -> bshd
                # when unpatchifying, the code will truncate the padded videos into the original video shape, based on the grid_sizes. ???
                unpatchified_noise_pred_cond = self.unpatchify(unpatchified_noise_pred_cond, grid_sizes, self.vae.model.z_dim)

                unpatchified_noise_pred_uncond = noise_pred_uncond
                unpatchified_noise_pred_uncond = unpatchified_noise_pred_uncond.transpose(0, 1) # bring sbhd -> bshd
                # when unpatchifying, the code will truncate the padded videos into the original video shape, based on the grid_sizes. ???
                unpatchified_noise_pred_uncond = self.unpatchify(unpatchified_noise_pred_uncond, grid_sizes, self.vae.model.z_dim)

                # DEBUGGING
                if run_debug and torch.distributed.get_rank()==0:
                    print(f"[DEBUG] unpatchified_noise_pred_cond[0].shape - unpatchified_noise_pred_cond[0].dtype - unpatchified_noise_pred_cond[0].mean() - unpatchified_noise_pred_cond[0].std() - unpatchified_noise_pred_cond[0].norm(): {unpatchified_noise_pred_cond[0].shape} - {unpatchified_noise_pred_cond[0].dtype} - {unpatchified_noise_pred_cond[0].mean()} - {unpatchified_noise_pred_cond[0].std()} - {unpatchified_noise_pred_cond[0].norm()}")
                    print(f"[DEBUG] unpatchified_noise_pred_uncond[0].shape - unpatchified_noise_pred_uncond[0].dtype - unpatchified_noise_pred_uncond[0].mean() - unpatchified_noise_pred_uncond[0].std() - unpatchified_noise_pred_uncond[0].norm(): {unpatchified_noise_pred_uncond[0].shape} - {unpatchified_noise_pred_uncond[0].dtype} - {unpatchified_noise_pred_uncond[0].mean()} - {unpatchified_noise_pred_uncond[0].std()} - {unpatchified_noise_pred_uncond[0].norm()}")


                noise_preds = []
                for i in range(batch_size):
                    noise_pred = unpatchified_noise_pred_uncond[i] + guide_scale * (
                        unpatchified_noise_pred_cond[i] - unpatchified_noise_pred_uncond[i])
                    noise_preds.append(noise_pred)

                    # unpatchified_noise_pred_uncond = unpatchified_noise_pred_uncond[0]
                    # unpatchified_noise_pred_cond = unpatchified_noise_pred_cond[0]

                    # noise_pred = unpatchified_noise_pred_uncond + guide_scale * (
                    #     unpatchified_noise_pred_cond - unpatchified_noise_pred_uncond)

                # # DEBUGGING
                # # we will be running unpatchify here???
                # # x0 = latents
                # if run_debug and torch.distributed.get_rank()==0:
                #     print(f"[DEBUG] [rank {torch.distributed.get_rank()}] (before unpatchify) noise_pred_cond.shape: {noise_pred_cond.shape}")
                #     print(f"[DEBUG] [rank {torch.distributed.get_rank()}] (before unpatchify) noise_pred_uncond.shape: {noise_pred_uncond.shape}")
                # noise_pred_cond = noise_pred_cond.transpose(0, 1)
                # noise_pred_cond = self.unpatchify(noise_pred_cond, grid_sizes, self.vae.model.z_dim)
                # noise_pred_cond = noise_pred_cond.transpose(0, 1)
                # noise_pred_uncond = noise_pred_uncond.transpose(0, 1)
                # noise_pred_uncond = self.unpatchify(noise_pred_uncond, grid_sizes, self.vae.model.z_dim)
                # noise_pred_uncond = noise_pred_uncond.transpose(0, 1)
                # if run_debug and torch.distributed.get_rank()==0:
                #     print(f"[DEBUG] [rank {torch.distributed.get_rank()}] (after unpatchify) noise_pred_cond.shape: {noise_pred_cond.shape}")
                #     print(f"[DEBUG] [rank {torch.distributed.get_rank()}] (after unpatchify) noise_pred_uncond.shape: {noise_pred_uncond.shape}")
                #     print(stop_here)

                # # we run unpatchify here, but unpatchify should be run seprately for each sample in the batch, because the video shape is different for each sample in the batch.
                # # ??? when batch_size > 1, we need to run sample_scheduler.step seprately for each sample in the batch.
                # noise_pred = noise_pred.transpose(0, 1) # bring sbhd -> bshd
                # noise_pred = self.unpatchify(noise_pred, grid_sizes, self.vae.model.z_dim)

                # print("[DEBUG] len(noise_pred): ", len(noise_pred))
                # print("[DEBUG] len(unpatchified_latents): ", len(unpatchified_latents))
                # print("[DEBUG] noise_pred[0].shape - noise_pred[0].dtype - noise_pred[0].mean() - noise_pred[0].std() - noise_pred[0].norm(): ", noise_pred[0].shape, noise_pred[0].dtype, noise_pred[0].mean(), noise_pred[0].std(), noise_pred[0].norm())
                # print("[DEBUG] unpatchified_latents[0].shape - unpatchified_latents[0].dtype - unpatchified_latents[0].mean() - unpatchified_latents[0].std() - unpatchified_latents[0].norm(): ", unpatchified_latents[0].shape, unpatchified_latents[0].dtype, unpatchified_latents[0].mean(), unpatchified_latents[0].std(), unpatchified_latents[0].norm())

                # latents = []
                # for i in range(len(noise_pred)):
                #     temp_x0 = sample_scheduler.step(
                #         noise_pred[i].unsqueeze(0),
                #         t,
                #         unpatchified_latents[i].unsqueeze(0),
                #         return_dict=False,
                #         generator=seed_g)[0]
                #     latents.append(temp_x0.squeeze(0))

                # print("len(latents): ", len(latents))
                # print("latents[0].shape: ", latents[0].shape)

                # latents = unpatchified_latents
                # print(f"[DEBUG] noise_pred.shape - noise_pred.dtype - noise_pred.mean() - noise_pred.std() - noise_pred.norm(): {noise_pred.shape} - {noise_pred.dtype} - {noise_pred.mean()} - {noise_pred.std()} - {noise_pred.norm()}")
                # print(f"[DEBUG] latents[0].shape - latents[0].dtype - latents[0].mean() - latents[0].std() - latents[0].norm(): {latents[0].shape} - {latents[0].dtype} - {latents[0].mean()} - {latents[0].std()} - {latents[0].norm()}")
                # print(f"[DEBUG] noise_pred: {noise_pred}")
                # print(f"[DEBUG] latents[0]: {latents[0]}")

                print("batch_size: ", batch_size)

                # step and update latents
                latents = []
                for i in range(batch_size):

                    # DEBUGGING
                    if run_debug and torch.distributed.get_rank()==0:
                        print("[DEBUG] len(unpatchified_latents): ", len(unpatchified_latents))
                        print("[DEBUG] len(noise_preds): ", len(noise_preds))
                        print("[DEBUG] unpatchified_latents[i].shape - unpatchified_latents[i].dtype - unpatchified_latents[i].mean() - unpatchified_latents[i].std() - unpatchified_latents[i].norm(): ", unpatchified_latents[i].shape, unpatchified_latents[i].dtype, unpatchified_latents[i].mean(), unpatchified_latents[i].std(), unpatchified_latents[i].norm())
                        print("[DEBUG] noise_preds[i].shape - noise_preds[i].dtype - noise_preds[i].mean() - noise_preds[i].std() - noise_preds[i].norm(): ", noise_preds[i].shape, noise_preds[i].dtype, noise_preds[i].mean(), noise_preds[i].std(), noise_preds[i].norm())


                    if sample_solver == 'unipc':
                        temp_x0 = schedulers[i].step(
                            noise_preds[i].unsqueeze(0),
                            t,
                            unpatchified_latents[i].unsqueeze(0),
                            return_dict=False,
                            generator=seed_g)[0]
                    else:
                        temp_x0 = sample_scheduler.step(
                            noise_preds[i].unsqueeze(0),
                            t,
                            unpatchified_latents[i].unsqueeze(0),
                            return_dict=False,
                            generator=seed_g)[0]
                    latents.append(temp_x0.squeeze(0))

            # # DEBUGGING
            # # we will be running unpatchify here???
            # # x0 = latents
            # x0 = self.unpatchify(latents, grid_sizes)

            # # loop through each sample in the batch
            # videos = []
            # if offload_model:
            #     self.model.cpu()
            #     torch.cuda.empty_cache()
            # x0 = latents
            # if self.rank == 0:
            #     videos = self.vae.decode(x0)

            # DEBUGGING
            print("[DEBUG] len(latents): ", len(latents))
            print("[DEBUG] latents[0].shape - latents[0].dtype - latents[0].mean() - latents[0].std() - latents[0].norm(): ", latents[0].shape, latents[0].dtype, latents[0].mean(), latents[0].std(), latents[0].norm())
            print("[DEBUG] latents[0]: ", latents[0])

            x0 = latents
            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()
            if self.rank == 0:
                videos = self.vae.decode(x0)
            else:
                videos = None


            # # DEBUGGING
            # print("len(latents): ", len(latents))
            # print("latents[0].shape - latents[0].dtype - latents[0].mean() - latents[0].std() - latents[0].norm(): ", latents[0].shape, latents[0].dtype, latents[0].mean(), latents[0].std(), latents[0].norm())
            # print("latents[0]: ", latents[0])
            # print("len(videos): ", len(videos))
            if videos is not None:
                print("len(videos): ", len(videos))
                print("[DEBUG] videos[0].shape - videos[0].dtype - videos[0].mean() - videos[0].std() - videos[0].norm(): ", videos[0].shape, videos[0].dtype, videos[0].mean(), videos[0].std(), videos[0].norm())
                print("[DEBUG] videos[0]: ", videos[0])

        del noises, latents
        if sample_solver == 'unipc':
            del schedulers
        else:
            del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos if self.rank == 0 else None
