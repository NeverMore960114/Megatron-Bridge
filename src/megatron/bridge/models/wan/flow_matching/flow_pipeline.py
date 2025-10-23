# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Callable, Dict, Optional, Tuple, List

import numpy as np
import torch
import torch.distributed
from megatron.core import parallel_state
# from megatron.bridge.models.DiTModel.sampler.context_parallel import cat_outputs_cp ???
from torch import Tensor
from diffusers import WanPipeline

class FlowPipeline:
    """
    FlowPipeline is a class that implements a diffusion model pipeline for video generation. It includes methods for
    initializing the pipeline, encoding and decoding video data, performing training steps, denoising, and generating
    samples.
    Attributes:
        ...
    Methods:
        ...
    """

    def __init__(
        self,
        model_id="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        vae=None,
        seed=1234,
    ):
        """
        Initializes the FlowPipeline with the given parameters.

        Args:
            net: The DiT model.
            vae: The Video Tokenizer (optional).
            seed (int): Random seed for reproducibility.

        Attributes:
            vae: The Video Tokenizer.
            net: The DiT model.
            _noise_generator: Generator for noise.
            seed (int): Random seed for reproducibility.
            input_data_key (str): Key for input data.
            input_image_key (str): Key for input images.
            tensor_kwargs (dict): Tensor keyword arguments for device and dtype.
        """
        self.vae = vae

        self.seed = seed
        self._noise_generator = None

        self.input_data_key = "video"
        self.input_image_key = "images_1024"
        self.tensor_kwargs = {"device": "cuda", "dtype": torch.bfloat16}

        pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.float32)
        self.scheduler = pipe.scheduler


    def _initialize_generators(self):
        """
        Initializes the random number generators for noise

        This method sets up a generator:
        1. A PyTorch generator for noise, seeded with a combination of the base seed and the data parallel rank.

        Returns:
            None
        """
        noise_seed = self.seed + 100 * parallel_state.get_data_parallel_rank(with_context_parallel=True)
        noise_level_seed = self.seed + 100 * parallel_state.get_data_parallel_rank(with_context_parallel=False)
        self._noise_generator = torch.Generator(device="cuda")
        self._noise_generator.manual_seed(noise_seed)

    def training_step(
        self, model, data_batch: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """
        Performs a single training step for the diffusion model.

        This method is responsible for executing one iteration of the model's training. It involves:
        1. Adding noise to the input data using the SDE process.
        2. Passing the noisy data through the network to generate predictions.
        3. Computing the loss based on the difference between the predictions and the original data.

        Args:
            data_batch (dict): raw data batch draw from the training data loader.

        Returns:
            A tuple with the output batch and the computed loss.
        """

        # DEBUGGING
        run_debug = False
        if run_debug and torch.distributed.get_rank()==0:
            print("---- Sample info [FlowPipeline.training_step] ----")
            print(f"data_batch['video_latents'] shape: {data_batch['video_latents'].shape}")
            print(f"data_batch['context_embeddings'] shape: {data_batch['context_embeddings'].shape}")
            print(f"data_batch['loss_mask'] shape: {data_batch['loss_mask'].shape}")
            print(f"data_batch['grid_sizes']: {data_batch['grid_sizes']}")
            print(f"data_batch['packed_seq_params']: {data_batch['packed_seq_params']}")
            print(f"data_batch['max_video_seq_len']: {data_batch['max_video_seq_len']}")


        video_latents = data_batch['video_latents']
        max_video_seq_len = data_batch['max_video_seq_len']
        context_embeddings = data_batch['context_embeddings']
        grid_sizes = data_batch['grid_sizes']
        packed_seq_params = data_batch['packed_seq_params']


        # Get the input data to noise and denoise~(image, video) and the corresponding conditioner.
        self.model = model
        

        # Get timesteps
        batch_size = video_latents.shape[1]
        device = video_latents.device
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (batch_size,), device=device)

        # Generate noise
        # shape of latents is [S, B, (C pF pH pW)]
        noise_batch = torch.randn_like(video_latents)


        # DEBUGGING
        if run_debug and torch.distributed.get_rank()==0:
            print("---- Sample info [FlowPipeline.training_step] ----")
            print(f"noise_batch shape: {noise_batch.shape}")
            print(f"timesteps shape: {timesteps.shape}")
            print(f"video_latents shape: {video_latents.shape}")
            print("--------------------------------")

        # ??? can this add_noise method used for videos of different sizes and just padding?
        #  => it should be, because the main formula is: noisy_latents = alpha_t * original_samples + sigma_t * noise
        # Apply scheduler noise based on timesteps
        # DEBUGGING
        # bring to shape [batch_size, ...] to run add_noise
        noisy_latents = self.scheduler.add_noise(video_latents.transpose(0, 1), noise_batch.transpose(0, 1), timesteps)
        noisy_latents = noisy_latents.transpose(0, 1)

        # Pass through model
        # noise only needed at the last stage
        if parallel_state.is_pipeline_last_stage():
            output_batch, loss = self.compute_loss(
                noisy_latents, noise_batch, timesteps, context_embeddings, grid_sizes, packed_seq_params, max_video_seq_len
            )

            return output_batch, loss
        else:
            hidden_states = self.compute_loss(
                noisy_latents, timesteps, context_embeddings, grid_sizes, packed_seq_params, max_video_seq_len
            )
            return hidden_states

    # def get_data_and_condition(self, data_batch: dict[str, Tensor]) -> Tuple[Tensor]:
    #     """
    #     Retrieves data and conditioning for model input.

    #     Args:
    #         data_batch: Batch of input data.

    #     Returns:
    #         ...
    #     """
    #     ...
    #     return None

    def compute_loss(
        self, 
        video_latents: torch.Tensor, 
        noise_batch: torch.Tensor, 
        timesteps: torch.Tensor, 
        context_embeddings: torch.Tensor, 
        grid_sizes: List[Tuple[int, int, int]], 
        packed_seq_params: dict,
        max_video_seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the loss for the given latents, timesteps, context_embeddings, grid_sizes, and packed_seq_params.
        """

        # ??? the shape of latents is [S, B, (ph pw pt C)]
        # ??? the shape of noise is [S, B, (ph pw pt C)]
        # loss_mask is [S, B], will be transffered in WanForwardStep to combine with loss to get the final loss

        # condition would be:
        # t5_text_embeddings, t5_text_mask, seq_len_q, seq_len_kv, pos_ids, latent_shape, grid_sizes
        # the shape of t5_text_embeddings is [S, B, (ph pw pt C)]
        # the shape of t5_text_mask is [S, B]
        # the shape of seq_len_q is [B]
        # the shape of seq_len_kv is [B]
        # the shape of pos_ids is [S, B, (ph pw pt C)]
        # the shape of latent_shape is [B, 4]
        # the shape of grid_sizes is [B, 3]

        # Pass through model
        if parallel_state.is_pipeline_last_stage():
            model_predict = self.model(
                x = video_latents,
                grid_sizes = grid_sizes,
                t = timesteps,
                context = context_embeddings,
                max_seq_len = max_video_seq_len,
                packed_seq_params=packed_seq_params,
            )

            # Compute target based on prediction type
            if self.scheduler.config.prediction_type == "epsilon":
                target = noise_batch
            elif self.scheduler.config.prediction_type == "v_prediction":
                target = self.scheduler.get_velocity(latents, noise_batch, timesteps)
            elif self.scheduler.config.prediction_type == "flow_prediction":
                # Flow matching
                target = video_latents - noise_batch
            else:
                raise ValueError(f"Unknown prediction type: {self.scheduler.config.prediction_type}")

            # Compute loss
            loss = torch.nn.functional.mse_loss(model_predict, target, reduction="mean")

            return model_predict, loss

        else:
            hidden_states = self.model(
                x = video_latents,
                grid_sizes = grid_sizes,
                t = timesteps,
                context = context_embeddings,
                max_seq_len = max_video_seq_len,
                packed_seq_params=packed_seq_params,
            )

            return hidden_states
