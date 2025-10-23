# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import contextlib
import inspect
import logging
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, Literal, Optional, Union

import torch
from megatron.core import parallel_state
from megatron.core.models.gpt import GPTModel as MCoreGPTModel
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.transformer import ModuleSpec
from megatron.bridge.models.transformer_config import TransformerConfig
from megatron.bridge.models.DiTModel.dit_utils import dynamic_import

from megatron.bridge.models.model_provider import ModelProviderMixin
from megatron.bridge.utils import fusions
from megatron.bridge.utils.vocab_utils import calculate_padded_vocab_size
from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.bridge.models.wan.wan_model import WanModel

logger = logging.getLogger(__name__)

@dataclass
class WanModelProvider(TransformerConfig, ModelProviderMixin[VisionModule]):
    crossattn_emb_size: int = 1536
    add_bias_linear: bool = True
    gated_linear_unit: bool = False

    num_layers: int = 30
    hidden_size: int = 1536
    ffn_hidden_size: int = 8960
    max_img_h: int = 80
    max_img_w: int = 80
    max_frames: int = 34
    patch_spatial: int = 2
    patch_temporal: int = 1
    num_attention_heads: int = 12
    layernorm_epsilon = 1e-6
    normalization = "RMSNorm"
    qk_layernorm_per_head: bool = False
    layernorm_zero_centered_gamma = False

    fp16_lm_cross_entropy: bool = False
    parallel_output: bool = True
    share_embeddings_and_output_weights: bool = True

    hidden_dropout: float = 0
    attention_dropout: float = 0

    bf16: bool = False
    params_dtype: torch.dtype = torch.float32

    vae_module: str = "nemo_vfm.diffusion.vae.diffusers_vae.AutoencoderKLVAE"
    vae_path: str = None
    sigma_data: float = 0.5

    in_channels: int = 16
    out_channels: int = 16

    replicated_t_embedder = True
    qkv_format: str = 'sbhd'

    # DEBUGGING
    # adding more attributes
    text_dim: int = 4096
    patch_size: list = field(default_factory=lambda: [1, 2, 2])
    freq_dim: int = 256
    out_dim: int = 16
    text_len: int = 512 



    # DEBUGGING
    # unused, we just set because bridge training requires this for LLMs
    seq_length: int = 1024
    vocab_size: int = None
    make_vocab_size_divisible_by: int = 128


    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> WanModel:
        vp_size = self.virtual_pipeline_model_parallel_size
        if vp_size:
            p_size = self.pipeline_model_parallel_size
            assert (self.num_layers // p_size) % vp_size == 0, (
                "Make sure the number of model chunks is the same across all pipeline stages."
            )

        model = WanModel

        return model(
            self,
            fp16_lm_cross_entropy=self.fp16_lm_cross_entropy,
            parallel_output=self.parallel_output,
            pre_process=parallel_state.is_pipeline_first_stage(),
            post_process=parallel_state.is_pipeline_last_stage(),
            max_img_h=self.max_img_h,
            max_img_w=self.max_img_w,
            max_frames=self.max_frames,
            patch_spatial=self.patch_spatial,
        )

    def configure_vae(self):
        return dynamic_import(self.vae_module)(self.vae_path)