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

# pylint: disable=C0115,C0116,C0301

from typing import Dict, Literal, Optional, Tuple, List, Union
import copy

import math
import torch
import torch.cuda.amp as amp
import torch.nn as nn
from megatron.core import parallel_state, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.transformer_block import TransformerBlock, TransformerBlockSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import make_sharded_tensor_for_checkpoint
from megatron.bridge.models.wan.wan_layer_spec import (
    get_wan_block_with_transformer_engine_spec as WanLayerWithAdaLNspec,
    get_vace_base_block_with_transformer_engine_spec as VACEBaseLayerspec,
    get_vace_context_block_with_transformer_engine_spec as VACEContextLayerspec,
)
from megatron.bridge.models.wan.wan_layer_spec import WanLayerNorm
from torch import Tensor
from .rope_utils import Wan3DRopeEmbeddings

from contextlib import nullcontext
from megatron.core.fp4_utils import get_fp4_context
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.enums import Fp8Recipe
from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
from megatron.core.utils import (
    WrappedTensor,
    deprecate_inference_params,
    get_pg_rank,
    make_viewless_tensor,
)

try:
    import transformer_engine.pytorch as te  # pylint: disable=unused-import

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

try:
    import apex  # pylint: disable=unused-import

    HAVE_APEX = True
except ImportError:
    HAVE_APEX = False

get_cpu_offload_context = None
te_checkpoint = None

if HAVE_TE:
    from megatron.core.extensions.transformer_engine import (
        TENorm,
        get_cpu_offload_context,
        te_checkpoint,
    )

    LayerNormImpl = TENorm

elif HAVE_APEX:
    LayerNormImpl = FusedLayerNorm

else:
    from megatron.core.transformer.torch_norm import WrappedTorchNorm

    LayerNormImpl = WrappedTorchNorm

class BaseTransformerBlock(TransformerBlock):
    def __init__(
        self,
        config: TransformerConfig,
        spec: Union[TransformerBlockSubmodules, ModuleSpec],
        post_layer_norm: bool = True,
        pre_process: bool = True,
        post_process: bool = True,
        pg_collection: ProcessGroupCollection = None,
        vp_stage: Optional[int] = None,
    ):
        # Pass block id and context_scale
        self.vace_layers = [i for i in range(0, config.num_layers, 2)] if config.vace_layers is None else config.vace_layers
        print(self.vace_layers)
        assert 0 in self.vace_layers
        self.vace_layers_mapping = {i: n for n, i in enumerate(self.vace_layers)}
        
        super().__init__(
            config=config, 
            spec=spec,
            post_layer_norm=post_layer_norm,
            pre_process=pre_process,
            post_process=post_process,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
        )
    
    def _build_layers(self):
        # Transformer layers.
        # @jcasper can we improve how we deal with layer_number?
        # currently it's only used in CoreAttention?
        # if self.apply_query_key_layer_scaling:
        #     coeff = self.layer_number
        #     self.norm_factor *= coeff
        def build_layer(layer_spec, layer_number):
            global_layer_number = layer_number + get_transformer_layer_offset(
                self.config, self.vp_stage, get_pg_rank(self.pg_collection.pp)
            )  # 1-based index
            if self.config.heterogeneous_block_specs:
                layer_config = self.config.get_config_for_layer(global_layer_number)
            else:
                layer_config = self.config

            # Get appropriate quantization context (FP8 and FP4 are mutually exclusive)
            if layer_config.fp8:
                quantization_context = get_fp8_context(
                    layer_config, global_layer_number - 1, is_init=True
                )
            elif layer_config.fp4:
                quantization_context = get_fp4_context(
                    layer_config, global_layer_number - 1, is_init=True
                )
            else:
                quantization_context = nullcontext()

            with quantization_context:
                module = build_module(
                    layer_spec,
                    config=layer_config,
                    layer_number=layer_number,
                    pg_collection=self.pg_collection,
                    vp_stage=self.vp_stage,
                )
                idx = global_layer_number - 1
                if idx in self.vace_layers:
                    module.idx = self.vace_layers_mapping[idx]
                    module.context_scale = self.config.context_scale
                else:
                    module.idx = None
            return module

        # offset is implicit in TransformerLayer
        self.layers = torch.nn.ModuleList(
            [
                build_layer(layer_spec, i + 1)
                for i, layer_spec in enumerate(self.submodules.layer_specs)
            ]
        )

        # @TODO: add back account_for_embedding_in_pipeline_split (see issue #293)
        # In pipeline parallelism, we want to add this LN only to the last stage of the pipeline
        # self.post_process and self.post_layer_norm guide this behavior
        if self.submodules.layer_norm and self.post_process and self.post_layer_norm:
            self.final_layernorm = build_module(
                self.submodules.layer_norm,
                config=self.config,
                hidden_size=self.config.hidden_size,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.final_layernorm = None  # Either this or nn.Identity
            
    def _checkpointed_forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        context: Tensor,
        context_mask: Tensor,
        context_signal: Tensor,
        rotary_pos_emb: Tensor,
        attention_bias: Tensor,
        packed_seq_params: PackedSeqParams,
        use_inner_quantization_context: bool,
    ):
        """Forward method with activation checkpointing."""

        def custom(start: int, end: int):
            def custom_forward(
                hidden_states, attention_mask, context, context_mask, context_signal, rotary_pos_emb
            ):
                for index in range(start, end):
                    layer = self._get_layer(index)

                    # Get appropriate inner quantization context
                    if use_inner_quantization_context:
                        if self.config.fp8:
                            inner_quantization_context = get_fp8_context(
                                self.config, layer.layer_number - 1
                            )
                        # TODO: check if fp4 is supported in this case
                        elif self.config.fp4:
                            inner_quantization_context = get_fp4_context(
                                self.config, layer.layer_number - 1
                            )
                        else:
                            inner_quantization_context = nullcontext()
                    else:
                        inner_quantization_context = nullcontext()

                    with inner_quantization_context:
                        hidden_states, context = layer(
                            hidden_states=hidden_states,
                            attention_mask=attention_mask,
                            context=context,
                            context_mask=context_mask,
                            context_signal=context_signal,
                            rotary_pos_emb=rotary_pos_emb,
                            attention_bias=attention_bias,
                            inference_context=None,
                            packed_seq_params=packed_seq_params,
                        )
                return hidden_states, context

            return custom_forward

        def checkpoint_handler(forward_func):
            """Determines whether to use the `te_checkpoint` or `tensor_parallel.checkpoint`"""
            # TODO: check if fp4 is supported in this case
            if self.config.fp8 or self.config.fp4:
                return te_checkpoint(
                    forward_func,
                    self.config.distribute_saved_activations,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    self.pg_collection.tp,
                    hidden_states,
                    attention_mask,
                    context,
                    context_mask,
                    context_signal,
                    rotary_pos_emb,
                )
            else:
                return tensor_parallel.checkpoint(
                    forward_func,
                    self.config.distribute_saved_activations,
                    hidden_states,
                    attention_mask,
                    context,
                    context_mask,
                    context_signal,
                    rotary_pos_emb,
                )

        if self.config.recompute_method == 'uniform':
            # Uniformly divide the total number of Transformer layers and checkpoint
            # the input activation of each divided chunk.
            # A method to further reduce memory usage reducing checkpoints.
            layer_idx = 0
            while layer_idx < self.num_layers_per_pipeline_rank:
                hidden_states, context = checkpoint_handler(
                    custom(layer_idx, layer_idx + self.config.recompute_num_layers)
                )

                layer_idx += self.config.recompute_num_layers

        elif self.config.recompute_method == 'block':
            # Checkpoint the input activation of only a set number of individual
            # Transformer layers and skip the rest.
            # A method fully use the device memory removing redundant re-computation.
            recompute_skip_num_layers = 0
            for layer_idx in range(self.num_layers_per_pipeline_rank):
                # Skip recomputation when input grad computation is not needed.
                # Need to have at least one input tensor with gradient computation
                # for re-enterant autograd engine.
                # TODO: check if fp4 is supported in this case
                if (self.config.fp8 or self.config.fp4) and not hidden_states.requires_grad:
                    recompute_skip_num_layers += 1
                if (
                    layer_idx >= recompute_skip_num_layers
                    and layer_idx < self.config.recompute_num_layers + recompute_skip_num_layers
                ):
                    hidden_states, context = checkpoint_handler(custom(layer_idx, layer_idx + 1))
                else:
                    hidden_states, context = custom(layer_idx, layer_idx + 1)(
                        hidden_states, attention_mask, context, context_mask, context_signal, rotary_pos_emb
                    )
        else:
            raise ValueError("Invalid activation recompute method.")

        return hidden_states
    
    def forward(
        self,
        hidden_states: Union[Tensor, WrappedTensor],
        attention_mask: Optional[Tensor],
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        context_signal: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        rotary_pos_cos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        dynamic_inference_decode_only: Optional[bool] = None,
    ):
        """
        Perform the forward pass through the transformer block.

        This method handles the core computation of the transformer, including
        self-attention, optional cross-attention, and feed-forward operations.

        Args:
            hidden_states (Union[Tensor, WrappedTensor]): Input tensor of shape [s, b, h]
                where s is the sequence length, b is the batch size, and h is the hidden size.
                Can be passed as a WrappedTensor during inference to avoid an obsolete
                reference in the calling function.
            attention_mask (Tensor): Boolean tensor of shape [1, 1, s, s] for masking
                self-attention.
            context (Tensor, optional): Context tensor for cross-attention.
            context_mask (Tensor, optional): Mask for cross-attention context
            context_signal (Tensor, optional): Signal from context tokens
            rotary_pos_emb (Tensor, optional): Rotary positional embeddings.
            rotary_pos_cos (Optional[Tensor]): Rotary embedding cosine.
            rotary_pos_sin (Optional[Tensor]): Rotary embedding sine.
            rotary_pos_cos_sin (Optional[Tensor]): Combined rotary embedding cosine and sine.
            Currently used exclusively for inference with dynamic batching and flashinfer RoPE.
            attention_bias (Tensor): Bias tensor for Q * K.T of shape in shape broadcastable
                to [b, num_head, sq, skv], e.g. [1, 1, sq, skv].
                Used as an alternative to apply attention mask for TE cuDNN attention.
            inference_context (BaseInferenceContext, optional): Parameters for inference-time
                optimizations.
            packed_seq_params (PackedSeqParams, optional): Parameters for packed sequence
                processing.
            dynamic_inference_decode_only: Optional[bool]: If true, indicates that the current
                inference context is for decode-only. This args is only used to uniquely
                identify decode and non-decode cuda graph runners in the cuda graph manager.

        Returns:
            Union[Tensor, Tuple[Tensor, Tensor]]: The output hidden states tensor of shape
            [s, b, h], and optionally the updated context tensor if cross-attention is used.
        """

        inference_context = deprecate_inference_params(inference_context, inference_params)
        # Remove 'dynamic_inference_decode_only' from kwargs if present
        # this is only used to uniquely identify decode and non-decode cuda graph
        # runners in the cuda graph manager

        # Delete the obsolete reference to the initial input tensor if necessary
        if isinstance(hidden_states, WrappedTensor):
            hidden_states = hidden_states.unwrap()

        if not self.pre_process:
            # See set_input_tensor()
            hidden_states = self.input_tensor

        # Viewless tensor.
        # - We only need to create a viewless tensor in the case of micro batch
        #   size (mbs) == 1, since in this case, 'hidden_states.transpose()'
        #   above creates a view tensor, and '.contiguous()' is a pass-through.
        #   For mbs >= 2, '.contiguous()' creates a new tensor, eliminating
        #   the need to make it viewless.
        #
        #   However, we don't explicitly check mbs == 1 here because
        #   make_viewless_tensor() has negligible overhead when its input
        #   is already viewless.
        #
        # - For the 'else' case above, calling make_viewless_tensor() here is
        #   likely redundant, since p2p_communication.py (likely originator)
        #   already creates viewless tensors. That said, make_viewless_tensor()
        #   is called here to be future-proof and corner-case-proof.
        hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)

        if self.config.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        # If fp8_recipe is delayed, wrap the entire pass with get_fp8_context(),
        # otherwise do nothing extra at the outer level
        # if we are using other fp8 recipes, then the context manager enter&exit are free
        # we can wrap fp8_context within the for loop over layers, so that we can fine-grained
        # control which layer will be fp8 or bf16
        # For FP4: NVFP4BlockScaling doesn't have delayed scaling, always uses inner context
        if self.config.fp8:
            use_outer_quantization_context = self.config.fp8_recipe == Fp8Recipe.delayed
            use_inner_quantization_context = self.config.fp8_recipe != Fp8Recipe.delayed
            outer_quantization_context = (
                get_fp8_context(self.config) if use_outer_quantization_context else nullcontext()
            )
        elif self.config.fp4:
            use_outer_quantization_context = False
            use_inner_quantization_context = True
            outer_quantization_context = nullcontext()
        else:
            # No quantization
            use_outer_quantization_context = False
            use_inner_quantization_context = False
            outer_quantization_context = nullcontext()

        with rng_context, outer_quantization_context:
            # Forward pass.
            if self.config.recompute_granularity == 'full' and self.training:
                hidden_states = self._checkpointed_forward(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    context=context,
                    context_mask=context_mask,
                    context_signal=context_signal,
                    rotary_pos_emb=rotary_pos_emb,
                    attention_bias=attention_bias,
                    packed_seq_params=packed_seq_params,
                    use_inner_quantization_context=use_inner_quantization_context,
                )
            else:
                for l_no, layer in enumerate(self.layers):
                    # Get appropriate inner quantization context
                    if use_inner_quantization_context:
                        if self.config.fp8:
                            inner_quantization_context = get_fp8_context(
                                self.config, layer.layer_number - 1
                            )
                        elif self.config.fp4:
                            inner_quantization_context = get_fp4_context(
                                self.config, layer.layer_number - 1
                            )
                        else:
                            inner_quantization_context = nullcontext()
                    else:
                        inner_quantization_context = nullcontext()

                    with self.offload_context, inner_quantization_context:
                        hidden_states, context = layer(
                            hidden_states=hidden_states,
                            attention_mask=attention_mask,
                            context=context,
                            context_mask=context_mask,
                            context_signal=context_signal,
                            rotary_pos_emb=rotary_pos_emb,
                            rotary_pos_cos=rotary_pos_cos,
                            rotary_pos_sin=rotary_pos_sin,
                            # rotary_pos_cos_sin=rotary_pos_cos_sin,
                            attention_bias=attention_bias,
                            inference_context=inference_context,
                            packed_seq_params=packed_seq_params,
                            sequence_len_offset=sequence_len_offset,
                        )

                    if (
                        torch.is_grad_enabled()
                        and self.config.cpu_offloading
                        and self.group_prefetch_offload_commit_async is not None
                    ):
                        hidden_states = self.group_prefetch_offload_commit_async(hidden_states)

        # Final layer norm.
        if self.final_layernorm is not None:
            hidden_states = self.final_layernorm(hidden_states)
            # TENorm produces a "viewed" tensor. This will result in schedule.py's
            # deallocate_output_tensor() throwing an error, so a viewless tensor is
            # created to prevent this.
            hidden_states = make_viewless_tensor(
                inp=hidden_states, requires_grad=True, keep_graph=True
            )

        # If this TransformerBlock is empty, input and output hidden states will be the same node
        # on the computational graph and will lead to unexpected errors in pipeline schedules.
        if not self.pre_process and len(self.layers) == 0 and not self.final_layernorm:
            hidden_states = hidden_states.clone()

        return hidden_states
            
class ContextTransformerBlock(TransformerBlock):
    def __init__(
        self,
        config: TransformerConfig,
        spec: Union[TransformerBlockSubmodules, ModuleSpec],
        post_layer_norm: bool = True,
        pre_process: bool = True,
        post_process: bool = True,
        pg_collection: ProcessGroupCollection = None,
        vp_stage: Optional[int] = None,
    ):
        # Pass block id and context_scale
        self.vace_id = [i for i in range(0, config.num_layers)] if config.vace_layers is None else [i for i in range(0, len(config.vace_layers))]
        print(self.vace_id)
        assert 0 in self.vace_id
        
        super().__init__(
            config=config, 
            spec=spec,
            post_layer_norm=post_layer_norm,
            pre_process=pre_process,
            post_process=post_process,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
        )
    
    def _build_layers(self):
        # Transformer layers.
        # @jcasper can we improve how we deal with layer_number?
        # currently it's only used in CoreAttention?
        # if self.apply_query_key_layer_scaling:
        #     coeff = self.layer_number
        #     self.norm_factor *= coeff
        def build_layer(layer_spec, layer_number):
            global_layer_number = layer_number + get_transformer_layer_offset(
                self.config, self.vp_stage, get_pg_rank(self.pg_collection.pp)
            )  # 1-based index
            if self.config.heterogeneous_block_specs:
                layer_config = self.config.get_config_for_layer(global_layer_number)
            else:
                layer_config = self.config

            # Get appropriate quantization context (FP8 and FP4 are mutually exclusive)
            if layer_config.fp8:
                quantization_context = get_fp8_context(
                    layer_config, global_layer_number - 1, is_init=True
                )
            elif layer_config.fp4:
                quantization_context = get_fp4_context(
                    layer_config, global_layer_number - 1, is_init=True
                )
            else:
                quantization_context = nullcontext()

            with quantization_context:
                module = build_module(
                    layer_spec,
                    config=layer_config,
                    layer_number=layer_number,
                    pg_collection=self.pg_collection,
                    vp_stage=self.vp_stage,
                )
                idx = global_layer_number - 1
                if idx in self.vace_id:
                    module.idx = idx
                else:
                    module.idx = None
            return module

        # offset is implicit in TransformerLayer
        self.layers = torch.nn.ModuleList(
            [
                build_layer(layer_spec, i + 1)
                for i, layer_spec in enumerate(self.submodules.layer_specs)
            ]
        )

        # @TODO: add back account_for_embedding_in_pipeline_split (see issue #293)
        # In pipeline parallelism, we want to add this LN only to the last stage of the pipeline
        # self.post_process and self.post_layer_norm guide this behavior
        if self.submodules.layer_norm and self.post_process and self.post_layer_norm:
            self.final_layernorm = build_module(
                self.submodules.layer_norm,
                config=self.config,
                hidden_size=self.config.hidden_size,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.final_layernorm = None  # Either this or nn.Identity
            
    def _checkpointed_forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        context: Tensor,
        context_mask: Tensor,
        context_signal: Tensor,
        rotary_pos_emb: Tensor,
        attention_bias: Tensor,
        packed_seq_params: PackedSeqParams,
        use_inner_quantization_context: bool,
    ):
        """Forward method with activation checkpointing."""

        def custom(start: int, end: int):
            def custom_forward(
                hidden_states, attention_mask, context, context_mask, context_signal, rotary_pos_emb
            ):
                for index in range(start, end):
                    layer = self._get_layer(index)

                    # Get appropriate inner quantization context
                    if use_inner_quantization_context:
                        if self.config.fp8:
                            inner_quantization_context = get_fp8_context(
                                self.config, layer.layer_number - 1
                            )
                        # TODO: check if fp4 is supported in this case
                        elif self.config.fp4:
                            inner_quantization_context = get_fp4_context(
                                self.config, layer.layer_number - 1
                            )
                        else:
                            inner_quantization_context = nullcontext()
                    else:
                        inner_quantization_context = nullcontext()

                    with inner_quantization_context:
                        hidden_states, context_signal = layer(
                            hidden_states=hidden_states,
                            attention_mask=attention_mask,
                            context=context,
                            context_mask=context_mask,
                            context_signal=context_signal,
                            rotary_pos_emb=rotary_pos_emb,
                            attention_bias=attention_bias,
                            inference_context=None,
                            packed_seq_params=packed_seq_params,
                        )
                return hidden_states, context_signal

            return custom_forward

        def checkpoint_handler(forward_func):
            """Determines whether to use the `te_checkpoint` or `tensor_parallel.checkpoint`"""
            # TODO: check if fp4 is supported in this case
            if self.config.fp8 or self.config.fp4:
                return te_checkpoint(
                    forward_func,
                    self.config.distribute_saved_activations,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    self.pg_collection.tp,
                    hidden_states,
                    attention_mask,
                    context,
                    context_mask,
                    context_signal,
                    rotary_pos_emb,
                )
            else:
                return tensor_parallel.checkpoint(
                    forward_func,
                    self.config.distribute_saved_activations,
                    hidden_states,
                    attention_mask,
                    context,
                    context_mask,
                    context_signal,
                    rotary_pos_emb,
                )

        if self.config.recompute_method == 'uniform':
            # Uniformly divide the total number of Transformer layers and checkpoint
            # the input activation of each divided chunk.
            # A method to further reduce memory usage reducing checkpoints.
            layer_idx = 0
            while layer_idx < self.num_layers_per_pipeline_rank:
                hidden_states, context_signal = checkpoint_handler(
                    custom(layer_idx, layer_idx + self.config.recompute_num_layers)
                )

                layer_idx += self.config.recompute_num_layers

        elif self.config.recompute_method == 'block':
            # Checkpoint the input activation of only a set number of individual
            # Transformer layers and skip the rest.
            # A method fully use the device memory removing redundant re-computation.
            recompute_skip_num_layers = 0
            for layer_idx in range(self.num_layers_per_pipeline_rank):
                # Skip recomputation when input grad computation is not needed.
                # Need to have at least one input tensor with gradient computation
                # for re-enterant autograd engine.
                # TODO: check if fp4 is supported in this case
                if (self.config.fp8 or self.config.fp4) and not hidden_states.requires_grad:
                    recompute_skip_num_layers += 1
                if (
                    layer_idx >= recompute_skip_num_layers
                    and layer_idx < self.config.recompute_num_layers + recompute_skip_num_layers
                ):
                    hidden_states, context_signal = checkpoint_handler(custom(layer_idx, layer_idx + 1))
                else:
                    hidden_states, context_signal = custom(layer_idx, layer_idx + 1)(
                        hidden_states, attention_mask, context, context_mask, context_signal, rotary_pos_emb
                    )
        else:
            raise ValueError("Invalid activation recompute method.")

        return hidden_states, context_signal
    
    def forward(
        self,
        hidden_states: Union[Tensor, WrappedTensor],
        attention_mask: Optional[Tensor],
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        context_signal: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        rotary_pos_cos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        dynamic_inference_decode_only: Optional[bool] = None,
    ):
        """
        Perform the forward pass through the transformer block.

        This method handles the core computation of the transformer, including
        self-attention, optional cross-attention, and feed-forward operations.

        Args:
            hidden_states (Union[Tensor, WrappedTensor]): Input tensor of shape [s, b, h]
                where s is the sequence length, b is the batch size, and h is the hidden size.
                Can be passed as a WrappedTensor during inference to avoid an obsolete
                reference in the calling function.
            attention_mask (Tensor): Boolean tensor of shape [1, 1, s, s] for masking
                self-attention.
            context (Tensor, optional): Context tensor for cross-attention.
            context_mask (Tensor, optional): Mask for cross-attention context
            context_signal (Tensor, optional): Signal from context tokens
            rotary_pos_emb (Tensor, optional): Rotary positional embeddings.
            rotary_pos_cos (Optional[Tensor]): Rotary embedding cosine.
            rotary_pos_sin (Optional[Tensor]): Rotary embedding sine.
            rotary_pos_cos_sin (Optional[Tensor]): Combined rotary embedding cosine and sine.
            Currently used exclusively for inference with dynamic batching and flashinfer RoPE.
            attention_bias (Tensor): Bias tensor for Q * K.T of shape in shape broadcastable
                to [b, num_head, sq, skv], e.g. [1, 1, sq, skv].
                Used as an alternative to apply attention mask for TE cuDNN attention.
            inference_context (BaseInferenceContext, optional): Parameters for inference-time
                optimizations.
            packed_seq_params (PackedSeqParams, optional): Parameters for packed sequence
                processing.
            dynamic_inference_decode_only: Optional[bool]: If true, indicates that the current
                inference context is for decode-only. This args is only used to uniquely
                identify decode and non-decode cuda graph runners in the cuda graph manager.

        Returns:
            Union[Tensor, Tuple[Tensor, Tensor]]: The output hidden states tensor of shape
            [s, b, h], and optionally the updated context tensor if cross-attention is used.
        """

        inference_context = deprecate_inference_params(inference_context, inference_params)
        # Remove 'dynamic_inference_decode_only' from kwargs if present
        # this is only used to uniquely identify decode and non-decode cuda graph
        # runners in the cuda graph manager

        # Delete the obsolete reference to the initial input tensor if necessary
        if isinstance(hidden_states, WrappedTensor):
            hidden_states = hidden_states.unwrap()

        if not self.pre_process:
            # See set_input_tensor()
            hidden_states = self.input_tensor

        # Viewless tensor.
        # - We only need to create a viewless tensor in the case of micro batch
        #   size (mbs) == 1, since in this case, 'hidden_states.transpose()'
        #   above creates a view tensor, and '.contiguous()' is a pass-through.
        #   For mbs >= 2, '.contiguous()' creates a new tensor, eliminating
        #   the need to make it viewless.
        #
        #   However, we don't explicitly check mbs == 1 here because
        #   make_viewless_tensor() has negligible overhead when its input
        #   is already viewless.
        #
        # - For the 'else' case above, calling make_viewless_tensor() here is
        #   likely redundant, since p2p_communication.py (likely originator)
        #   already creates viewless tensors. That said, make_viewless_tensor()
        #   is called here to be future-proof and corner-case-proof.
        hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)

        if self.config.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        # If fp8_recipe is delayed, wrap the entire pass with get_fp8_context(),
        # otherwise do nothing extra at the outer level
        # if we are using other fp8 recipes, then the context manager enter&exit are free
        # we can wrap fp8_context within the for loop over layers, so that we can fine-grained
        # control which layer will be fp8 or bf16
        # For FP4: NVFP4BlockScaling doesn't have delayed scaling, always uses inner context
        if self.config.fp8:
            use_outer_quantization_context = self.config.fp8_recipe == Fp8Recipe.delayed
            use_inner_quantization_context = self.config.fp8_recipe != Fp8Recipe.delayed
            outer_quantization_context = (
                get_fp8_context(self.config) if use_outer_quantization_context else nullcontext()
            )
        elif self.config.fp4:
            use_outer_quantization_context = False
            use_inner_quantization_context = True
            outer_quantization_context = nullcontext()
        else:
            # No quantization
            use_outer_quantization_context = False
            use_inner_quantization_context = False
            outer_quantization_context = nullcontext()

        with rng_context, outer_quantization_context:
            # Forward pass.
            if self.config.recompute_granularity == 'full' and self.training:
                hidden_states, context_signal = self._checkpointed_forward(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    context=context,
                    context_mask=context_mask,
                    context_signal=context_signal,
                    rotary_pos_emb=rotary_pos_emb,
                    attention_bias=attention_bias,
                    packed_seq_params=packed_seq_params,
                    use_inner_quantization_context=use_inner_quantization_context,
                )
            else:
                for l_no, layer in enumerate(self.layers):
                    # Get appropriate inner quantization context
                    if use_inner_quantization_context:
                        if self.config.fp8:
                            inner_quantization_context = get_fp8_context(
                                self.config, layer.layer_number - 1
                            )
                        elif self.config.fp4:
                            inner_quantization_context = get_fp4_context(
                                self.config, layer.layer_number - 1
                            )
                        else:
                            inner_quantization_context = nullcontext()
                    else:
                        inner_quantization_context = nullcontext()

                    with self.offload_context, inner_quantization_context:
                        hidden_states, context_signal = layer(
                            hidden_states=hidden_states,
                            attention_mask=attention_mask,
                            context=context,
                            context_mask=context_mask,
                            context_signal=context_signal,
                            rotary_pos_emb=rotary_pos_emb,
                            rotary_pos_cos=rotary_pos_cos,
                            rotary_pos_sin=rotary_pos_sin,
                            # rotary_pos_cos_sin=rotary_pos_cos_sin,
                            attention_bias=attention_bias,
                            inference_context=inference_context,
                            packed_seq_params=packed_seq_params,
                            sequence_len_offset=sequence_len_offset,
                        )

                    if (
                        torch.is_grad_enabled()
                        and self.config.cpu_offloading
                        and self.group_prefetch_offload_commit_async is not None
                    ):
                        hidden_states = self.group_prefetch_offload_commit_async(hidden_states)

        # Final layer norm.
        if self.final_layernorm is not None:
            hidden_states = self.final_layernorm(hidden_states)
            # TENorm produces a "viewed" tensor. This will result in schedule.py's
            # deallocate_output_tensor() throwing an error, so a viewless tensor is
            # created to prevent this.
            hidden_states = make_viewless_tensor(
                inp=hidden_states, requires_grad=True, keep_graph=True
            )

        # If this TransformerBlock is empty, input and output hidden states will be the same node
        # on the computational graph and will lead to unexpected errors in pipeline schedules.
        if not self.pre_process and len(self.layers) == 0 and not self.final_layernorm:
            hidden_states = hidden_states.clone()

        return hidden_states, context_signal

def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x

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

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
        x = (self.head(self.norm(x) * (1 + e[1]) + e[0]))
        return x


class WanModel(VisionModule):
    """
    WanModel is a VisionModule that implements a Wan model.
    Attributes:
        config (TransformerConfig): Configuration for the transformer.
        pre_process (bool): Whether to apply pre-processing steps.
        post_process (bool): Whether to apply post-processing steps.
        fp16_lm_cross_entropy (bool): Whether to use fp16 for cross-entropy loss.
        parallel_output (bool): Whether to use parallel output.
        transformer_decoder_layer_spec (WanLayerWithAdaLNspec): Specification for the transformer decoder layer.
        model_type (ModelType): Type of the model.
    """

    def __init__(
        self,
        config: TransformerConfig,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        transformer_decoder_layer_spec=WanLayerWithAdaLNspec,
        **kwargs,
    ):
        super(WanModel, self).__init__(config=config)

        self.config: TransformerConfig = config

        self.transformer_decoder_layer_spec = transformer_decoder_layer_spec()
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy
        self.parallel_output = parallel_output

        # megatron core pipelining currently depends on model type
        # TODO: remove this dependency ?
        self.model_type = ModelType.encoder_or_decoder

        self.num_heads = self.config.num_attention_heads
        self.freq_dim = self.config.freq_dim
        self.in_channels = self.config.in_channels
        self.out_channels = self.config.out_channels
        self.patch_spatial = self.config.patch_spatial
        self.patch_temporal = self.config.patch_temporal
        self.patch_size = (self.patch_temporal, self.patch_spatial, self.patch_spatial)

        # these attributes are unused for images/videos, we just set because bridge training requires for LLMs
        self.share_embeddings_and_output_weights = False

        ######################################
        ########## Wan architecture ##########

        # embeddings
        if self.pre_process:
            self.patch_embedding = nn.Conv3d(
                self.in_channels, self.config.hidden_size, kernel_size=self.patch_size, stride=self.patch_size)

        self.text_embedding = nn.Sequential(
            nn.Linear(self.config.text_dim, self.config.hidden_size), nn.GELU(approximate='tanh'),
            nn.Linear(self.config.hidden_size, self.config.hidden_size))

        self.time_embedding = nn.Sequential(
            nn.Linear(self.freq_dim, self.config.hidden_size), nn.SiLU(), nn.Linear(self.config.hidden_size, self.config.hidden_size))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(self.config.hidden_size, self.config.hidden_size * 6))

        self.rope_embeddings = Wan3DRopeEmbeddings(dim_head = self.config.hidden_size // self.num_heads, max_position_len = 1024)

        # decoder blocks
        self.decoder = TransformerBlock(
            config=self.config,
            spec=self.transformer_decoder_layer_spec,
            pre_process=self.pre_process,
            post_process=self.post_process,
            post_layer_norm=False,
        )

        # output head
        if self.post_process:
            self.head = Head(self.config.hidden_size, self.out_channels, self.patch_size, eps = 1e-6)


    def forward(
        self,
        x: Tensor,
        grid_sizes: list[Tuple[int, int, int]],
        t: Tensor,
        context: Tensor,
        max_seq_len: int,
        packed_seq_params: PackedSeqParams = None,
        **kwargs,
    ) -> Tensor:
        """Forward pass.

        Args:
            x List[Tensor]: list of vae encoded data (s, b, c * pF * pH * pW)
            grid_sizes List[Tuple[int, int, int]]: list of grid sizes (f, h, w)
            t Tensor: timesteps
            context List[Tensor]: list of context (text_len, hidden_size)
            max_seq_len int: maximum sequence length
            packed_seq_params PackedSeqParams: packed sequence parameters

        Returns:
            Tensor: output tensor (still patchified) of shape [seq_len, batch_size, hidden_size]
        """
        #################################
        ########## Wan forward ##########

        # ============= embedders =============

        # run input embedding
        if self.pre_process:
            # x.shape [s, b, c * pF * pH * pW]
            seq_len, batch_size, _ = x.shape
            c = self.in_channels
            pF, pH, pW = self.patch_size
            x = x.reshape(seq_len * batch_size, pF, pH, pW, c) # output: x.shape [s * b, pF, pH, pW, c]
            x = x.permute(0, 4, 1, 2, 3) # output: x.shape [s * b, c, pF, pH, pW]
            x = self.patch_embedding(x) # output: x.shape [s * b, hidden_size, 1, 1, 1]
            x = x.flatten(1) # output: x.shape [s * b, hidden_size]
            x = x.reshape(seq_len, batch_size, -1) # output: x.shape [s, b, hidden_size]

            # split sequence for sequence_parallel
            # TODO: for PP, do we move scatter_to_sequence_parallel_region here or after "x = self.decoder.input_tensor" ???
            if self.config.sequence_parallel:
                x = tensor_parallel.scatter_to_sequence_parallel_region(x) # output: x.shape [s * b // tp_size, hidden_size]

        else:
            # intermediate stage of pipeline
            x = self.decoder.input_tensor

        # time embeddings
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).to(x.dtype)
        )
        e0 = self.time_projection(e).unflatten(1, (6, self.config.hidden_size))

        # context embeddings
        context = self.text_embedding(context) # shape [text_len, b, hidden_size]


        # ============= decoder =============
        # calculate rotary pos emb
        n_head, dim_head = self.num_heads, self.config.hidden_size // self.num_heads
        rotary_pos_emb = self.rope_embeddings(n_head, dim_head, max_seq_len, grid_sizes, t.device) # output: rotary_pos_emb.shape [s, b, 1, dim_head]

        # run decoder
        x = self.decoder(
            hidden_states=x,
            attention_mask=e0,
            context=context,
            context_mask=None,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=None,
            rotary_pos_sin=None,
            packed_seq_params=packed_seq_params,
        )

        # return if not post_process
        if not self.post_process:
            return x

        # head
        x = x.transpose(0, 1) # head expects shape [b, s, hidden_size]
        x = self.head(x, e) # output: x.shape [b, s, c * pF * pH * pW]
        x = x.transpose(0, 1) # reshape back to shape [s, b, c * pF * pH * pW]

        # gather outputs for sequence_parallel
        # Note: in GPT models, because the vocab projection matrix is ColumnParallelLinear, the sequence is 
        #   automatically gathered in ColumnParallelLinear forward pass.
        #   However, in Wan models, we need to gather the outputs manually.
        if self.config.sequence_parallel:
            x = tensor_parallel.gather_from_sequence_parallel_region(x)

        return x # output: x.shape [s, b, c * pF * pH * pW]


    def set_input_tensor(self, input_tensor: Tensor) -> None:
        """Sets input tensor to the model.

        See megatron.model.transformer.set_input_tensor()

        Args:
            input_tensor (Tensor): Sets the input tensor for the model.
        """
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]

        assert len(input_tensor) == 1, "input_tensor should only be length 1 for gpt/bert"
        self.decoder.set_input_tensor(input_tensor[0])


    def sharded_state_dict(
        self, prefix: str = "", sharded_offsets: tuple = (), metadata: Optional[Dict] = None
    ) -> ShardedStateDict:
        """Sharded state dict implementation for GPTModel backward-compatibility (removing extra state).

        Args:
            prefix (str): Module name prefix.
            sharded_offsets (tuple): PP related offsets, expected to be empty at this module level.
            metadata (Optional[Dict]): metadata controlling sharded state dict creation.

        Returns:
            ShardedStateDict: sharded state dict for the GPTModel
        """
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)

        # DEBUGGING
        # for module in ["t_embedder"]:
        #     for param_name, param in getattr(self, module).named_parameters():
        #         weight_key = f"{prefix}{module}.{param_name}"
        #         self._set_embedder_weights_replica_id(param, sharded_state_dict, weight_key)
        # DEBUGGING
        # Ensure replica ids for non-transformer embedder weights include pipeline dimension
        for module in ["text_embedding", "time_embedding", "time_projection"]:
            if hasattr(self, module):
                for param_name, param in getattr(self, module).named_parameters():
                    weight_key = f"{prefix}{module}.{param_name}"
                    if weight_key in sharded_state_dict:
                        self._set_embedder_weights_replica_id(param, sharded_state_dict, weight_key)

        return sharded_state_dict


    def _set_embedder_weights_replica_id(
        self, tensor: Tensor, sharded_state_dict: ShardedStateDict, embedder_weight_key: str
    ) -> None:
        """set replica ids of the weights in t_embedder for sharded state dict.

        Args:
            sharded_state_dict (ShardedStateDict): state dict with the weight to tie
            weight_key (str): key of the weight in the state dict.
                This entry will be replaced with a tied version

        Returns: None, acts in-place
        """
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        vpp_rank = parallel_state.get_virtual_pipeline_model_parallel_rank()
        vpp_rank = vpp_rank if vpp_rank else 0
        vpp_world = parallel_state.get_virtual_pipeline_model_parallel_world_size()
        vpp_world = vpp_world if vpp_world else 1
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        del sharded_state_dict[embedder_weight_key]
        replica_id = (
            tp_rank,
            (vpp_rank + pp_rank * vpp_world),
            parallel_state.get_data_parallel_rank(with_context_parallel=True),
        )

        sharded_state_dict[embedder_weight_key] = make_sharded_tensor_for_checkpoint(
            tensor=tensor,
            key=embedder_weight_key,
            replica_id=replica_id,
            allow_shape_mismatch=False,
        )


class VACEModel(WanModel):
    def __init__(
        self,
        config: TransformerConfig,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        transformer_decoder_layer_spec=VACEBaseLayerspec,
        vace_transformer_decoder_layer_spec=VACEContextLayerspec,
        **kwargs,
    ):
        super().__init__(
            config,
            pre_process,
            post_process,
            fp16_lm_cross_entropy,
            parallel_output,
            transformer_decoder_layer_spec,
            **kwargs
        )
        
        self.vace_in_channels = self.config.vace_in_channels
        self.vace_transformer_decoder_layer_spec = vace_transformer_decoder_layer_spec()
        
        if self.pre_process:
            self.vace_patch_embedding = nn.Conv3d(
                self.vace_in_channels, self.config.hidden_size, kernel_size=self.patch_size, stride=self.patch_size)
            
        self.decoder = BaseTransformerBlock(
            config=self.config,
            spec=self.transformer_decoder_layer_spec,
            pre_process=self.pre_process,
            post_process=self.post_process,
            post_layer_norm=False,
        )
        # print(self.decoder)
        self.vace_config = copy.deepcopy(self.config)
        self.vace_config.num_layers = len(self.decoder.vace_layers)
        self.vace_decoder = ContextTransformerBlock(
            config=self.vace_config,
            spec=self.vace_transformer_decoder_layer_spec,
            pre_process=self.pre_process,
            post_process=self.post_process,
            post_layer_norm=False,
        )
        # print(self.vace_decoder.state_dict().keys())
        
        self.vace_init_proj = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        
        # Freeze base WAN parameters if specified
        if getattr(self.config, 'freeze_base_model', False):
            self.freeze_base_parameters()
    
    def freeze_base_parameters(self):
        """
        Freeze all base WAN model parameters, only allow VACE-specific parameters to be trained.
        
        Frozen parameters (from base WAN model):
        - patch_embedding
        - text_embedding
        - time_embedding
        - time_projection
        - rope_embeddings
        - decoder.layers (base transformer layers, not VACE layers)
        - head
        
        Trainable parameters (VACE-specific):
        - vace_patch_embedding
        - vace_decoder (separate transformer for VACE context)
        - vace_init_proj
        - decoder.vace_layers (VACE context attention layers within decoder)
        """
        # Freeze base model embeddings
        for param in self.patch_embedding.parameters():
            param.requires_grad = False
        for param in self.text_embedding.parameters():
            param.requires_grad = False
        for param in self.time_embedding.parameters():
            param.requires_grad = False
        for param in self.time_projection.parameters():
            param.requires_grad = False
        for param in self.rope_embeddings.parameters():
            param.requires_grad = False
        
        # Freeze output head
        for param in self.head.parameters():
            param.requires_grad = False
        
        # Freeze base decoder layers (but not vace_layers)
        if hasattr(self.decoder, 'layers'):
            for layer in self.decoder.layers:
                for param in layer.parameters():
                    param.requires_grad = False
        
        print("[VACEModel] Frozen base WAN model parameters. Only VACE-specific parameters will be trained:")
        print(f"  - vace_patch_embedding")
        print(f"  - vace_decoder ({self.vace_config.num_layers} layers)")
        print(f"  - vace_init_proj")
        if hasattr(self.decoder, 'vace_layers'):
            print(f"  - decoder.vace_layers ({len(self.decoder.vace_layers)} VACE context layers)")
    
    def forward(
        self,
        x: Tensor,
        grid_sizes: list[Tuple[int, int, int]],
        t: Tensor,
        context: Tensor,
        vace_context: Tensor,
        max_seq_len: int,
        packed_seq_params: PackedSeqParams = None,
        **kwargs,
    ) -> Tensor:
        """Forward pass.

        Args:
            x List[Tensor]: list of vae encoded data (s, b, c * pF * pH * pW)
            grid_sizes List[Tuple[int, int, int]]: list of grid sizes (f, h, w)
            t Tensor: timesteps
            context List[Tensor]: list of context (text_len, hidden_size)
            max_seq_len int: maximum sequence length
            packed_seq_params PackedSeqParams: packed sequence parameters

        Returns:
            Tensor: output tensor (still patchified) of shape [seq_len, batch_size, hidden_size]
        """
        #################################
        ########## Wan forward ##########

        # ============= embedders =============

        # run input embedding
        if self.pre_process:
            # x.shape [s, b, c * pF * pH * pW]
            seq_len, batch_size, _ = x.shape
            c = self.in_channels
            pF, pH, pW = self.patch_size
            x = x.reshape(seq_len * batch_size, pF, pH, pW, c) # output: x.shape [s * b, pF, pH, pW, c]
            x = x.permute(0, 4, 1, 2, 3) # output: x.shape [s * b, c, pF, pH, pW]
            x = self.patch_embedding(x) # output: x.shape [s * b, hidden_size, 1, 1, 1]
            x = x.flatten(1) # output: x.shape [s * b, hidden_size]
            x = x.reshape(seq_len, batch_size, -1) # output: x.shape [s, b, hidden_size]
            
            # vace_context.shape [s, b, c * pF * pH * pW]
            vace_seq_len, _, vace_flat_dim = vace_context.shape
            # Calculate actual channels from the tensor shape
            vace_c = vace_flat_dim // (pF * pH * pW)
            # pF, pH, pW = self.patch_size
            vace_context = vace_context.reshape(vace_seq_len * batch_size, pF, pH, pW, vace_c) # output: vace_context.shape [s * b, pF, pH, pW, c]
            vace_context = vace_context.permute(0, 4, 1, 2, 3) # output: vace_context.shape [s * b, c, pF, pH, pW]
            # Use patch_embedding if vace_context has same channels as main input (self-editing mode)
            # Otherwise use vace_patch_embedding for different channel counts
            if vace_c == self.in_channels:
                vace_context = self.patch_embedding(vace_context) # output: vace_context.shape [s * b, hidden_size, 1, 1, 1]
            else:
                vace_context = self.vace_patch_embedding(vace_context) # output: vace_context.shape [s * b, hidden_size, 1, 1, 1]
            vace_context = vace_context.flatten(1) # output: vace_context.shape [s * b, hidden_size]
            vace_context = vace_context.reshape(vace_seq_len, batch_size, -1) # output: vace_context.shape [s, b, hidden_size]
            vace_context = self.vace_init_proj(vace_context) + x
            # vace_context = vace_context.unsqueeze(0)
            vace_context = torch.stack([vace_context] * (self.vace_config.num_layers))
            
            # split sequence for sequence_parallel
            # TODO: for PP, do we move scatter_to_sequence_parallel_region here or after "x = self.decoder.input_tensor" ???
            if self.config.sequence_parallel:
                x = tensor_parallel.scatter_to_sequence_parallel_region(x) # output: x.shape [s * b // tp_size, hidden_size]
                vace_context = tensor_parallel.scatter_to_sequence_parallel_region(vace_context) # output: vace_context.shape [s * b // tp_size, hidden_size]

        else:
            # intermediate stage of pipeline
            x = self.decoder.input_tensor
            vace_context = self.vace_decoder.input_tensor
            
        # run context token embedding
        
        # time embeddings
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).to(x.dtype)
        )
        e0 = self.time_projection(e).unflatten(1, (6, self.config.hidden_size))

        # context embeddings
        context = self.text_embedding(context) # shape [text_len, b, hidden_size]


        # ============= decoder =============
        # calculate rotary pos emb
        n_head, dim_head = self.num_heads, self.config.hidden_size // self.num_heads
        rotary_pos_emb = self.rope_embeddings(n_head, dim_head, max_seq_len, grid_sizes, t.device) # output: rotary_pos_emb.shape [s, b, 1, dim_head]

        s, b, sq, h = rotary_pos_emb.shape
        rotary_pos_emb = rotary_pos_emb.transpose(0, 1).reshape(s*b, 1, sq, h)
        
        # run vace decoder
        vace_context = self.vace_decoder(
            hidden_states=vace_context[0],
            attention_mask=e0,
            context=context,
            context_mask=None,
            context_signal=vace_context,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=None,
            rotary_pos_sin=None,
            packed_seq_params=packed_seq_params,
        )[1]
        
        # run decoder
        x = self.decoder(
            hidden_states=x,
            attention_mask=e0,
            context=context,
            context_mask=None,
            context_signal=vace_context,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=None,
            rotary_pos_sin=None,
            packed_seq_params=packed_seq_params,
        )

        # return if not post_process
        if not self.post_process:
            return x

        # head
        x = x.transpose(0, 1) # head expects shape [b, s, hidden_size]
        x = self.head(x, e) # output: x.shape [b, s, c * pF * pH * pW]
        x = x.transpose(0, 1) # reshape back to shape [s, b, c * pF * pH * pW]

        # gather outputs for sequence_parallel
        # Note: in GPT models, because the vocab projection matrix is ColumnParallelLinear, the sequence is 
        #   automatically gathered in ColumnParallelLinear forward pass.
        #   However, in Wan models, we need to gather the outputs manually.
        if self.config.sequence_parallel:
            x = tensor_parallel.gather_from_sequence_parallel_region(x)

        return x # output: x.shape [s, b, c * pF * pH * pW]