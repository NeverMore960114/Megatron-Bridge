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

from pathlib import Path
from typing import Optional, Union

from diffusers import WanTransformer3DModel
from transformers import AutoConfig

from megatron.bridge.models.hf_pretrained.base import PreTrainedBase


class PreTrainedWAN(PreTrainedBase):
    """
    Lightweight pretrained wrapper for Diffusers WAN models.

    Provides access to WAN config and state through the common PreTrainedBase API
    so bridges can consume `.config` and `.state` uniformly.
    """

    def __init__(self, model_name_or_path: Union[str, Path], **kwargs):
        self._model_name_or_path = str(model_name_or_path)
        super().__init__(**kwargs)

    @property
    def model_name_or_path(self) -> str:
        return self._model_name_or_path

    # Model loading is optional for conversion; implemented for completeness
    def _load_model(self) -> WanTransformer3DModel:
        return WanTransformer3DModel.from_pretrained(self.model_name_or_path)

    # Config is required by the WAN bridge
    def _load_config(self) -> AutoConfig:
        # WanTransformer3DModel returns a config-like object with required fields

        print(f"Loading config from {self.model_name_or_path}")
        
        return WanTransformer3DModel.from_pretrained(self.model_name_or_path, subfolder="transformer").config


