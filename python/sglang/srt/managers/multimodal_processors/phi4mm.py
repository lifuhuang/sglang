from typing import List, Union

import torch
from transformers.image_utils import SizeDict

from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models.mllama import MllamaForConditionalGeneration
from sglang.srt.utils import load_image

from sglang.srt.managers.multimodal_processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models.phi4mmvllm import Phi4MMForCausalLM


class Phi4MMImageProcessor(BaseMultimodalProcessor):
    models = [Phi4MMForCausalLM]


    def __init__(self, hf_config, server_args, _processor):
        super().__init__(hf_config, server_args, _processor)
        
        self.attention_dropout = 0.0
        self.crop_size = 448
        self.feature_layer = -2
        self.hidden_act = "gelu_pytorch_tanh"
        self.hidden_size = 1152
        self.image_size = 448
        self.image_token_id = 200010
        self.intermediate_size = 4304
        self.layer_norm_eps = 1e-06
        self.model_type = "phi4_multimodal_vision"
        self.num_attention_heads = 16
        self.num_channels = 3
        self.num_hidden_layers = 27
        self.patch_size = 14


    async def process_mm_data_async(
        self, image_data: List[Union[str, bytes]], input_text, *args, **kwargs
    ):
        if not image_data:
            return None

        if isinstance(input_text, list):
            assert len(input_text) and isinstance(input_text[0], int)
            input_text = self._processor.tokenizer.decode(input_text)

        if not isinstance(image_data, list):
            image_data = [image_data]

        if len(image_data) > 0:
            images = [load_image(image)[0] for image in image_data]
        else:
            images = load_image(image_data[0])[0]

        image_inputs = self.process_mm_data(input_text=input_text, images=images)
        image_inputs["input_ids"] = image_inputs["input_ids"].tolist()[0]
        image_inputs["mm_items"] = [
            MultimodalDataItem(
                pixel_values=image_inputs["pixel_values"],
                aspect_ratio_id=image_inputs["aspect_ratio_ids"],
                aspect_ratio_mask=image_inputs["aspect_ratio_mask"],
                modality=Modality.IMAGE,
            )
        ]

        return image_inputs