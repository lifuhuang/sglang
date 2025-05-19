from typing import List, Optional, Union, Mapping
import math
import torch
import re
from transformers import BaseImageProcessorFast
from transformers.image_utils import SizeDict
from transformers import (
    BatchFeature,
    PretrainedConfig,
    ProcessorMixin,
    SequenceFeatureExtractor,
    SiglipVisionConfig,
)


from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models.mllama import MllamaForConditionalGeneration
from sglang.srt.utils import load_image

from sglang.srt.managers.multimodal_processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models.phi4mmvllm import Phi4MMForCausalLM

_IMAGE_SPECIAL_TOKEN = "<|endoftext10|>"
_AUDIO_SPECIAL_TOKEN = "<|endoftext11|>"
_IMAGE_SPECIAL_TOKEN_ID = 200010
_AUDIO_SPECIAL_TOKEN_ID = 200011


class Phi4MMImageProcessor(BaseMultimodalProcessor):
    models = [Phi4MMForCausalLM]

    def __init__(self, hf_config, server_args, _processor):
        super().__init__(hf_config, server_args, _processor)
        self.multimodal_tokens = MultimodalSpecialTokens(
            image_token=_IMAGE_SPECIAL_TOKEN,
        )

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        input_text,
        request_obj,
        max_req_input_len,
        **kwargs,
    ):
        audio_data = request_obj.audio_data
        if not image_data and not audio_data:
            return None
        if not isinstance(image_data, list):
            image_data = [image_data]
        if not isinstance(audio_data, list):
            audio_data = [audio_data]

        base_output = self.load_mm_data(
            prompt=input_text,
            max_req_input_len=max_req_input_len,
            audio_data=audio_data,
            image_data=image_data,
            multimodal_tokens=self.multimodal_tokens,
        )
        if base_output is None:
            return None

        res = self.process_mm_data(
            input_text=base_output.input_text,
            images=base_output.images,
            audios=base_output.audios,
        )

        pixel_values = torch.split(res["input_image_embeds"], 1)
        image_sizes = torch.split(res["image_sizes"], 1)
        image_attention_mask = torch.split(res["image_attention_mask"], 1)

        items = []
        for i in range(len(base_output.images)):
            item = MultimodalDataItem(
                pixel_values=pixel_values[i],
                image_sizes=image_sizes[i],
                image_emb_mask=image_attention_mask[i],
                modality=Modality.IMAGE,
            )
            items += [item]

        return {
            "mm_items": items,
            "input_ids": res["input_ids"].flatten().tolist(),
            "im_token_id": _IMAGE_SPECIAL_TOKEN_ID,
        }
