from typing import List, Optional, Union, Mapping
import math
import torch
from transformers import BaseImageProcessorFast
from transformers.image_utils import SizeDict
from transformers import (BatchFeature, PretrainedConfig, ProcessorMixin,
                          SequenceFeatureExtractor, SiglipVisionConfig)


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
        self.multimodal_tokens = MultimodalSpecialTokens(
            image_token=r'(?:<\|image_\d+\|>)',
            audio_token=r'(?:<\|audio_\d+\|>)',
            is_regex=True
        )

    def process_data_task(self, input_text, images=None, audios=None):

        if isinstance(images, list) and len(images) == 0:
            images = None
        if isinstance(audios, list) and len(audios) == 0:
            audios = None
        processor = self._processor
        args = {}
        if isinstance(processor, BaseImageProcessorFast):
            args["device"] = "cuda"
        result = self._processor.__call__(
            text=input_text,
            images=images,
            audios=audios,
            return_tensors="pt",
            chunk_input=True,
            **args,
        )
        return {
            "input_ids": result.input_ids,
            "pixel_values": getattr(result, "pixel_values", None),
            "tgt_sizes": getattr(result, "tgt_sizes", None),
            "audio_features": getattr(result, "audio_features", None),
            "audio_feature_lens": getattr(result, "audio_feature_lens", None),
            "audio_bounds": getattr(result, "audio_bounds", None),
        }

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
            multimodal_tokens=self.multimodal_tokens
        )
        if base_output is None:
            return None

        res = self.process_mm_data(
            input_text=base_output.input_text,
            images=base_output.images,
            audios=base_output.audios,
        )

        # Collect special token ids
        pixel_values = res["pixel_values"]
        tgt_sizes = res["tgt_sizes"]

        if not isinstance(pixel_values, (torch.Tensor, list)):
            raise ValueError(
                "Incorrect type of pixel values. " f"Got type: {type(pixel_values)}"
            )

        if not isinstance(tgt_sizes, (torch.Tensor, list)):
            raise ValueError(
                "Incorrect type of target sizes. " f"Got type: {type(tgt_sizes)}"
            )

        if len(pixel_values) != len(tgt_sizes):
            raise ValueError(
                "Inconsistent batch lengths, found: "
                f"{len(pixel_values)} vs. {len(tgt_sizes)}"
            )

        pixel_values_flat: List[torch.Tensor] = []
        tgt_sizes_flat: List[torch.Tensor] = []
        for pixel_b, tgt_b in zip(pixel_values, tgt_sizes):
            # per image
            if len(pixel_b) != len(tgt_b):
                raise ValueError(
                    "Inconsistent N lengths, found: " f"{len(pixel_b)} vs {len(tgt_b)}"
                )
            for pixel_n, tgt_n in zip(pixel_b, tgt_b):
                pixel_values_flat += [pixel_n]
                tgt_sizes_flat += [tgt_n]

        pixel_values = pixel_values_flat

        items = []
        if len(pixel_values) != 0:
            item = MultimodalDataItem(
                pixel_values=pixel_values,
                tgt_size=tgt_sizes_flat,
                modality=Modality.IMAGE,
            )
            items += [item]

        return {
            "mm_items": items,
            "input_ids": res["input_ids"].flatten().tolist(),
        }
