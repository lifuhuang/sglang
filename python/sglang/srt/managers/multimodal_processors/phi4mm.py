from typing import List, Optional, Union, Enum
import torch
import types
from transformers import (
    BatchFeature,
)

from transformers.image_processing_utils import BatchFeature
from transformers.image_utils import (
    ImageInput,
)
from transformers.tokenization_utils_base import PaddingStrategy, TextInput, TruncationStrategy
from transformers.utils import TensorType


from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem

from sglang.srt.managers.multimodal_processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models.phi4mmvllm import Phi4MMForCausalLM

_IMAGE_SPECIAL_TOKEN = "<|endoftext10|>"
_IMAGE_SPECIAL_TOKEN_ID = 200010
# _AUDIO_SPECIAL_TOKEN = "<|endoftext11|>"
# _AUDIO_SPECIAL_TOKEN_ID = 200011

class InputMode(Enum):
    LANGUAGE = 0
    VISION = 1
    SPEECH = 2
    VISION_SPEECH = 3


# TODO (lifuhuang): the native Phi4MMProcessor provided by Microsoft does not export num_img_tokens, 
# which is needed for handling multiple images. I applied a patch to the original processor to export 
# this needed value. vLLM handles this by re-calculate # tokens at inferencing time, which does not 
# appear to be efficient, but I haven't got a chance to benchmark. In the future, we should consider 
# to add this patch to the original processor to avoid the hack.
def call_wrapper(
    self,
    text: Union[TextInput, List[TextInput]],
    images: Optional[ImageInput] = None,
    audios: Optional[List] = None,
    padding: Union[bool, str, PaddingStrategy] = False,
    truncation: Optional[Union[bool, str, TruncationStrategy]] = None,
    max_length=None,
    return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
) -> BatchFeature:
    
    image_inputs = self.image_processor(images, return_tensors=return_tensors) if images is not None else {}
    audio_inputs = self.audio_processor(audios, return_tensors=return_tensors) if audios is not None else {}
    inputs = self._convert_images_audios_text_to_inputs(
        image_inputs,
        audio_inputs,
        text,
        padding=padding,
        truncation=truncation,
        max_length=max_length,
        return_tensors=return_tensors,
    )

    # idenfity the input mode
    if len(image_inputs) > 0 and len(audio_inputs) > 0:
        input_mode = InputMode.VISION_SPEECH
    elif len(image_inputs) > 0:
        input_mode = InputMode.VISION
    elif len(audio_inputs) > 0:
        input_mode = InputMode.SPEECH
    else:
        input_mode = InputMode.LANGUAGE
    inputs["input_mode"] = torch.tensor([input_mode.value], dtype=torch.long)
    inputs["num_img_tokens"] = image_inputs.get("num_img_tokens", None)

    return inputs


class Phi4MMImageProcessor(BaseMultimodalProcessor):
    models = [Phi4MMForCausalLM]

    def __init__(self, hf_config, server_args, _processor):
        # TODO (lifuhuang): hack to export num_img_tokens info
        _processor.__call__ = types.MethodType(call_wrapper, _processor)
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

