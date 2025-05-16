from typing import List, Optional, Union, Mapping
import math
import torch
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

        processed_outputs = self.process_mm_data(input_text=input_text, images=images)
        num_img_tokens = [
            self.get_num_image_tokens(image_width=img_size[0],
                                           image_height=img_size[1])
            for img_size in processed_outputs["image_sizes"]
        ]
        processed_outputs["num_img_tokens"] = num_img_tokens

        return processed_outputs
    
    @property
    def dynamic_hd(self):
        return self._processor.image_processor.dynamic_hd

    def get_feature_extractor(self) -> SequenceFeatureExtractor:
        return self.get_hf_processor().audio_processor

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"audio": None, "image": None}

    def _find_target_aspect_ratio(
        self,
        orig_width: int,
        orig_height: int,
        image_size: int,
        max_num: int,
        min_num: int,
    ):
        w_crop_num = math.ceil(orig_width / float(image_size))
        h_crop_num = math.ceil(orig_height / float(image_size))
        if w_crop_num * h_crop_num > max_num:
            aspect_ratio = orig_width / orig_height

            # calculate the existing image aspect ratio
            target_ratios = set((i, j) for i in range(1, max_num + 1)
                                for j in range(1, max_num + 1)
                                if i * j <= max_num and i * j >= min_num)
            target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

            # find the closest aspect ratio to the target
            image_processor = self.get_hf_processor().image_processor
            target_aspect_ratio = image_processor.find_closest_aspect_ratio(
                aspect_ratio,
                target_ratios,
                orig_width,
                orig_height,
                image_size,
            )

            # calculate the target width and height
            target_width = image_size * target_aspect_ratio[0]
            target_height = image_size * target_aspect_ratio[1]
        else:
            target_width = image_size * w_crop_num
            target_height = image_size * h_crop_num
            target_aspect_ratio = (w_crop_num, h_crop_num)
        return target_aspect_ratio, target_height, target_width

    def _compute_num_image_tokens(
        self,
        orig_width: int,
        orig_height: int,
        dynamic_hd_size: int,
        vit_image_size: int,
        vit_patch_size: int,
        token_compression_factor: int = 2,
    ):
        """
        compute the number of tokens an image is expected to take up considering
        the image encoder architecture and exclude output features containing 
        only padding pixels

        for siglip, vit_image_size=448, vit_patch_size=14, so output will be 
        32x32 feature map
        NOTE right now, Phi4MM uses hard-coded token_compression_factor=2
        """
        assert vit_image_size % vit_patch_size == 0, (
            "vit_image_size must be divisible by vit_patch_size")
        assert (vit_image_size // vit_patch_size %
                token_compression_factor == 0), (
                    "vit_image_size // vit_patch_size must be divisible by "
                    "token_compression_factor")

        target_aspect_ratio, target_height, target_width = (
            self._find_target_aspect_ratio(orig_width,
                                           orig_height,
                                           vit_image_size,
                                           dynamic_hd_size,
                                           min_num=1))
        assert target_aspect_ratio[0] * vit_image_size == target_width, (
            f"{target_aspect_ratio[0]} * {vit_image_size} != {target_width}")
        assert target_aspect_ratio[1] * vit_image_size == target_height, (
            f"{target_aspect_ratio[1]} * {vit_image_size} != {target_height}")
        assert (target_height % vit_image_size == 0
                and target_width % vit_image_size == 0)

        padding_height, padding_width = _get_padding_size(
            orig_width, orig_height, target_height, target_width)
        assert padding_width == 0 or padding_height == 0, \
            "padding_width or padding_height must be 0"

        target_feat_width = target_width // vit_patch_size
        target_feat_height = target_height // vit_patch_size
        if padding_width >= vit_patch_size:
            assert padding_height == 0, "padding_height not 0"
            non_pad_feat_width = target_feat_width - math.floor(
                padding_width / vit_patch_size)
            non_pad_feat_height = target_feat_height
        elif padding_height >= vit_patch_size:
            assert padding_width == 0, "padding_width not 0"
            non_pad_feat_height = target_feat_height - math.floor(
                padding_height / vit_patch_size)
            non_pad_feat_width = target_feat_width
        else:
            # small padding shorter than a vit patch
            non_pad_feat_width = target_feat_width
            non_pad_feat_height = target_feat_height

        feat_width = non_pad_feat_width // token_compression_factor
        feat_height = non_pad_feat_height // token_compression_factor
        # NOTE it's possible that the non-padding feature is not divisible
        if non_pad_feat_width % token_compression_factor != 0:
            feat_width += 1
        if non_pad_feat_height % token_compression_factor != 0:
            feat_height += 1
        num_hd_patch_tokens = feat_width * feat_height
        num_hd_newline_tokens = feat_height
        vit_feature_size = vit_image_size // vit_patch_size
        num_global_image_tokens = (vit_feature_size //
                                   token_compression_factor)**2
        num_sep_tokens = 1
        num_global_image_newline_tokens = \
            vit_feature_size // token_compression_factor

        return (num_global_image_tokens + num_sep_tokens +
                num_hd_patch_tokens + num_hd_newline_tokens +
                num_global_image_newline_tokens)

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        processor: Optional[ProcessorMixin] = None,
    ) -> int:
        hf_config = self.get_hf_config()
        vision_encoder_name = hf_config.img_processor
        if vision_encoder_name is None:
            vision_encoder_name = SIGLIP_NAME
        prepro_config = VISION_ENCODER_TO_PROCESSING_CONFIG[
            vision_encoder_name]
        vit_image_size = prepro_config['vit_image_size']
        vit_patch_size = prepro_config['vit_patch_size']
        token_compression_factor = prepro_config['token_compression_factor']

        dynamic_hd_size = self.get_dynamic_hd(processor=processor)

        image_num_tokens = self._compute_num_image_tokens(
            image_width,
            image_height,
            dynamic_hd_size=dynamic_hd_size,
            vit_image_size=vit_image_size,
            vit_patch_size=vit_patch_size,
            token_compression_factor=token_compression_factor,
        )

        return image_num_tokens
