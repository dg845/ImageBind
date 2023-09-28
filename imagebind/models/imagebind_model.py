#!/usr/bin/env python3
# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
from functools import partial
from types import SimpleNamespace

import torch
import torch.nn as nn

from imagebind.models.helpers import (EinOpsRearrange, LearnableLogitScaling, Normalize,
                            SelectElement, SelectEOSAndProject)
from imagebind.models.multimodal_preprocessors import (AudioPreprocessor,
                                             IMUPreprocessor, PadIm2Video,
                                             PatchEmbedGeneric,
                                             RGBDTPreprocessor,
                                             SpatioTemporalPosEmbeddingHelper,
                                             TextPreprocessor,
                                             ThermalPreprocessor)
from imagebind.models.transformer import MultiheadAttention, SimpleTransformer

ModalityType = SimpleNamespace(
    VISION="vision",
    TEXT="text",
    AUDIO="audio",
    THERMAL="thermal",
    DEPTH="depth",
    IMU="imu",
)


class ImageBindModel(nn.Module):
    def __init__(
        self,
        video_frames=2,
        image_size=224,
        kernel_size=(2, 14, 14),
        audio_kernel_size=16,
        audio_stride=10,
        out_embed_dim=768,
        vision_embed_dim=1024,
        vision_num_blocks=24,
        vision_num_heads=16,
        vision_intermediate_size=None,
        audio_embed_dim=768,
        audio_num_blocks=12,
        audio_num_heads=12,
        audio_num_mel_bins=128,
        audio_target_len=204,
        audio_drop_path=0.1,
        audio_intermediate_size=None,
        text_context_length=77,
        text_vocab_size=49408,
        text_embed_dim=768,
        text_num_blocks=12,
        text_num_heads=12,
        text_intermediate_size=None,
        depth_embed_dim=384,
        depth_kernel_size=16,
        depth_num_blocks=12,
        depth_num_heads=8,
        depth_drop_path=0.0,
        depth_intermediate_size=None,
        thermal_embed_dim=768,
        thermal_kernel_size=16,
        thermal_num_blocks=12,
        thermal_num_heads=12,
        thermal_drop_path=0.0,
        thermal_intermediate_size=None,
        imu_num_features=6,
        imu_seq_len=2000,
        imu_embed_dim=512,
        imu_kernel_size=8,
        imu_patch_features=48,
        imu_num_blocks=6,
        imu_num_heads=8,
        imu_drop_path=0.7,
        imu_intermediate_size=None,
    ):
        super().__init__()

        self.modality_preprocessors = self._create_modality_preprocessors(
            video_frames,
            image_size,
            vision_embed_dim,
            kernel_size,
            text_context_length,
            text_vocab_size,
            text_embed_dim,
            audio_embed_dim,
            audio_kernel_size,
            audio_stride,
            audio_num_mel_bins,
            audio_target_len,
            depth_embed_dim,
            depth_kernel_size,
            thermal_embed_dim,
            thermal_kernel_size,
            imu_num_features,
            imu_seq_len,
            imu_kernel_size,
            imu_patch_features,
            imu_embed_dim,
        )

        self.modality_trunks = self._create_modality_trunks(
            vision_embed_dim,
            vision_num_blocks,
            vision_num_heads,
            vision_intermediate_size,
            text_embed_dim,
            text_num_blocks,
            text_num_heads,
            text_intermediate_size,
            audio_embed_dim,
            audio_num_blocks,
            audio_num_heads,
            audio_intermediate_size,
            audio_drop_path,
            depth_embed_dim,
            depth_num_blocks,
            depth_num_heads,
            depth_intermediate_size,
            depth_drop_path,
            thermal_embed_dim,
            thermal_num_blocks,
            thermal_num_heads,
            thermal_intermediate_size,
            thermal_drop_path,
            imu_embed_dim,
            imu_num_blocks,
            imu_num_heads,
            imu_intermediate_size,
            imu_drop_path,
        )

        self.modality_heads = self._create_modality_heads(
            out_embed_dim,
            vision_embed_dim,
            text_embed_dim,
            audio_embed_dim,
            depth_embed_dim,
            thermal_embed_dim,
            imu_embed_dim,
        )

        self.modality_postprocessors = self._create_modality_postprocessors(
            out_embed_dim
        )

    def _create_modality_preprocessors(
        self,
        video_frames=2,
        image_size=224,
        vision_embed_dim=1024,
        kernel_size=(2, 14, 14),
        text_context_length=77,
        text_vocab_size=49408,
        text_embed_dim=768,
        audio_embed_dim=768,
        audio_kernel_size=16,
        audio_stride=10,
        audio_num_mel_bins=128,
        audio_target_len=204,
        depth_embed_dim=768,
        depth_kernel_size=16,
        thermal_embed_dim=768,
        thermal_kernel_size=16,
        imu_num_features=6,
        imu_seq_len=2000,
        imu_kernel_size=8,
        imu_patch_features=48,
        imu_embed_dim=512,
    ):
        rgbt_stem = PatchEmbedGeneric(
            proj_stem=[
                PadIm2Video(pad_type="repeat", ntimes=2),
                nn.Conv3d(
                    in_channels=3,
                    kernel_size=kernel_size,
                    out_channels=vision_embed_dim,
                    stride=kernel_size,
                    bias=False,
                ),
            ]
        )
        rgbt_preprocessor = RGBDTPreprocessor(
            img_size=[3, video_frames, image_size, image_size],
            num_cls_tokens=1,
            pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
            rgbt_stem=rgbt_stem,
            depth_stem=None,
        )

        text_preprocessor = TextPreprocessor(
            context_length=text_context_length,
            vocab_size=text_vocab_size,
            embed_dim=text_embed_dim,
            causal_masking=True,
        )

        audio_stem = PatchEmbedGeneric(
            proj_stem=[
                nn.Conv2d(
                    in_channels=1,
                    kernel_size=audio_kernel_size,
                    stride=audio_stride,
                    out_channels=audio_embed_dim,
                    bias=False,
                ),
            ],
            norm_layer=nn.LayerNorm(normalized_shape=audio_embed_dim),
        )
        audio_preprocessor = AudioPreprocessor(
            img_size=[1, audio_num_mel_bins, audio_target_len],
            num_cls_tokens=1,
            pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
            audio_stem=audio_stem,
        )

        depth_stem = PatchEmbedGeneric(
            [
                nn.Conv2d(
                    kernel_size=depth_kernel_size,
                    in_channels=1,
                    out_channels=depth_embed_dim,
                    stride=depth_kernel_size,
                    bias=False,
                ),
            ],
            norm_layer=nn.LayerNorm(normalized_shape=depth_embed_dim),
        )

        depth_preprocessor = RGBDTPreprocessor(
            img_size=[1, image_size, image_size],
            num_cls_tokens=1,
            pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
            rgbt_stem=None,
            depth_stem=depth_stem,
        )

        thermal_stem = PatchEmbedGeneric(
            [
                nn.Conv2d(
                    kernel_size=thermal_kernel_size,
                    in_channels=1,
                    out_channels=thermal_embed_dim,
                    stride=thermal_kernel_size,
                    bias=False,
                ),
            ],
            norm_layer=nn.LayerNorm(normalized_shape=thermal_embed_dim),
        )
        thermal_preprocessor = ThermalPreprocessor(
            img_size=[1, image_size, image_size],
            num_cls_tokens=1,
            pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
            thermal_stem=thermal_stem,
        )

        imu_stem = PatchEmbedGeneric(
            [
                nn.Linear(
                    in_features=imu_patch_features,
                    out_features=imu_embed_dim,
                    bias=False,
                ),
            ],
            norm_layer=nn.LayerNorm(normalized_shape=imu_embed_dim),
        )

        imu_preprocessor = IMUPreprocessor(
            img_size=[imu_num_features, imu_seq_len],
            num_cls_tokens=1,
            kernel_size=imu_kernel_size,
            embed_dim=imu_embed_dim,
            pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
            imu_stem=imu_stem,
        )

        modality_preprocessors = {
            ModalityType.VISION: rgbt_preprocessor,
            ModalityType.TEXT: text_preprocessor,
            ModalityType.AUDIO: audio_preprocessor,
            ModalityType.DEPTH: depth_preprocessor,
            ModalityType.THERMAL: thermal_preprocessor,
            ModalityType.IMU: imu_preprocessor,
        }

        return nn.ModuleDict(modality_preprocessors)

    def _create_modality_trunks(
        self,
        vision_embed_dim=1024,
        vision_num_blocks=24,
        vision_num_heads=16,
        vision_intermediate_size=None,
        text_embed_dim=768,
        text_num_blocks=12,
        text_num_heads=12,
        text_intermediate_size=None,
        audio_embed_dim=768,
        audio_num_blocks=12,
        audio_num_heads=12,
        audio_intermediate_size=None,
        audio_drop_path=0.0,
        depth_embed_dim=768,
        depth_num_blocks=12,
        depth_num_heads=12,
        depth_intermediate_size=None,
        depth_drop_path=0.0,
        thermal_embed_dim=768,
        thermal_num_blocks=12,
        thermal_num_heads=12,
        thermal_intermediate_size=None,
        thermal_drop_path=0.0,
        imu_embed_dim=512,
        imu_num_blocks=6,
        imu_num_heads=8,
        imu_intermediate_size=None,
        imu_drop_path=0.7,
    ):
        def instantiate_trunk(
            embed_dim, num_blocks, num_heads, intermediate_size, pre_transformer_ln, add_bias_kv, drop_path
        ):
            return SimpleTransformer(
                embed_dim=embed_dim,
                num_blocks=num_blocks,
                intermediate_size=intermediate_size,
                ffn_dropout_rate=0.0,
                drop_path_rate=drop_path,
                attn_target=partial(
                    MultiheadAttention,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    bias=True,
                    add_bias_kv=add_bias_kv,
                ),
                pre_transformer_layer=nn.Sequential(
                    nn.LayerNorm(embed_dim, eps=1e-6)
                    if pre_transformer_ln
                    else nn.Identity(),
                    EinOpsRearrange("b l d -> l b d"),
                ),
                post_transformer_layer=EinOpsRearrange("l b d -> b l d"),
            )

        modality_trunks = {}
        modality_trunks[ModalityType.VISION] = instantiate_trunk(
            vision_embed_dim,
            vision_num_blocks,
            vision_num_heads,
            vision_intermediate_size,
            pre_transformer_ln=True,
            add_bias_kv=False,
            drop_path=0.0,
        )
        modality_trunks[ModalityType.TEXT] = instantiate_trunk(
            text_embed_dim,
            text_num_blocks,
            text_num_heads,
            text_intermediate_size,
            pre_transformer_ln=False,
            add_bias_kv=False,
            drop_path=0.0,
        )
        modality_trunks[ModalityType.AUDIO] = instantiate_trunk(
            audio_embed_dim,
            audio_num_blocks,
            audio_num_heads,
            audio_intermediate_size,
            pre_transformer_ln=False,
            add_bias_kv=True,
            drop_path=audio_drop_path,
        )
        modality_trunks[ModalityType.DEPTH] = instantiate_trunk(
            depth_embed_dim,
            depth_num_blocks,
            depth_num_heads,
            depth_intermediate_size,
            pre_transformer_ln=False,
            add_bias_kv=True,
            drop_path=depth_drop_path,
        )
        modality_trunks[ModalityType.THERMAL] = instantiate_trunk(
            thermal_embed_dim,
            thermal_num_blocks,
            thermal_num_heads,
            thermal_intermediate_size,
            pre_transformer_ln=False,
            add_bias_kv=True,
            drop_path=thermal_drop_path,
        )
        modality_trunks[ModalityType.IMU] = instantiate_trunk(
            imu_embed_dim,
            imu_num_blocks,
            imu_num_heads,
            imu_intermediate_size,
            pre_transformer_ln=False,
            add_bias_kv=True,
            drop_path=imu_drop_path,
        )

        return nn.ModuleDict(modality_trunks)

    def _create_modality_heads(
        self,
        out_embed_dim,
        vision_embed_dim,
        text_embed_dim,
        audio_embed_dim,
        depth_embed_dim,
        thermal_embed_dim,
        imu_embed_dim,
    ):
        modality_heads = {}

        modality_heads[ModalityType.VISION] = nn.Sequential(
            nn.LayerNorm(normalized_shape=vision_embed_dim, eps=1e-6),
            SelectElement(index=0),
            nn.Linear(vision_embed_dim, out_embed_dim, bias=False),
        )

        modality_heads[ModalityType.TEXT] = SelectEOSAndProject(
            proj=nn.Sequential(
                nn.LayerNorm(normalized_shape=text_embed_dim, eps=1e-6),
                nn.Linear(text_embed_dim, out_embed_dim, bias=False),
            )
        )

        modality_heads[ModalityType.AUDIO] = nn.Sequential(
            nn.LayerNorm(normalized_shape=audio_embed_dim, eps=1e-6),
            SelectElement(index=0),
            nn.Linear(audio_embed_dim, out_embed_dim, bias=False),
        )

        modality_heads[ModalityType.DEPTH] = nn.Sequential(
            nn.LayerNorm(normalized_shape=depth_embed_dim, eps=1e-6),
            SelectElement(index=0),
            nn.Linear(depth_embed_dim, out_embed_dim, bias=False),
        )

        modality_heads[ModalityType.THERMAL] = nn.Sequential(
            nn.LayerNorm(normalized_shape=thermal_embed_dim, eps=1e-6),
            SelectElement(index=0),
            nn.Linear(thermal_embed_dim, out_embed_dim, bias=False),
        )

        modality_heads[ModalityType.IMU] = nn.Sequential(
            nn.LayerNorm(normalized_shape=imu_embed_dim, eps=1e-6),
            SelectElement(index=0),
            nn.Dropout(p=0.5),
            nn.Linear(imu_embed_dim, out_embed_dim, bias=False),
        )

        return nn.ModuleDict(modality_heads)

    def _create_modality_postprocessors(self, out_embed_dim):
        modality_postprocessors = {}

        modality_postprocessors[ModalityType.VISION] = Normalize(dim=-1)
        modality_postprocessors[ModalityType.TEXT] = nn.Sequential(
            Normalize(dim=-1), LearnableLogitScaling(learnable=True)
        )
        modality_postprocessors[ModalityType.AUDIO] = nn.Sequential(
            Normalize(dim=-1),
            LearnableLogitScaling(logit_scale_init=20.0, learnable=False),
        )
        modality_postprocessors[ModalityType.DEPTH] = nn.Sequential(
            Normalize(dim=-1),
            LearnableLogitScaling(logit_scale_init=5.0, learnable=False),
        )
        modality_postprocessors[ModalityType.THERMAL] = nn.Sequential(
            Normalize(dim=-1),
            LearnableLogitScaling(logit_scale_init=10.0, learnable=False),
        )
        modality_postprocessors[ModalityType.IMU] = nn.Sequential(
            Normalize(dim=-1),
            LearnableLogitScaling(logit_scale_init=5.0, learnable=False),
        )

        return nn.ModuleDict(modality_postprocessors)

    def forward(self, inputs):
        outputs = {}
        for modality_key, modality_value in inputs.items():
            reduce_list = (
                modality_value.ndim >= 5
            )  # Audio and Video inputs consist of multiple clips
            if reduce_list:
                B, S = modality_value.shape[:2]
                modality_value = modality_value.reshape(
                    B * S, *modality_value.shape[2:]
                )

            if modality_value is not None:
                modality_value = self.modality_preprocessors[modality_key](
                    **{modality_key: modality_value}
                )
                trunk_inputs = modality_value["trunk"]
                head_inputs = modality_value["head"]
                modality_value = self.modality_trunks[modality_key](**trunk_inputs)
                modality_value = self.modality_heads[modality_key](
                    modality_value, **head_inputs
                )
                modality_value = self.modality_postprocessors[modality_key](
                    modality_value
                )

                if reduce_list:
                    modality_value = modality_value.reshape(B, S, -1)
                    modality_value = modality_value.mean(dim=1)

                outputs[modality_key] = modality_value

        return outputs


def imagebind_huge(pretrained=False):
    model = ImageBindModel(
        vision_embed_dim=1280,
        vision_num_blocks=32,
        vision_num_heads=16,
        text_embed_dim=1024,
        text_num_blocks=24,
        text_num_heads=16,
        out_embed_dim=1024,
        audio_drop_path=0.1,
        imu_drop_path=0.7,
    )

    if pretrained:
        if not os.path.exists(".checkpoints/imagebind_huge.pth"):
            print(
                "Downloading imagebind weights to .checkpoints/imagebind_huge.pth ..."
            )
            os.makedirs(".checkpoints", exist_ok=True)
            torch.hub.download_url_to_file(
                "https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth",
                ".checkpoints/imagebind_huge.pth",
                progress=True,
            )

        model.load_state_dict(torch.load(".checkpoints/imagebind_huge.pth"))

    return model


def imagebind_test():
    model = ImageBindModel(
        video_frames=2,
        image_size=30,
        kernel_size=(2, 2, 2),
        audio_kernel_size=4,
        audio_stride=2,
        out_embed_dim=32,
        vision_embed_dim=32,
        vision_num_blocks=5,
        vision_num_heads=4,
        vision_intermediate_size=37,
        audio_embed_dim=32,
        audio_num_blocks=5,
        audio_num_heads=4,
        audio_num_mel_bins=128,
        audio_target_len=204,
        audio_drop_path=0.1,
        audio_intermediate_size=37,
        text_context_length=512,
        text_vocab_size=99,
        text_embed_dim=32,
        text_num_blocks=5,
        text_num_heads=4,
        text_intermediate_size=37,
        depth_embed_dim=32,
        depth_kernel_size=2,
        depth_num_blocks=5,
        depth_num_heads=4,
        depth_drop_path=0.0,
        depth_intermediate_size=37,
        thermal_embed_dim=32,
        thermal_kernel_size=2,
        thermal_num_blocks=5,
        thermal_num_heads=4,
        thermal_drop_path=0.0,
        thermal_intermediate_size=37,
        imu_num_features=6,
        imu_seq_len=30,
        imu_embed_dim=32,
        imu_kernel_size=2,
        imu_patch_features=15,
        imu_num_blocks=5,
        imu_num_heads=4,
        imu_drop_path=0.7,
        imu_intermediate_size=37,
    )

    return model


class ImageBindEncoder(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_blocks,
        num_heads,
        intermediate_size,
        pre_transformer_ln,
        add_bias_kv,
        drop_path,
        out_embed_dim=1024,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.pre_transformer_ln = pre_transformer_ln
        self.add_bias_kv = add_bias_kv
        self.drop_path = drop_path
        self.out_embed_dim = out_embed_dim

        self.preprocessor = None
        self.trunk = None
        self.head = None
        self.postprocessor = None
    
    def instantiate_trunk(self):
        return SimpleTransformer(
            embed_dim=self.embed_dim,
            num_blocks=self.num_blocks,
            intermediate_size=self.intermediate_size,
            ffn_dropout_rate=0.0,
            drop_path_rate=self.drop_path,
            attn_target=partial(
                MultiheadAttention,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                bias=True,
                add_bias_kv=self.add_bias_kv,
            ),
            pre_transformer_layer=nn.Sequential(
                nn.LayerNorm(self.embed_dim, eps=1e-6)
                if self.pre_transformer_ln
                else nn.Identity(),
                EinOpsRearrange("b l d -> l b d"),
            ),
            post_transformer_layer=EinOpsRearrange("l b d -> b l d"),
        )
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Audio and Video inputs consist of multiple clips
        reduce_list = inputs.ndim >= 5
        if reduce_list:
            B, S = inputs.shape[:2]
            inputs = inputs.reshape(
                B * S, *inputs.shape[2:]
            )

        preprocessed_inputs = self.preprocessor(inputs)
        trunk_output = self.trunk(preprocessed_inputs)
        head_output = self.head(trunk_output)
        output = self.postprocessor(head_output)

        if reduce_list:
            output = output.reshape(B, S, -1)
            output = output.mean(dim=1)

        return output


class ImageBindTextEncoder(ImageBindEncoder):
    def __init__(
        self,
        text_context_length=77,
        text_vocab_size=49408,
        text_embed_dim=1024,
        text_num_blocks=24,
        text_num_heads=16,
        text_intermediate_size=None,
        text_pre_transformer_ln=False,
        text_add_bias_kv=False,
        text_drop_path=0.0,
        out_embed_dim=1024,
    ):
        super().__init__(
            embed_dim=text_embed_dim,
            num_blocks=text_num_blocks,
            num_heads=text_num_heads,
            intermediate_size=text_intermediate_size,
            pre_transformer_ln=text_pre_transformer_ln,
            add_bias_kv=text_add_bias_kv,
            drop_path=text_drop_path,
            out_embed_dim=out_embed_dim,
        )

        self.context_length = text_context_length
        self.vocab_size = text_vocab_size

        self.preprocessor = TextPreprocessor(
            context_length=text_context_length,
            vocab_size=text_vocab_size,
            embed_dim=text_embed_dim,
            causal_masking=True,
        )

        self.trunk = self.instantiate_trunk()

        self.head = SelectEOSAndProject(
            proj=nn.Sequential(
                nn.LayerNorm(normalized_shape=text_embed_dim, eps=1e-6),
                nn.Linear(text_embed_dim, self.out_embed_dim, bias=False),
            )
        )

        self.post_processor = nn.Sequential(
            Normalize(dim=-1), LearnableLogitScaling(learnable=True)
        )


def imagebind_test_text_encoder():
    text_encoder = ImageBindTextEncoder(
        text_context_length=512,
        text_vocab_size=99,
        text_embed_dim=32,
        text_num_blocks=5,
        text_num_heads=4,
        text_intermediate_size=37,
        text_pre_transformer_ln=False,
        text_add_bias_kv=False,
        text_drop_path=0.0,
        out_embed_dim=32,
    )

    return text_encoder


class ImageBindImageLikeEncoder(ImageBindEncoder):
    def __init__(
        self,
        image_size,
        in_channels,
        kernel_size,
        stride,
        embed_dim,
        num_blocks,
        num_heads,
        intermediate_size,
        pre_transformer_ln,
        add_bias_kv,
        drop_path,
        logit_scale_init,
        out_embed_dim=1024,
    ):
        super().__init__(
            embed_dim=embed_dim,
            num_blocks=num_blocks,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            pre_transformer_ln=pre_transformer_ln,
            add_bias_kv=add_bias_kv,
            drop_path=drop_path,
            out_embed_dim=out_embed_dim,
        )

        self.image_size = image_size
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.logit_scale_init = logit_scale_init
    
    def create_2d_stem(self):
        return PatchEmbedGeneric(
            proj_stem=[
                nn.Conv2d(
                    in_channels=self.in_channels,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    out_channels=self.embed_dim,
                    bias=False,
                ),
            ],
            norm_layer=nn.LayerNorm(normalized_shape=self.embed_dim),
        )
    
    def create_3d_stem(self, ntimes=2):
        return PatchEmbedGeneric(
            proj_stem=[
                PadIm2Video(pad_type="repeat", ntimes=ntimes),
                nn.Conv3d(
                    in_channels=self.in_channels,
                    kernel_size=self.kernel_size,
                    out_channels=self.embed_dim,
                    stride=self.kernel_size,
                    bias=False,
                ),
            ]
        )
    
    def create_image_like_preprocessor(self, img_size, rgbt_stem=None, depth_stem=None):
        return RGBDTPreprocessor(
            img_size=img_size,
            num_cls_tokens=1,
            pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
            rgbt_stem=rgbt_stem,
            depth_stem=depth_stem,
        )
    
    def create_image_like_head(self):
        return nn.Sequential(
            nn.LayerNorm(normalized_shape=self.embed_dim, eps=1e-6),
            SelectElement(index=0),
            nn.Linear(self.embed_dim, self.out_embed_dim, bias=False),
        )
    
    def create_image_like_postprocessor(self, learnable=False):
        if self.logit_scale_init is not None:
            return nn.Sequential(
                Normalize(dim=-1),
                LearnableLogitScaling(logit_scale_init=self.logit_scale_init, learnable=learnable),
            )
        else:
            return Normalize(dim=-1)


class ImageBindVisionEncoder(ImageBindImageLikeEncoder):
    def __init__(
        self,
        image_size=224,
        video_frames=2,
        vision_in_channels=3,
        vision_kernel_size=(2, 14, 14),
        vision_stride=(2, 14, 14),
        vision_embed_dim=1280,
        vision_num_blocks=32,
        vision_num_heads=16,
        vision_intermediate_size=None,
        vision_pre_transformer_ln=True,
        vision_add_bias_kv=False,
        vision_drop_path=0.0,
        vision_logit_scale_init=None,
        out_embed_dim=1024,
    ):
        super().__init__(
            image_size=image_size,
            in_channels=vision_in_channels,
            kernel_size=vision_kernel_size,
            stride=vision_stride,
            embed_dim=vision_embed_dim,
            num_blocks=vision_num_blocks,
            num_heads=vision_num_heads,
            intermediate_size=vision_intermediate_size,
            pre_transformer_ln=vision_pre_transformer_ln,
            add_bias_kv=vision_add_bias_kv,
            drop_path=vision_drop_path,
            logit_scale_init=vision_logit_scale_init,
            out_embed_dim=out_embed_dim,
        )

        self.num_frames = video_frames
        self.img_shape = [self.in_channels, self.num_frames, self.image_size, self.image_size]

        stem = self.create_3d_stem()
        self.preprocessor = self.create_image_like_preprocessor(self.img_shape, rgbt_stem=stem)

        self.trunk = self.instantiate_trunk()

        self.head = self.create_image_like_head()

        self.postprocessor = self.create_image_like_postprocessor()


def imagebind_test_vision_encoder():
    vision_encoder = ImageBindVisionEncoder(
        image_size=30,
        video_frames=2,
        vision_in_channels=3,
        vision_kernel_size=(2, 2, 2),
        vision_stride=(2, 2, 2),
        vision_embed_dim=32,
        vision_num_blocks=5,
        vision_num_heads=4,
        vision_intermediate_size=37,
        vision_pre_transformer_ln=True,
        vision_add_bias_kv=False,
        vision_drop_path=0.0,
        vision_logit_scale_init=None,
        out_embed_dim=32,
    )

    return vision_encoder


class ImageBindAudioEncoder(ImageBindImageLikeEncoder):
    def __init__(
        self,
        image_size=224,
        audio_in_channels=1,
        audio_kernel_size=16,
        audio_stride=10,
        audio_num_mel_bins = 128,
        audio_target_len=204,
        audio_embed_dim=768,
        audio_num_blocks=12,
        audio_num_heads=12,
        audio_intermediate_size=None,
        audio_pre_transformer_ln=False,
        audio_add_bias_kv=True,
        audio_drop_path=0.1,
        audio_logit_scale_init=20.0,
        out_embed_dim=1024,
    ):
        super().__init__(
            image_size=image_size,
            in_channels=audio_in_channels,
            kernel_size=audio_kernel_size,
            stride=audio_stride,
            embed_dim=audio_embed_dim,
            num_blocks=audio_num_blocks,
            num_heads=audio_num_heads,
            intermediate_size=audio_intermediate_size,
            pre_transformer_ln=audio_pre_transformer_ln,
            add_bias_kv=audio_add_bias_kv,
            drop_path=audio_drop_path,
            logit_scale_init=audio_logit_scale_init,
            out_embed_dim=out_embed_dim,
        )

        self.num_mel_bins = audio_num_mel_bins
        self.target_len = audio_target_len
        self.img_shape = [self.in_channels, self.num_mel_bins, self.target_len]

        stem = self.create_2d_stem()
        self.preprocessor = AudioPreprocessor(
            img_size=self.img_shape,
            num_cls_tokens=1,
            pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
            audio_stem=stem,
        )

        self.trunk = self.instantiate_trunk()

        self.head = self.create_image_like_head()

        self.postprocessor = self.create_image_like_postprocessor()


def imagebind_test_audio_encoder():
    audio_encoder = ImageBindAudioEncoder(
        image_size=30,
        audio_in_channels=1,
        audio_kernel_size=4,
        audio_stride=2,
        audio_num_mel_bins=128,
        audio_target_len=204,
        audio_embed_dim=32,
        audio_num_blocks=5,
        audio_num_heads=4,
        audio_intermediate_size=37,
        audio_pre_transformer_ln=False,
        audio_add_bias_kv=True,
        audio_drop_path=0.1,
        audio_logit_scale_init=20.0,
        out_embed_dim=32,
    )

    return audio_encoder


class ImageBindDepthEncoder(ImageBindImageLikeEncoder):
    def __init__(
        self,
        image_size=224,
        depth_in_channels=1,
        depth_kernel_size=16,
        depth_stride=16,
        depth_embed_dim=384,
        depth_num_blocks=12,
        depth_num_heads=8,
        depth_intermediate_size=None,
        depth_pre_transformer_ln=False,
        depth_add_bias_kv=True,
        depth_drop_path=0.0,
        depth_logit_scale_init=5.0,
        out_embed_dim=1024,
    ):
        super().__init__(
            image_size=image_size,
            in_channels=depth_in_channels,
            kernel_size=depth_kernel_size,
            stride=depth_stride,
            embed_dim=depth_embed_dim,
            num_blocks=depth_num_blocks,
            num_heads=depth_num_heads,
            intermediate_size=depth_intermediate_size,
            pre_transformer_ln=depth_pre_transformer_ln,
            add_bias_kv=depth_add_bias_kv,
            drop_path=depth_drop_path,
            logit_scale_init=depth_logit_scale_init,
            out_embed_dim=out_embed_dim,
        )

        self.img_shape = [self.in_channels, self.image_size, self.image_size]

        stem = self.create_2d_stem()
        self.preprocessor = self.create_image_like_preprocessor(img_size=self.img_shape, depth_stem=stem)

        self.trunk = self.instantiate_trunk()

        self.head = self.create_image_like_head()

        self.postprocessor = self.create_image_like_postprocessor()


def imagebind_test_depth_encoder():
    depth_encoder = ImageBindDepthEncoder(
        image_size=30,
        depth_in_channels=1,
        depth_kernel_size=2,
        depth_stride=2,
        depth_embed_dim=32,
        depth_num_blocks=5,
        depth_num_heads=4,
        depth_intermediate_size=37,
        depth_pre_transformer_ln=False,
        depth_add_bias_kv=True,
        depth_drop_path=0.0,
        depth_logit_scale_init=5.0,
        out_embed_dim=32,
    )

    return depth_encoder


class ImageBindThermalEncoder(ImageBindImageLikeEncoder):
    def __init__(
        self,
        image_size=224,
        thermal_in_channels=1,
        thermal_kernel_size=16,
        thermal_stride=16,
        thermal_embed_dim=768,
        thermal_num_blocks=12,
        thermal_num_heads=12,
        thermal_intermediate_size=None,
        thermal_pre_transformer_ln=False,
        thermal_add_bias_kv=True,
        thermal_drop_path=0.0,
        thermal_logit_scale_init=10.0,
        out_embed_dim=1024,
    ):
        super().__init__(
            image_size=image_size,
            in_channels=thermal_in_channels,
            kernel_size=thermal_kernel_size,
            stride=thermal_stride,
            embed_dim=thermal_embed_dim,
            num_blocks=thermal_num_blocks,
            num_heads=thermal_num_heads,
            intermediate_size=thermal_intermediate_size,
            pre_transformer_ln=thermal_pre_transformer_ln,
            add_bias_kv=thermal_add_bias_kv,
            drop_path=thermal_drop_path,
            logit_scale_init=thermal_logit_scale_init,
            out_embed_dim=out_embed_dim,
        )

        self.img_shape = [self.in_channels, self.image_size, self.image_size]

        stem = self.create_2d_stem()
        self.preprocessor = self.create_image_like_preprocessor(img_size=self.img_shape, rgbt_stem=stem)

        self.trunk = self.instantiate_trunk()

        self.head = self.create_image_like_head()

        self.postprocessor = self.create_image_like_postprocessor()


def imagebind_test_thermal_encoder():
    thermal_encoder = ImageBindThermalEncoder(
        image_size=30,
        thermal_in_channels=1,
        thermal_kernel_size=2,
        thermal_stride=2,
        thermal_embed_dim=32,
        thermal_num_blocks=5,
        thermal_num_heads=4,
        thermal_intermediate_size=37,
        thermal_pre_transformer_ln=False,
        thermal_add_bias_kv=True,
        thermal_drop_path=0.0,
        thermal_logit_scale_init=10.0,
        out_embed_dim=32,
    )

    return thermal_encoder


class ImageBindImuEncoder(ImageBindEncoder):
    def __init__(
        self,
        imu_num_features=6,
        imu_seq_len=2000,
        imu_kernel_size=8,
        imu_patch_features=48,
        imu_embed_dim=512,
        imu_num_blocks=6,
        imu_num_heads=8,
        imu_intermediate_size=None,
        imu_pre_transformer_ln=False,
        imu_add_bias_kv=True,
        imu_drop_path=0.7,
        imu_logit_scale_init=5.0,
        out_embed_dim=1024,
    ):
        super().__init__(
            embed_dim=imu_embed_dim,
            num_blocks=imu_num_blocks,
            num_heads=imu_num_heads,
            intermediate_size=imu_intermediate_size,
            pre_transformer_ln=imu_pre_transformer_ln,
            add_bias_kv=imu_add_bias_kv,
            drop_path=imu_drop_path,
            out_embed_dim=out_embed_dim,
        )

        self.num_features = imu_num_features
        self.seq_len = imu_seq_len
        self.kernel_size = imu_kernel_size
        self.patch_features = imu_patch_features

        stem = PatchEmbedGeneric(
            [
                nn.Linear(
                    in_features=imu_patch_features,
                    out_features=imu_embed_dim,
                    bias=False,
                ),
            ],
            norm_layer=nn.LayerNorm(normalized_shape=imu_embed_dim),
        )
        self.preprocessor = IMUPreprocessor(
            img_size=[imu_num_features, imu_seq_len],
            num_cls_tokens=1,
            kernel_size=imu_kernel_size,
            embed_dim=imu_embed_dim,
            pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
            imu_stem=stem,
        )

        self.trunk = self.instantiate_trunk()

        self.head = nn.Sequential(
            nn.LayerNorm(normalized_shape=imu_embed_dim, eps=1e-6),
            SelectElement(index=0),
            nn.Dropout(p=0.5),
            nn.Linear(imu_embed_dim, out_embed_dim, bias=False),
        )

        self.post_processor = nn.Sequential(
            Normalize(dim=-1),
            LearnableLogitScaling(logit_scale_init=imu_logit_scale_init, learnable=False),
        )


def imagebind_test_imu_encoder():
    imu_encoder = ImageBindImuEncoder(
        imu_num_features=6,
        imu_seq_len=30,
        imu_kernel_size=2,
        imu_patch_features=15,
        imu_embed_dim=32,
        imu_num_blocks=5,
        imu_num_heads=4,
        imu_intermediate_size=37,
        imu_pre_transformer_ln=False,
        imu_add_bias_kv=True,
        imu_drop_path=0.7,
        imu_logit_scale_init=5.0,
    )

    return imu_encoder
