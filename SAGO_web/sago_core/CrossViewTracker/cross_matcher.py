#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path as osp

import torch
import torch.nn as nn

from sago_core.CrossViewTracker.lib.make_sam import make_sam_from_state_dict


class CrossMatch:
    def __init__(self, device="cuda:0", dtype=torch.float32,
                 model_path="../../sam_v2/checkpoints/sam2_hiera_large.pt"):
        self.device = device
        self.dtype = dtype

        model_config_dict, model = make_sam_from_state_dict(model_path)
        self.model_name = osp.basename(model_path)

        self.sam = model
        self.sam.to(device=device, dtype=dtype)
        total_params = sum(p.numel() for p in model.parameters())
        # print(f"Total parameters: {total_params}")
        self.img_encoder_config_dict = {"max_side_length": 1024,
                                        "use_square_sizing": True}

        self.token_hw = [64, 64]
        self.image_height = 1080
        self.image_width = 1920

    @torch.inference_mode
    def image_encoder(self, np_img):
        img_code, self.token_hw, preencode_img_hw = self.sam.encode_image(
            np_img,
            **self.img_encoder_config_dict)
        self.image_height, self.image_width = preencode_img_hw
        return img_code

    @torch.inference_mode
    def image_encode_batch(self, np_img_batch):
        img_codes, self.token_hw, preencode_img_hw = self.sam.encode_image(
            np_img_batch,
            **self.img_encoder_config_dict)
        self.image_height, self.image_width = preencode_img_hw
        return img_codes

    def prompt_encoder(self, box=None, fg_pts=None, bg_pts=None):
        if bg_pts is None:
            bg_pts = []
        if fg_pts is None:
            fg_pts = []
        if box is None:
            box = []
        return self.sam.encode_prompts(box, fg_pts, bg_pts)

    def output_masks(self, img_code, prompt_code):
        return self.sam.generate_masks(img_code, prompt_code)

    def decode_mask(self, img_code, prompt_code):
        grid_posenc = self.sam.coordinate_encoder.get_grid_position_encoding(self.token_hw)
        ref_preds, _, ref_ptrs, obj_score = self.sam.mask_decoder(img_code,
                                                                  prompt_code,
                                                                  grid_posenc)
        return ref_preds, ref_ptrs, obj_score

    def make_hires_mask_uint8(self, mask_prediction, mask_threshold=0.0, height=None, width=None):
        """
        Helper used to draw a high-resolution binary uint8 (numpy)
        version of a given mask prediction. Expects a single mask
        prediction, but will work with a variety of input shapes.
        Supports inputs of shape:
            BxNxHxW, NxHxW, HxW

        Will return result with matching shape dimensions,
        but with output H/W sizing in numpy uint8 format
        """

        # Set up dimension padding (for interpolation func) & post-indexing to remove padded dimensions
        preds, squeeze_idx = mask_prediction, slice(None)
        if preds.ndim == 3:
            preds = mask_prediction[None]
            squeeze_idx = (0,)
        elif preds.ndim == 2:
            preds = mask_prediction[None, None]
            squeeze_idx = (0, 0)
        if height is None: height = self.image_height
        if width is None: width = self.image_width

        mask_upscale = nn.functional.interpolate(preds,
                                                 size=(height, width),
                                                 mode="bilinear", align_corners=False)
        return ((mask_upscale[squeeze_idx] > mask_threshold) * 255).byte()  # .cpu().numpy()
