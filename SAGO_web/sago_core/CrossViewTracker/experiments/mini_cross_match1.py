#!/usr/bin/env python3
# -*- coding: utf-8 -*-
try:
    import lib  # NOQA
except ModuleNotFoundError:
    import os
    import sys

    parent_folder = os.path.dirname(os.path.dirname(__file__))
    if "lib" in os.listdir(parent_folder):
        sys.path.insert(0, parent_folder)
    else:
        raise ImportError("Can't find path to lib folder!")

import os.path as osp

import torch
import torch.nn as nn

from widgets.muggled_sam.lib.make_sam import make_sam_from_state_dict

from widgets.muggled_sam.lib.demo_helpers.misc import get_default_device_string, make_device_config
# ---------------------------------------------------------------------------------------------------------------------
# %% Set up script args

# Set argparse defaults
default_device = get_default_device_string()
default_image_path_1 = None
default_image_path_2 = None
default_model_path = None
default_prompts_path = None
default_display_size = 900
default_base_size = 1024


class CrossMatch:
    def __init__(self, model_path="../../sam_v2/checkpoints/sam2_hiera_large.pt"):
        device_config_dict = make_device_config(default_device, use_float32=False)
        self.device = device_config_dict["device"]
        self.dtype = device_config_dict["dtype"]

        model_config_dict, model = make_sam_from_state_dict(model_path)
        self.model_name = osp.basename(model_path)

        self.sam = model
        self.sam.to(**device_config_dict)

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

    def prompt_encoder(self, box=[], fg_pts=[], bg_pts=[]):
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

