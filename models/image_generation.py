# Modified from: https://github.com/facebookresearch/dino/blob/main/video_generation.py
# Copyright (c) Facebook, Inc. and its affiliates.
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
import os
import glob
import sys
import argparse
import cv2

from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from PIL import Image
from models import DenoisingViT
# from utils.visualization_tools import (
#     get_pca_map,
#     get_robust_pca,
# )
import imageio
# from torch_kmeans import KMeans, CosineSimilarity

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class ImageGenerator:
    def __init__(self):
        # self.args = args
        self.model = self.load_model()

    def run(self):


        with torch.no_grad():

            self.inference(ref_img_paths)

            # self.generate_video_from_images()





    def inference(self, img):
        # cache_to_compute_pca_raw = []
        # cache_to_compute_pca_denoised = []
        # interested_files = inp

        # img = Image.open(interested_files)
        # img = img.convert("RGB")

        # img = self.transform(img)
        # # print(img.shape)
        # # make the image divisible by the patch size
        # w, h = (
        #     img.shape[1] - img.shape[1] % 7,
        #     img.shape[2] - img.shape[2] % 7,
        # )
        # img = img[:, :w, :h].unsqueeze(0)
        # # print(img.shape)
        
        output_dict = self.model(img, return_dict=True)
        raw_vit_feats = output_dict["raw_vit_feats"]
        raw_vit_feats = raw_vit_feats.reshape(1, -1, raw_vit_feats.shape[-1])
        random_patches = torch.randperm(raw_vit_feats.shape[1])  # [:100]
        raw_vit_feats = raw_vit_feats[:, random_patches, :]
        # kmeans_raw.fit(raw_vit_feats)
        # cache_to_compute_pca_raw.append(raw_vit_feats)
        denoised_features = output_dict["pred_denoised_feats"]
        # print(denoised_features.shape)
        return denoised_features
    def load_model(self):
        # build model
        vit = DenoisingViT.ViTWrapper(
            model_type='vit_base_patch14_reg4_dinov2.lvd142m',
            stride=7*2,
        )
        vit = vit.to(DEVICE)
        model = DenoisingViT.Denoiser(
            noise_map_height=37,
            noise_map_width=37,
            feature_dim=vit.n_output_dims,
            vit=vit,
            enable_pe=True,
        ).to(DEVICE)
        # if args.load_denoiser_from is not None:
        freevit_model_ckpt = torch.load('/home/ubuntuu/hdd/whl/RHNet/models/DenoisingViT/reg4_v1.pth')["denoiser"]
        msg = model.load_state_dict(freevit_model_ckpt, strict=False)
        for k in model.state_dict().keys():
            if k in freevit_model_ckpt:
                print(k, "loaded")
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        model.to(DEVICE)
        normalizer = vit.transformation.transforms[-1]
        # print('normalizer:',normalizer)
        # normalizer = Normalize(mean=tensor([0.4850, 0.4560, 0.4060]), std=tensor([0.2290, 0.2240, 0.2250]))
        self.transform = transforms.Compose([transforms.ToTensor(), normalizer])
        return model


def parse_args():
    parser = argparse.ArgumentParser("Generation self-attention video")
    parser.add_argument(
        "--enable_pe",
        action="store_true",
    )
    parser.add_argument(
        "--vit_type",
        default="vit_base_patch14_reg4_dinov2.lvd142m",
        type=str,
    )
    parser.add_argument(
        "--vit_stride", default=7, type=int, help="Patch resolution of the self.model."
    )
    parser.add_argument(
        "--n_clusters", default=20, type=int, help="Patch resolution of the self.model."
    )
    parser.add_argument(
        "--subsample_ratio", default=1, type=int, help="stride to sample from the video"
    )
    parser.add_argument("--noise_map_height", default=37, type=int)
    parser.add_argument("--noise_map_width", default=37, type=int)
    parser.add_argument("--load_denoiser_from", default='/home/oem/Tasks/Denoising-ViT-main/checkpoints_DiT/reg4_v1.pth', type=str)
    parser.add_argument("--input_path", default='./inputs/',required=True, type=str)
    parser.add_argument("--output_path", default="./", type=str)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="""We visualize masks
        obtained by thresholding the self-attention maps to keep xx percent of the mass.""",
    )
    parser.add_argument(
        "--resize",
        default=224,
        type=int,
        nargs="+",
        help="""Apply a resize transformation to input image(s). Use if OOM error.
        Usage (single or H W): --resize 512, --resize 720 1280""",
    )
    # parser.add_argument(
    #     "--fps",
    #     default=30.0,
    #     type=float,
    #     help="FPS of input / output video. Automatically set if you extract frames from a video.",
    # )
    # parser.add_argument(
    #     "--video_format",
    #     default="mp4",
    #     type=str,
    #     choices=["mp4", "avi"],
    #     help="Format of generated video (mp4 or avi).",
    # )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    vg = ImageGenerator(args)
    vg.run()
