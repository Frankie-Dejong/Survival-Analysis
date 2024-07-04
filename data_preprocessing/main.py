import os
import re
import yaml
import json
import numpy as np
from tqdm import tqdm
import torch
import sys
sys.path.append('./')
from pretrain.encdec import EncDec
import openslide
import torch.nn as nn

from tiling import *
from normalization import *
from extraction import load_tsv
import pandas as pd

def get_model(hidden_channels=32, path='YOUR PATH'):
    model = EncDec(hidden_channels=hidden_channels)
    state_dict = torch.load(path)
    model.load_state_dict(state_dict=state_dict, strict=True)
    model.eval()
    model.to('cuda')
    return model.enc

def pre_process(images, mean, std):
    bs, c, h, w = images.shape    
    assert c == 3 and h == 256 and w == 256
    images = (images - mean.reshape(1, -1, 1, 1)) / std.reshape(1, -1, 1, 1)
    return torch.Tensor(images)

def get_data(cfg):
    COLD_df = load_tsv(cfg["TCGA_COAD_survival_info_path"])
    READ_df = load_tsv(cfg["TCGA_READ_survival_info_path"])
    tsv_df = pd.concat([COLD_df, READ_df])
    tsv_df = tsv_df.reset_index(drop=True)
    samples_data = tsv_df.set_index("sample").to_dict(orient='index')
    return samples_data

def main(
    feat_save_path,
    cfg_path,
    io : int=240,
    alpha : int=1, 
    beta : float=0.15,
    filter_size : int=(1 << 14),
    norm_only=False
    ):
    if not norm_only:
        model = get_model()
        block = nn.Sequential(
            model,
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten()
        )
        mean = np.array([180.87767965, 147.45242127, 172.7674556])
        std = np.array([47.7491585, 61.73678167, 46.65056828])
    
    cfg = yaml.safe_load(open(cfg_path, "r"))
    sample_pattern = r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-[A-Z0-9]{3}"
    samples_data = get_data(cfg)

    format_info = json.load(open(cfg["format_info_path"], "r"))
    import random
    for info in tqdm(format_info, desc="Processing data"):

        match = re.search(sample_pattern, info["file_name"])
        if match:
            sample_id = match.group()
        else:
            raise ValueError("Sample ID not found")
        
        if sample_id not in samples_data.keys():
            tqdm.write(f"Warning: {sample_id} not found in tsv files, but exists in {cfg['format_info_path']}")
            continue

        sample_data = samples_data[sample_id]

        #--> load image
        svs_image = os.path.join(cfg["original_data_path"] , info["file_id"], info["file_name"])    
        resolution_level: int=1
        overlapping_percentage: int=0
        window_size: int=256
        img = openslide.OpenSlide(svs_image)
        width, height = img.level_dimensions[resolution_level] 

        #--> start processing
        x_start_positions = get_start_positions(width, height, window_size, Axis.X, overlapping_percentage)
        y_start_positions = get_start_positions(width, height, window_size, Axis.Y, overlapping_percentage)

        patches = []
        norms = []
        features = []

        for x_index, x_start_position in enumerate(x_start_positions):
            for y_index, y_start_position in enumerate(y_start_positions):

                #--> tile patches
                x_end_position = min(width, x_start_position + window_size)
                y_end_position = min(height, y_start_position + window_size)
                patch_width = x_end_position - x_start_position
                patch_height = y_end_position - y_start_position
                
                if patch_width != window_size or patch_height != window_size:
                    continue

                SVS_level_ratio = get_SVS_level_ratio(resolution_level)
                patch = img.read_region((x_start_position * SVS_level_ratio, y_start_position * SVS_level_ratio),
                                        resolution_level,
                                        (patch_width, patch_height))
                patch.load()
                patch_rgb = Image.new("RGB", patch.size, (255, 255, 255))
                patch_rgb.paste(patch, mask=patch.split()[3])

                patch_rgb = np.array(patch_rgb)
                if norm_only:
                    #--> norm and save
                    norm = normalize_staining(patch_rgb, "", None, io, alpha, beta, filter_size)
                    if norm is None:
                        continue
                    norms.append(norm)
                else:
                    #--> extract features per 1 patches
                    patches.append(patch_rgb)
                    if len(patches) == 1:
                        for i in patches:
                            norm = normalize_staining(i, "", None, io, alpha, beta, filter_size)
                            if norm is None:
                                continue
                            norms.append(norm)
                        if len(norms) == 0:
                            patches = []
                            norms = []
                            continue
                        
                        norms = np.array(norms)
                        norms = np.transpose(norms, (0, 3, 1, 2))
                        images = pre_process(norms, mean, std)
                        feature = np.array(block(images.to('cuda')).detach().cpu())
                        assert feature.shape[1] == 2048
                        features.append(feature)
                        patches = []
                        norms = []
        
        
        if norm_only:
            data = {
                "bcr_patient_barcode": sample_data["_PATIENT"],
                "vital_status": sample_data["OS"],
                "days": sample_data["OS.time"],
                "norms": np.array(norms),
            }
            np.save(os.path.join('/media/kevin/DATA2/cancer_prediction_frb/dataset/TCGA/normed_lowres', sample_id), data)
            tqdm.write(f"Save to {sample_id}.npy, total {len(norms)} patches")
        else:
            if len(features) == 0:
                tqdm.write(f"Fail To Save {sample_id}")
                continue
            data = {
                "bcr_patient_barcode": sample_data["_PATIENT"],
                "vital_status": sample_data["OS"],
                "days": sample_data["OS.time"],
                "features": np.concatenate(features, axis=0),
            }
            np.save(os.path.join(feat_save_path, sample_id), data)
            tqdm.write(f"Save to {feat_save_path}/{sample_id}.npy, total {len(features)} features")
        
        
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--feat_save_path', type=str, default='../dataset/TCGA/features_lowres')
    parser.add_argument('--cfg_path', type=str, default='configuration/tcga_cfg.yml')
    parser.add_argument('--io', type=int, default=240)
    parser.add_argument('--alpha', type=int, default=1)
    parser.add_argument('--beta', type=float, default=0.15)
    parser.add_argument('--filter_size', type=int, default=(1 << 14))
    parser.add_argument('--norm_only', action='store_true')
    args = parser.parse_args()
    main(
        feat_save_path=args.feat_save_path,
        cfg_path=args.cfg_path,
        io=args.io,
        alpha=args.alpha,
        beta=args.beta,
        filter_size=args.filter_size,
        norm_only=args.norm_only
    )
