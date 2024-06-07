import os
import argparse
import numpy as np
import pandas as pd
import random
import umap
import plotly.express as px
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from arch.DiViT import DiViT

def main(args, device):
    model = DiViT(args.pretrained_model, args.num_classes, args.input_size).to(device)
    model.eval()

    mic_features = []
    mic_labels = []

    with torch.no_grad():
        for images, labels in mic_dataloader:
            images = images.to(device)
            _, features = model(images)
            mic_features.append(features.cpu().numpy())
            mic_labels.extend(labels.cpu().numpy())

    mic_features = np.concatenate(mic_features, axis=0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    # MISC Settings
    parser.add_argument('--ckpt_root', type=str, default='./ckpt/')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    if args.gpu != '':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    main(config, args, device)