import os
import shutil
import argparse
import random
import timm
import wandb
import numpy as np
import pandas as pd
import albumentations as A
from PIL import Image
from sklearn.model_selection import train_test_split
from albumentations.pytorch import ToTensorV2

from arch.srnet.model import SRNet
from utils.datasets import StegDataset, BenchmarkDataset
from utils.utils import save_checkpoint

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)
    np.random.seed(args.seed + worker_id)

def train(train_loader, model, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        probs = torch.softmax(outputs.data, dim=1)
        _, predicted = torch.max(probs, 1)
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / len(train_loader.dataset)
    
    return avg_loss, accuracy

def validate(valid_loader, model, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            probs = torch.softmax(outputs.data, dim=1)
            _, predicted = torch.max(probs, 1)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(valid_loader)
    accuracy = 100 * correct / len(valid_loader.dataset)
    
    return avg_loss, accuracy

def main(args):
    if args.use_wandb:
        if args.run_name == 'auto':
            args.run_name = f'{args.backbone}_intra_{args.benchmark}_{args.stego_method}_{args.batch_size}_{args.lr}'
        wandb.run.name = args.run_name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.backbone == 'srnet':
        model = SRNet().to(device)
    else:
        model = timm.create_model(args.backbone, pretrained=args.load_pretrain, num_classes=2).to(device)
    
    data_path = os.path.join(args.data_root, args.benchmark)
    cover_paths = []
    stego_paths = []

    cover_dir = os.path.join(data_path, args.cover_size, 'cover')
    stego_dir = os.path.join(data_path, args.cover_size, args.stego_method)
    if os.path.exists(cover_dir):
        cover_paths.extend([os.path.join(cover_dir, img) for img in os.listdir(cover_dir)])
    if os.path.exists(stego_dir):
        stego_paths.extend([os.path.join(stego_dir, img) for img in os.listdir(stego_dir)])
    
    df = pd.DataFrame({
        'path': cover_paths + stego_paths,
        'label': [0]*len(cover_paths) + [1]*len(stego_paths)
    })

    train_df, valid_df = train_test_split(df, stratify=df['label'], test_size=1-args.train_rate, random_state=args.seed)
    train_dataset = BenchmarkDataset(train_df, transform=transform)
    valid_dataset = BenchmarkDataset(valid_df, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers_per_loader, shuffle=True, drop_last=True, worker_init_fn=worker_init_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.workers_per_loader, shuffle=False, drop_last=False, worker_init_fn=worker_init_fn)
    
    # Print the size of train, valid set
    print(f'Train Batch: {len(train_loader)}, Valid Batch: {len(valid_loader)}')
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_valid_acc = 0.0
    for epoch in range(1, args.epochs+1):
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, device)
        valid_loss, valid_acc = validate(valid_loader, model, criterion, device)
        print(f'Epoch [{epoch}/{args.epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}, Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.2f}')

        if args.use_wandb:
            wandb.log({'train_loss': train_loss, 'train_acc': train_acc, 'valid_loss': valid_loss, 'valid_acc': valid_acc}, step=epoch)
        is_best = valid_acc >= best_valid_acc
        if is_best:
            best_valid_acc = valid_acc
            if args.use_wandb:
                wandb.log({'best_valid_acc': best_valid_acc}, step=epoch)
                
            if args.save_model:
                save_checkpoint(
                    state={
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_valid_acc': best_valid_acc,
                    },
                    gpus=args.gpus,
                    is_best=is_best,
                    model_path=f'{args.ckpt_root}/{args.run_name}/',
                    model_name=f'ep{epoch}.pth.tar')
    wandb.finish()

if __name__ == '__main__':
    # SRNet Github
    # https://github.com/brijeshiitg/Pytorch-implementation-of-SRNet    

    # Argument parsing
    parser = argparse.ArgumentParser(description='2024 NSR Steganlysis Training Parser')

    # Model Parsers
    parser.add_argument('--backbone', type=str, default='srnet', help='Backbone name from timm library, if backbone is srnet, set it to srnet')
    parser.add_argument('--load_pretrain', type=bool, default=True, help='Whether to load pretrained model')
    parser.add_argument('--pretrained_path', type=str, default='./ckpt/', help='Path to pretrained model')
    parser.add_argument('--save_model', action='store_true', help='Whether to save the model')

    # Training Parsers
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--workers_per_loader', type=int, default=4, help='Number of workers per data loader')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='Weight decay for optimizer')
    parser.add_argument('--train_rate', type=float, default=0.7, help='Train split size')

    # MISC Parsers
    parser.add_argument('--data_root', type=str, default='../../data/2024_NSR_Steganalysis/', help='Path to the input data')
    parser.add_argument('--gpus', type=str, default='1', help='Comma-separated list of GPU IDs to use')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility') 
    parser.add_argument('--ckpt_root', type=str, default='./ckpt/', help='Root directory for checkpoint saving')
    parser.add_argument('--use_wandb', action='store_true', help='Whether to use wandb')
    parser.add_argument('--run_name', type=str, default='auto', help='Run name for checkpoint saving')

    parser.add_argument('--benchmark', type=str, default='BOSSbase', help='BOSSbase, BOWS2, ...')
    parser.add_argument('--cover_size', type=str, default='224', help='Cover size')
    parser.add_argument('--stego_method', type=str, default='S-UNIWARD_0.5', help='S-UNIWARD_0.5, LSB_0.5, WOW_0.5, ...')
    args = parser.parse_args()

    if args.gpus != '':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # A.Normalize(
        #     mean=[0.485, 0.456, 0.406],
        #     std=[0.229, 0.224, 0.225],
        # ),
        ToTensorV2(),
    ])

    if args.use_wandb:
        wandb.init(project='2024_NSR_Steganalysis', entity='kumdingso')
        wandb.save(f'./utils/datasets.py', policy='now')
        wandb.save(f'./train_benchmark.py', policy='now')
        for arg in vars(args):
            wandb.config[arg] = getattr(args, arg)
    main(args)

