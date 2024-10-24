import os
import shutil
import argparse
import random
import timm
import wandb
import time
from glob import glob
import numpy as np
import pandas as pd
import albumentations as A
from PIL import Image
from timeit import default_timer as timer
from albumentations.pytorch import ToTensorV2

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from arch.iformer.inception_transformer import iformer_small, iformer_base
from utils.datasets import StegDataset
from utils.utils import save_checkpoint, time_to_str

def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)
    np.random.seed(args.seed + worker_id)

def train(train_loader, model, criterion, optimizer, device, start_time):
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

def validate(valid_loader, model, criterion, device, start_time):
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
    start_time = time.time()
    if args.use_wandb:
        if args.run_name == 'auto':
            if args.stride is not None:
                args.run_name = f'{args.backbone}_stride_{args.stride}_intra_{args.mobile_device}_{args.stego_method}_{args.batch_size}_{args.lr}'
            else:
                args.run_name = f'{args.backbone}_intra_{args.mobile_device}_{args.stego_method}_{args.batch_size}_{args.lr}'
            if args.suffix != '':
                args.run_name += f'_{args.suffix}'
        wandb.run.name = args.run_name
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.backbone.split('-')[0] == 'timm':
        model = timm.create_model(args.backbone.split('-')[1], pretrained=True, num_classes=2)
    elif args.backbone == 'iformer-small':
        model = iformer_small(pretrained=True)
        model.head = nn.Linear(model.head.in_features, 2)
    elif args.backbone == 'iformer-base':
        model = iformer_base(pretrained=True)
        model.head = nn.Linear(model.head.in_features, 2)

    if args.stride is not None:
        if args.backbone.split('-')[1] == 'mobilevit_s':
            model.stem.conv.stride = (args.stride, args.stride)
        elif args.backbone.split('-')[1] == 'efficientnet_b0':
            model.conv_stem.stride = (args.stride, args.stride)
        elif args.backbone.split('-')[0] == 'iformer':
            model.patch_embed.proj1.conv = (args.stride , args.stride)

    if args.pretrained_path is not None:
        state_dict = torch.load(args.pretrained_path)['state_dict']
        model.load_state_dict(state_dict)
    
    model = model.to(device)

    train_df = pd.read_csv(f'{args.csv_root}/single/{args.data_type}/{args.cover_size}_{args.stego_method}/{args.mobile_device}_train.csv')
    valid_df = pd.read_csv(f'{args.csv_root}/single/{args.data_type}/{args.cover_size}_{args.stego_method}/{args.mobile_device}_valid.csv')
    
    if args.apply_ipp:
        ipp_train_df = pd.read_csv(f'{args.csv_root}/single/{args.data_type}/{args.cover_size}_ipp_random_{args.stego_method}/{args.mobile_device}_train.csv')
        train_df = pd.concat([train_df[:len(train_df)//2], ipp_train_df[:len(ipp_train_df)//2]], ignore_index=True)

    train_dataset = StegDataset(train_df, transform=train_transform)
    valid_dataset = StegDataset(valid_df, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers_per_loader, shuffle=True, drop_last=True, worker_init_fn=worker_init_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.workers_per_loader, shuffle=False, drop_last=False, worker_init_fn=worker_init_fn)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Print data num of each loader
    if args.stride is not None:
        print(f'Model: {args.backbone} with First conv stride({args.stride}, {args.stride})')
    else:
        print(f'Model: {args.backbone}')
        
    start_time = timer()      
    print(f'Train Data: {len(train_loader.dataset)}')
    print(f'Valid Data: {len(valid_loader.dataset)}')
    print(f'=================================== {args.mobile_device} intra training ===================================')
    best_valid_acc = 0.0
    for epoch in range(1, args.epochs+1):
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, device, start_time)
        valid_loss, valid_acc = validate(valid_loader, model, criterion, device, start_time)
        print(f'Epoch [{epoch}/{args.epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}, Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.2f} | {time_to_str(timer()-start_time, mode="min")}')

        if args.use_wandb:
            wandb.log({'train_loss': train_loss, 'train_acc': train_acc, 'valid_loss': valid_loss, 'valid_acc': valid_acc}, step=epoch)
        is_best = valid_acc >= best_valid_acc
        if is_best:
            best_epoch = epoch
            best_train_acc = train_acc
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
    print(f'Best Epoch Acc[{best_epoch}] - Train Acc: {best_train_acc:.2f}, Valid Acc: {best_valid_acc:.2f} | {time_to_str(timer()-start_time, mode="min")}')
    wandb.finish()

if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser(description='2024 NSR Steganlysis Training Parser')
    # Model Parsers
    parser.add_argument('--backbone', type=str, default='timm-efficientnet_b0', help='Backbone name from timm library')
    parser.add_argument('--pretrained_path', type=str, default=None, help='Path to pretrained model')
    parser.add_argument('--stride', type=int, default=None, help='Stride of the backbone')
    parser.add_argument('--save_model', action='store_true', help='Whether to save the model')

    # Training Parsers
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--workers_per_loader', type=int, default=4, help='Number of workers per data loader')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='Weight decay for optimizer')
    parser.add_argument('--train_rate', type=float, default=0.7, help='Train split size')

    # MISC Parsers
    parser.add_argument('--csv_root', type=str, default='./csv/', help='Path to the csv')
    parser.add_argument('--gpus', type=str, default='0', help='Comma-separated list of GPU IDs to use')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility') 
    parser.add_argument('--ckpt_root', type=str, default='./ckpt/', help='Root directory for checkpoint saving')
    parser.add_argument('--use_wandb', action='store_true', help='Whether to use wandb')
    parser.add_argument('--run_name', type=str, default='auto', help='Run name for checkpoint saving')
    parser.add_argument('--suffix', type=str, default='', help='Suffix for run name')

    parser.add_argument('--mobile_device', type=str, default='Galaxy_Flip3',
                        choices=['Galaxy_S10+','Galaxy_Flip3', 'Galaxy_Flip4', 'Galaxy_Fold3', 'Galaxy_Fold4', 'Galaxy_Note9',
                                'Galaxy_S20+', 'Galaxy_S20FE', 'Galaxy_S21', 'Galaxy_S21_Ultra', 'Galaxy_S22', 
                                'iPhone12_ProMax', 'Huawei_P30', 'LG_Wing'])
    parser.add_argument('--data_type', type=str, default='PNG', choices=['PNG', 'JPEG'])
    parser.add_argument('--cover_size', type=str, default='224')
    parser.add_argument('--apply_ipp', action='store_true', help='Whether to apply IPP')
    parser.add_argument('--stego_method', type=str, default='LSB_0.5')
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

    train_transform = A.Compose([
        A.CenterCrop(height=224, width=224),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])
    test_transform = A.Compose([
        A.CenterCrop(height=224, width=224),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])

    if args.use_wandb:
        wandb.init(project='2024_NSR_Steganalysis', entity='kumdingso')
        wandb.save(f'./utils/datasets.py', policy='now')
        wandb.save(f'./train.py', policy='now')
        for arg in vars(args):
            wandb.config[arg] = getattr(args, arg)
    main(args)

