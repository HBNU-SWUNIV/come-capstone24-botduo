import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import timm
import wandb
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from inception_transformer import iformer_small

def worker_init_fn(worker_id):
    np.random.seed(args.seed + worker_id)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  
    
def preprocess_image(image_path):
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])
    image = Image.open(image_path).convert("RGB")
    image = np.array(image)
    image = transform(image=image)["image"]
    return image.unsqueeze(0)  # 배치 차원 추가

def inference(image_path, model, device): 
    model.eval()
    image = preprocess_image(image_path).to(device)
    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()
        predicted_class = np.argmax(probs)
    return probs, predicted_class

def main(args):
    
    set_seed(args.seed)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # model = timm.create_model(args.backbone, pretrained=False, num_classes=2).to(device)
    model = iformer_small(pretrained=True, num_classes=2).to(device)
    checkpoint = torch.load(args.pretrained_path)
    model.load_state_dict(checkpoint['state_dict'])

    probs, predicted_class = inference(args.image_path, model, device)
    
    print(f"추론 결과: Cover = {probs[0] * 100:.2f}%, Stego = {probs[1] * 100:.2f}%")
    print(f"예측된 클래스: {'Cover' if predicted_class == 0 else 'Stego'}")

# Argument parsing
parser = argparse.ArgumentParser(description='Validation Parser')
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train')
parser.add_argument('--backbone', type=str, default='iformer_small', help='Backbone name from timm library')
parser.add_argument('--pretrained_path', type=str, default='./ckpt/best_model.pth', help='Path to pretrained model')
parser.add_argument('--image_path', type=str, default='../images')
parser.add_argument('--gpus', type=str, default='0', help='GPU IDs to use')
parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
parser.add_argument('--workers', type=int, default=8, help='Number of workers for data loading')
args = parser.parse_args()

if args.gpus != '':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


main(args)
