import os
import numpy as np
import pandas as pd
import random
import argparse
from sklearn.model_selection import train_test_split
from glob import glob
from tqdm import tqdm

def single_device_csv(args):
    # device_list = ['Galaxy_Flip3', 'Galaxy_Flip4', 'Galaxy_Fold3', 'Galaxy_Fold4', 'Galaxy_Note9',
    #                'Galaxy_S10+', 'Galaxy_S20+', 'Galaxy_S20FE', 'Galaxy_S21', 'Galaxy_S21_Ultra',
    #                'Galaxy_S22', 'iPhone12_ProMax', 'Huawei_P30', 'LG_Wing']
    # device_list = ['Galaxy_Flip4', 'Galaxy_Fold3', 'Galaxy_Fold4', 'Galaxy_Quantum3',
    #                'Galaxy_S10+', 'Galaxy_S20+', 'Galaxy_S20FE', 'Galaxy_S21', 'Galaxy_S21', 'Galaxy_S22', 'Galaxy_S22_Ultra',\
    #                'iPhone7', 'iPhone11_Pro', 'iPhone12_ProMax']
    # device_list = ['Galaxy_Flip3', 'Galaxy_S20+', 'iPhone12', 'iPhone13_Mini', 'Huawei_P30', 'LG_Wing']
    device_list = ['Galaxy_Flip3', 'Galaxy_S20+', 'iPhone12_ProMax', 'Huawei_P30', 'LG_Wing']
    # device_list = ['Naver', 'Instagram']
    for device in device_list:
        if args.data_type == 'JPG':
            os.makedirs(f'{args.csv_root}/single/JPEG/{args.cover_size}_{args.stego_method}', exist_ok=True)
            data_path = os.path.join(args.data_root, 'JPEG', device)
        else:
            os.makedirs(f'{args.csv_root}/single/PNG/{args.cover_size}_{args.stego_method}', exist_ok=True)
            data_path = os.path.join(args.data_root, 'PNG', device)
        
        cover_paths = glob(os.path.join(data_path, 'cover', f'{args.cover_size}_ipp',  f'*.{args.data_type.lower()}'))
        stego_paths = glob(os.path.join(data_path, 'stego', f'{args.cover_size}_ipp_{args.stego_method}', f'*.{args.data_type.lower()}'))
        cover_paths += glob(os.path.join(data_path, 'cover', f'{args.cover_size}_ipp', f'*.{args.data_type.upper()}'))
        stego_paths += glob(os.path.join(data_path, 'stego', f'{args.cover_size}_ipp_{args.stego_method}', f'*.{args.data_type.upper()}'))

        df = pd.DataFrame({
            'path': cover_paths + stego_paths,
            'label': [0]*len(cover_paths) + [1]*len(stego_paths)
        })

        train_size = int(len(df) * args.train_rate)
        valid_size = len(df) - train_size

        try:
            train_df, valid_df = train_test_split(df, stratify=df['label'], train_size=train_size, test_size=valid_size, random_state=args.seed)
        except:
            print(f'{device} Error')
            print(len(cover_paths), len(stego_paths))
            continue
        
        if args.data_type == 'JPG':
            os.makedirs(f'{args.csv_root}/single/JPEG/{args.cover_size}_{args.stego_method}/', exist_ok=True)
        else:
            os.makedirs(f'{args.csv_root}/single/PNG/{args.cover_size}_ipp_{args.stego_method}/', exist_ok=True)
        
        if args.data_type == 'JPG':
            train_df.to_csv(f'{args.csv_root}/single/JPEG/{args.cover_size}_{args.stego_method}/{device}_train.csv', index=False)
            valid_df.to_csv(f'{args.csv_root}/single/JPEG/{args.cover_size}_{args.stego_method}/{device}_valid.csv', index=False)
        else:
            train_df.to_csv(f'{args.csv_root}/single/PNG/{args.cover_size}_ipp_{args.stego_method}/{device}_train.csv', index=False)
            valid_df.to_csv(f'{args.csv_root}/single/PNG/{args.cover_size}_ipp_{args.stego_method}/{device}_valid.csv', index=False)

if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser(description='2024 NSR Steganlysis Training Parser')
    parser.add_argument('--train_rate', type=float, default=0.7, help='Train split size')

    # MISC Parsers
    parser.add_argument('--data_root', type=str, default='../../data/2024_NSR_Steganalysis/', help='Path to the input data')
    parser.add_argument('--csv_root', type=str, default='./csv/', help='Path to the output csv')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility') 

    parser.add_argument('--data_type', type=str, default='PNG', choices=['PNG', 'JPG'])
    parser.add_argument('--cover_size', type=str, default='224')
    parser.add_argument('--stego_method', type=str, default='LSB_0.1')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    
    single_device_csv(args)
    # intergrated_csv(args, total_data_num=200000)
    # intergrated_except_csv(args)
    