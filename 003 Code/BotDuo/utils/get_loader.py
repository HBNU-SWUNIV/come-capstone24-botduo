import os
import random
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from utils.dataset import CapstoneDataset
from torch.utils.data import DataLoader

def worker_init_fn(worker_id):
    np.random.seed(42 + worker_id)
    random.seed(42 + worker_id)

def load_csv_data(file_path):
    return pd.read_csv(file_path)

def get_dataset(args):
    real_train_data = load_csv_data(f'{args.csv_path}/train_live.csv')
    real_test_data = load_csv_data(f'{args.csv_path}/test_live.csv')
    
    spoof_train_data = {}
    for csv_file in os.listdir(args.csv_path):
        if 'spoof' in csv_file and args.test_attack_type not in csv_file:
            attack_type = csv_file.split('spoof_')[1].split('.csv')[0]
            spoof_train_data[attack_type] = load_csv_data(f'{args.csv_path}/{csv_file}')
    
    spoof_test_data = load_csv_data(f'{args.csv_path}/test_spoof_{args.test_attack_type}.csv')

    train_dataloader_real = DataLoader(CapstoneDataset(real_train_data, train=True), batch_size=args.batch_size_per_loader, shuffle=True, num_workers=4, worker_init_fn=worker_init_fn)
    train_dataloader_spoof = {attack_type: DataLoader(CapstoneDataset(data, train=True), batch_size=args.batch_size_per_loader, shuffle=True, num_workers=4, worker_init_fn=worker_init_fn) for attack_type, data in spoof_train_data.items()}
    test_dataloader_real = DataLoader(CapstoneDataset(real_test_data, train=False), batch_size=args.batch_size_per_loader, shuffle=False, num_workers=4, worker_init_fn=worker_init_fn)
    test_dataloader_spoof = DataLoader(CapstoneDataset(spoof_test_data, train=False), batch_size=args.batch_size_per_loader, shuffle=False, num_workers=4, worker_init_fn=worker_init_fn)

    return train_dataloader_real, train_dataloader_spoof, test_dataloader_real, test_dataloader_spoof

