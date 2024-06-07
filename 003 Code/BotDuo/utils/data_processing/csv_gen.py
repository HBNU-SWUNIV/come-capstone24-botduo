import os
import csv
import random
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split

def collect_all_files(base_dir, label):
    all_files = []
    for video_folder in os.listdir(base_dir):
        video_folder_path = os.path.join(base_dir, video_folder)
        
        frames = glob(os.path.join(video_folder_path, '*.png'))
        for frame in frames:
            basename = os.path.basename(frame).split('.')[0]

            # if int(basename) % 5 == 0:
            #     all_files.append((frame, label))
            all_files.append((frame, label))

    return all_files

def save_to_csv(file_list, output_csv):
    df = pd.DataFrame(file_list, columns=['img_path', 'label'])
    df.to_csv(output_csv, index=False)

def split_data(live_root, spoof_root, save_root):
    # 0 to Live
    # 1 to Spoof
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    
    live_files = collect_all_files(live_root, 0)
    live_train, live_test = train_test_split(live_files, test_size=0.3, random_state=42)
    
    save_to_csv(live_train, f'{save_root}/train_live.csv')
    save_to_csv(live_test, f'{save_root}/test_live.csv')
    
    for attack_type in os.listdir(spoof_root):
        attack_dir = os.path.join(spoof_root, attack_type)
        if os.path.isdir(attack_dir):
            attack_files = collect_all_files(attack_dir, 1)
            attack_train, attack_test = train_test_split(attack_files, test_size=0.3, random_state=42)
            save_to_csv(attack_train, f'{save_root}/train_spoof_{attack_type}.csv')
            save_to_csv(attack_test, f'{save_root}/test_spoof_{attack_type}.csv')

if __name__ == '__main__':
    live_root = '/workspace/DS/Datas/FAS/SiW-Mv2/official/live/'
    spoof_root = '/workspace/DS/Datas/FAS/SiW-Mv2/official/spoof/'
    save_root = './csv/all_frame/'
    
    split_data(live_root, spoof_root, save_root)