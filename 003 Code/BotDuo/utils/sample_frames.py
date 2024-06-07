import pandas as pd
import numpy as np

from glob import glob

def sample_frames(attack_type, mode, flag, num_frames):
    img_path = f'/workspace/Datas/SiW-Mv2/official/{mode}/{attack_type}/'
    image_list = glob(f'{img_path}/*.png')
    
    
    df = pd.read_csv(txt_path, delimiter=' ', header=None, names=['photo_path', 'photo_label'])
    df['video_id'] = df['photo_path'].apply(lambda x: '_'.join(x.split('/')[-1].split('_')[:-2]))
    sampled_df = pd.DataFrame()
    
    for video_id, group in df.groupby('video_id'):
        sampled_frames = group if len(group) < num_frames else group.sample(n=num_frames, random_state=42)
        if flag == 0:
            sampled_df = pd.concat([sampled_df, sampled_frames[sampled_frames['photo_label'] == 0]])
        elif flag == 1:
            sampled_df = pd.concat([sampled_df, sampled_frames[sampled_frames['photo_label'] != 0]])
        else:
            sampled_df = pd.concat([sampled_df, sampled_frames])
    
    sampled_df = sampled_df.reset_index(drop=True)
    return sampled_df