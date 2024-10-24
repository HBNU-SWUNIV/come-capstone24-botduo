import os
import argparse
import zipfile
from tqdm import tqdm

def compress(args):
    os.makedirs('./compressed/', exist_ok=True)
    root_dir = args.root_dir + args.data_type + '/'
    for device in tqdm(args.devices, desc='Total Progress', total=len(args.devices)):
        with zipfile.ZipFile(f'./utils/compressed/{device}.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
            for target_dir in args.target_dir:
                # 압축을 원하는 최상위 폴더 경로 설정
                device_folder = os.path.join(root_dir, device, target_dir)
                for base_path, subfolders, filenames in os.walk(device_folder):
                    base_path = base_path.replace('\\', '/')
                    # 압축을 원하는 최하위 폴더명 입력
                    if base_path.split('/')[-1] == '224_J_UERD_0.5':
                        for filename in filenames:
                            file_path = os.path.join(base_path, filename)
                            zipf.write(file_path, os.path.relpath(file_path, root_dir))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='/workspace/data/2024_NSR_Steganalysis/')
    parser.add_argument('--data_type', type=str, default='JPEG', choices=['PNG', 'JPEG'])
    parser.add_argument('--target_dir', type=list, nargs='+', default=['cover','stego'])
    parser.add_argument('--devices', type=list, nargs='+', default=['Galaxy_Flip3', 'Galaxy_S20+', 'iPhone12_ProMax', 'Huawei_P30', 'LG_Wing'])
    args = parser.parse_args()

    # args.devices = ['Galaxy_Flip3', 'Galaxy_S20+', 'iPhone12_ProMax', 'Huawei_P30', 'LG_Wing',\
    #                 'Galaxy_Flip4', 'Galaxy_Fold3', 'Galaxy_Fold4', 'Galaxy_Note9', 'Galaxy_S10+', 'Galaxy_S20FE', 'Galaxy_S21', 'Galaxy_S21_Ultra', 'Galaxy_S22']
    # args.devices = ['Galaxy_Flip3', 'Galaxy_S20+', 'iPhone12_ProMax', 'Huawei_P30', 'LG_Wing']
    args.devices = ['Galaxy_Flip3', 'Galaxy_S20+', 'iPhone12', 'iPhone13_Mini', 'Huawei_P30', 'LG_Wing']
    # args.devices = ['iPhone12_ProMax']

    compress(args)