import sys
# sys.path.append('../../')
# sys.path.append('./')
 
import os
import argparse
import wandb
import time
import timm
import random
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from utils.utils import save_checkpoint, AverageMeter, Logger, OvRScore, time_to_str
from utils.get_loader import get_dataset
from timeit import default_timer as timer

def train(args, device):
    if args.use_wandb:
        wandb.init(project='2024_CapstoneDesign', entity='kumdingso')

        if args.run_name != '':
            wandb.run.name = f'{args.run_name}_to_{args.test_attack_type}'
        else:
            wandb.run.name = f'Capstone_Design_Train_to_{args.test_attack_type}'
    else:
        pass

    os.makedirs(args.logs, exist_ok=True)
    
    train_dataloader_real, \
    train_dataloader_spoof, \
    test_dataloader_real, \
    test_dataloader_spoof = get_dataset(args)

    # Accessing individual spoof dataloaders
    print('-------------Training Dataset Info -------------')
    print(f"Live Dataset: {len(train_dataloader_real.dataset)}")
    for attack_type, dataloader in train_dataloader_spoof.items():
        print(f"{attack_type} Dataset: {len(dataloader.dataset)}")

    print('-------------Test Dataset Info -------------')
    print(f"Live Dataset: {len(test_dataloader_real.dataset)}")
    print(f'Spoof Dataset: {len(test_dataloader_spoof.dataset)}')
    
    train_total_loss = AverageMeter()
    
    log = Logger()
    # log.open(args.logs + args.test_attack_type + '_log.txt', mode='w')
    log.open(args.logs + f'[{datetime.now().strftime("%Y%m%d_%H%M%S")}] ' + args.test_attack_type + '_log.txt', mode='w')
    log.write("\n----------------------------------------------- [START %s] %s\n\n" % (
    datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 51))
    log.write(
        '--------------|------------- TRAIN -------------|-------------- TEST -------------|-------- BEST -------|--------------|\n')
    log.write(
        '     ITER     |     Loss      HTER      AUC     |     Loss      HTER      AUC     |    ITER      HTER   |     time     |\n')
    log.write(
        '-----------------------------------------------------------------------------------------------------------------------|\n')
    start = timer()
    

    # net = DiViT(encoder=encoder, num_classes=2).to(device)
    net = timm.create_model('mobilevit_s', pretrained=True, num_classes=2).to(device)
    
    criterion = {
        'CE': nn.CrossEntropyLoss().cuda(),
    }
    optimizer_dict = [
        {"params": filter(lambda p: p.requires_grad, net.parameters()), "lr": args.init_lr},
    ]
    optimizer = optim.AdamW(optimizer_dict, lr=args.init_lr, weight_decay=args.weight_decay)

    if(len(args.gpu) > 1):
        net = torch.nn.DataParallel(net).cuda()

    best_hter = 1.0
    best_iter = 0
    
    max_iter = args.max_iter
    epoch = 1
    iter_per_epoch = 10
    
    real_iter = iter(train_dataloader_real)
    real_iter_per_epoch = len(real_iter)
    spoof_iters = {attack_type: iter(dataloader) for attack_type, dataloader in train_dataloader_spoof.items()}
    spoof_iters_per_epoch = {attack_type: len(dataloader) for attack_type, dataloader in train_dataloader_spoof.items()}

    for iter_num in range(max_iter):
        if (iter_num % real_iter_per_epoch == 0):
            real_iter = iter(train_dataloader_real)
        for attack_type in spoof_iters:
            if (iter_num % spoof_iters_per_epoch[attack_type] == 0):
                spoof_iters[attack_type] = iter(train_dataloader_spoof[attack_type])
        if (iter_num != 0 and iter_num % iter_per_epoch == 0):
            epoch = epoch + 1
            
        param_lr_tmp = []
        for param_group in optimizer.param_groups:
            param_lr_tmp.append(param_group["lr"])

        # Training Phase
        net.train(True)
        optimizer.zero_grad()
        
        real_img, real_label = next(real_iter)
        real_img = real_img.cuda()
        real_label = real_label.cuda()
        
        spoof_imgs = []
        spoof_labels = []
        for attack_type in spoof_iters:
            img, label = next(spoof_iters[attack_type])
            spoof_imgs.append(img.cuda())
            spoof_labels.append(label.cuda())
        
        input_data = torch.cat([real_img] + spoof_imgs, dim=0)
        source_label = torch.cat([real_label] + spoof_labels, dim=0)
        
        ######### forward #########
        outputs = net(input_data)
        
        ######### loss #########
        loss = criterion["CE"](outputs, source_label)
        train_loss = loss

        train_total_loss.update(train_loss.item())
        
        ######### backward #########
        train_loss.backward()
        optimizer.step()

        ######### Evaluation #########
        probs = []
        train_probs = []
        train_labels = []
        probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()
        train_probs.extend(probs)
        train_labels.extend(source_label.detach().cpu().numpy())

        train_probs = np.array(train_probs)
        train_labels = np.array(train_labels)
        
        train_auc, train_hter = OvRScore(train_probs, train_labels)

        # Test Phase
        net.eval()

        test_probs = []
        test_labels = []
        test_total_loss = 0.0
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(test_dataloader_real):
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = net(inputs)
                
                loss = criterion['CE'](outputs, labels)
                test_total_loss += loss.item()
                
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                test_probs.extend(probs)
                test_labels.extend(labels.detach().cpu().numpy())

            for batch_idx, (inputs, labels) in enumerate(test_dataloader_spoof):
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = net(inputs)
                
                loss = criterion['CE'](outputs, labels)
                test_total_loss += loss.item()
                
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                test_probs.extend(probs)
                test_labels.extend(labels.detach().cpu().numpy())

        test_total_loss /= (len(test_dataloader_real) + len(test_dataloader_spoof))
        test_probs = np.array(test_probs)
        test_labels = np.array(test_labels)
        
        test_auc, test_hter = OvRScore(test_probs, test_labels)

        is_best= best_hter >= test_hter
        if is_best:
            best_hter = test_hter
            best_iter = iter_num

        log.write('     %4i     |   %7.3f    %6.2f    %6.2f   |   %7.3f    %6.2f    %6.2f   |    %4i     %6.2f  | %s |' % (
            iter_num,
            train_total_loss.avg,
            train_hter*100,
            train_auc*100,
            test_total_loss,
            test_hter*100,
            test_auc*100,
            best_iter,
            best_hter*100,
            time_to_str(timer() - start, 'min')))
        log.write('\n') 
        
        if args.save_models:
            save_checkpoint(state={
                'iter': iter_num,
                'state_dict': net.state_dict(),
                'best_hter': best_hter
                }, gpus=args.gpu, is_best=is_best,
                model_path=args.ckpt_root + f'/{args.run_name}_to_{args.test_attack_type}/',
                model_name=f'iter{iter_num}.pth.tar')
        if args.use_wandb:
            wandb.log({
                "train_loss": train_total_loss.avg,
                "train_hter": train_hter,
                "train_auc": train_auc,
                "test_loss": test_total_loss,
                'test_hter': test_hter,
                'test_auc': test_auc,
                'lr':optimizer.param_groups[0]["lr"]},
                step=iter_num)
        
        time.sleep(0.01)
    print('Best HTER: %6.2f' % (best_hter*100))
    if args.use_wandb:
        wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model Settings
    parser.add_argument('--encoder', type=str, default='mobilevit_s', help='Backbone Name from timm library')

    # Training Settings
    parser.add_argument('--max_iter', type=int, default=100)
    parser.add_argument('--batch_size_per_loader', type=int, default=8)
    parser.add_argument('--init_lr', type=float, default=1e-4)
    parser.add_argument('--warmup_epochs', type=int, default=0, metavar='N', help='epochs to warmup LR')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR', help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='weight decay for SGD or Adam, ...')
    parser.add_argument('--test_attack_type', type=str, default='Partial_FunnyeyeGlasses')
    
    # MISC Settings
    parser.add_argument('--ckpt_root', type=str, default='./ckpt/')
    parser.add_argument('--csv_path', type=str, default='./csv/5_multiple_frame')
    parser.add_argument('--logs', type=str, default='./logs/')
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--save_models', action='store_true')
    parser.add_argument('--run_name', type=str, default='Capstone_Design')
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

    train(args, device)
