from ast import arg
import glob
import random
import os
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from dataloader.LoadData import LoadDataset ,train_transforms, val_transforms, test_transforms
from tqdm.notebook import tqdm
from model.VIT import Vit
import argparse
from torch.utils.data.distributed import DistributedSampler

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", help="local device id on current node", type=int)
    args = parser.parse_args()



    bs = 64
    epoches = 20
    lr = 1e-3
    gamma = 0.7
    seed = 42


    seed_everything(seed)

    device = 'cuda'
    train_dir = '/home/b212/ReapearPaper/Vit/Data/train'
    test_dir = '/home/b212/ReapearPaper/Vit/Data/test'
    train_list = glob.glob(os.path.join(train_dir,'*.jpg'))
    test_list = glob.glob(os.path.join(test_dir,'*.jpg'))


    labels = [path.split('/')[-1].split('.')[0] for path in train_list]


    #分训练集和测试集
    train_list, valid_list = train_test_split(train_list, test_size=0.2, stratify=labels, random_state=seed)
    # print(len(train_list))



    n_gpus = 1
    torch.distributed.init_process_group("nccl", world_size = n_gpus, rank=args.local_rank)
    torch.cuda.set_device(args.local_rank)
    train_data = LoadDataset(train_list, transform=train_transforms)
    val_data = LoadDataset(valid_list, transform=val_transforms)
    test_data = LoadDataset(test_list, transform=test_transforms)

    train_sampler = DistributedSampler(train_data)
    val_sampler = DistributedSampler(val_data)
    test_sampler = DistributedSampler(test_data)

    train_loader = DataLoader(dataset=train_data, batch_size= bs, drop_last= True, sampler=train_sampler)
    val_loader = DataLoader(dataset=val_data, batch_size= bs, shuffle=True, drop_last= True)
    test_loader = DataLoader(dataset=test_data, batch_size= bs, shuffle=True, drop_last= True)




    model = Vit(bs=bs, num_class=2, model_dim=256, n_head=4, num_layers=6).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)


    model = nn.parallel.DistributedDataParallel(model.cuda(args.local_rank), device_ids=[args.local_rank],find_unused_parameters=True)


    for epoch in range(epoches):
        epoch_loss = 0
        epoch_accuracy = 0
        train_sampler.set_epoch(epoch)
        
        for data, label in train_loader:
            data = data.cuda(args.local_rank)
            label = label.cuda(args.local_rank)
            output = model(data)
            loss = criterion(output, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            acc = (output.argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc / len(train_loader)
            epoch_loss += loss / len(train_loader)
        if epoch % 2 == 0 :
            with torch.no_grad():
                epoch_val_accuracy = 0 
                epoch_val_loss = 0
                
                for data, label in val_loader:
                    data = data.to(device)
                    label = label.to(device)
                    
                    val_output = model(data)
                    val_loss = criterion(val_output, label)
                    
                    acc = (val_output.argmax(dim=1) == label).float().mean()
                    epoch_val_accuracy += acc / len(val_loader)
                    epoch_val_loss += val_loss / len(val_loader)
                
                print(f"val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f} \n")    
        print(f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} \n") 
