import glob
import random
import os
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.distributed import DistributedSampler
from dataloader.LoadData import LoadDataset ,train_transforms, val_transforms, test_transforms


from tqdm.notebook import tqdm
from model.VIT import Vit



def seed_everything(seed):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    

def train(train_data_loader, eval_data_loader, model, optimizer, num_epoch, log_step_interval,
          save_step_interval, eval_step_interval, save_path, resume="", device=""):
    start_epoch = 0
    stat_step = 0
    criterion = nn.CrossEntropyLoss()
    # 断点重续
    if resume != "":
        logging.warning(f"loading from {resume}")
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        start_step = checkpoint['step']
        
    for epoch_index in range(start_epoch, num_epoch):
        
        num_batches = len(train_data_loader)
        
        for batch_index, (data, label) in enumerate(train_data_loader):
            # 训练过程
            optimizer.zero_grad()
            step = num_batches*(epoch_index)+batch_index + 1
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            
            if step % log_step_interval == 0:
                logging.warning(f"epoch_index: {epoch_index}, batch_index: {batch_index}, loss: {loss}") 
            if step % save_step_interval == 0:
                os.makedirs(save_path,exist_ok=True)
                save_file = os.path.join(save_path, f"step_{step}.pt")
                torch.save({
                    'epoch':epoch_index,
                    'step':step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss':loss
                }, save_file)
                logging.warning(f"checkpoint has been saved in {save_file}")
            if step % eval_step_interval == 0:
                logging.warning("start to do evaluation...")
                # model.eval()
                with torch.no_grad():
                    epoch_val_accuracy = 0 
                    epoch_val_loss = 0
                    
                    for val_data, val_label in eval_data_loader:
                    
                        val_data = val_data.to(device)
                        val_label = val_label.to(device)
                        val_output = model(val_data)
                        val_loss = criterion(val_output, val_label)
                        
                        acc = (val_output.argmax(dim=1) == val_label).float().mean()
                        epoch_val_accuracy += acc / len(eval_data_loader)
                        epoch_val_loss += val_loss / len(eval_data_loader)
                    logging.warning(f"eval_acc:{epoch_val_accuracy}, eval_loss:{epoch_val_loss}")
                # model.train()


if __name__ == "__main__":
    
    bs = 64
    epoches = 20
    lr = 1e-3
    gamma = 0.7
    seed = 42
    device = 'cuda'
    
    seed_everything(seed)
        
    model = Vit(bs=bs, num_class=2, model_dim=256, n_head=4, num_layers=6).to(device)
    
    train_dir = '/home/b212/ReapearPaper/Vit/Data/train'
    test_dir = '/home/b212/ReapearPaper/Vit/Data/test'
    train_list = glob.glob(os.path.join(train_dir,'*.jpg'))
    test_list = glob.glob(os.path.join(test_dir,'*.jpg'))
        
    labels = [path.split('/')[-1].split('.')[0] for path in train_list]

    train_list, valid_list = train_test_split(train_list, test_size=0.2, stratify=labels, random_state=seed)


    train_data = LoadDataset(train_list, transform=train_transforms)
    val_data = LoadDataset(valid_list, transform=val_transforms) 

    train_loader = DataLoader(dataset=train_data, batch_size= bs, shuffle=True ,drop_last= True)
    val_loader = DataLoader(dataset=val_data, batch_size= bs, shuffle=True, drop_last= True)
    
    optimizer = optim.Adam(model.parameters(), lr = lr)

    train(train_data_loader=train_loader, eval_data_loader=val_loader, model=model, optimizer=optimizer, num_epoch=100, log_step_interval=20, save_step_interval=100, eval_step_interval=50,
          save_path="/home/b212/ReapearPaper/Vit/logs",device=device)
 
                    