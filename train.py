import h5py
import pandas as pd
import numpy as np
import argparse
from datetime import date

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score

from util.data import CustomDataset
from util.util import str2bool
from model.model import BaseModel

def train(args):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    all_df = pd.read_csv('/data/jyji/datasets/3D_NUMBER/train.csv')
    all_points= h5py.File('/data/jyji/datasets/3D_NUMBER/train.h5', 'r') 
    train_df = all_df.sample(frac=0.8)
    val_df = all_df.drop(train_df.index)
    train_dataset = CustomDataset(train_df['ID'].values, train_df['label'].values, all_points, 'train')
    train_loader = DataLoader(train_dataset, batch_size = args.train_batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataset = CustomDataset(val_df['ID'].values, val_df['label'].values, all_points, 'val')
    val_loader = DataLoader(val_dataset, batch_size=args.valid_batch_size, shuffle=False, num_workers=args.num_workers)

    model = BaseModel()
    optimizer = optim.Adam(params = model.parameters(), lr = args.lr)
    scheduler = None

    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    best_score = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = []
        for data, label in tqdm(iter(train_loader)):
            data, label = data.float().to(device), label.long().to(device)
            optimizer.zero_grad()
            
            output = model(data)
            loss = criterion(output, label)
            
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
        
        if scheduler is not None:
            scheduler.step()
            
        val_loss, val_acc = validation(model, criterion, val_loader, device)
        print(f'Epoch : [{epoch}] Train Loss : [{np.mean(train_loss):.4f}] Val Loss : [{val_loss:.4f}] Val ACC : [{val_acc:.4f}]')
        
        if best_score < val_acc:
            best_score = val_acc
            if args.save:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, f'./weights/best_model.pth')

        if best_score > 0.9 and epoch % 10 == 0:
            torch.save(model.state_dict(), f'./weights/{date.today()}_epoch_{epoch:03d}_lr_{args.lr}_batch_{args.train_batch_size}_loss_{val_acc:.5f}.pth')

def validation(model, criterion, val_loader, device):
    model.eval()
    true_labels = []
    model_preds = []
    val_loss = []
    with torch.no_grad():
        for data, label in tqdm(iter(val_loader)):
            data, label = data.float().to(device), label.long().to(device)
            
            model_pred = model(data)
            loss = criterion(model_pred, label)
            
            val_loss.append(loss.item())
            
            model_preds += model_pred.argmax(1).detach().cpu().numpy().tolist()
            true_labels += label.detach().cpu().numpy().tolist()
    
    return np.mean(val_loss), accuracy_score(true_labels, model_preds)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--valid_batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--save', type=str2bool, nargs='?', const=True, default=True)
    # parser.add_argument("--normalize", type=str2bool, nargs='?', const=True, default=False, required=True)
    ## test

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    train(args)