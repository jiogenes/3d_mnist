import h5py
import pandas as pd
import argparse

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from util.data import CustomDataset
from util.util import str2bool
from model.model import BaseModel

def predict(args):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    test_df = pd.read_csv('/data/jyji/datasets/3D_NUMBER/sample_submission.csv')
    test_points = h5py.File('/data/jyji/datasets/3D_NUMBER/test.h5', 'r')
    test_dataset = CustomDataset(test_df['ID'].values, None, test_points, 'val')
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers)

    checkpoint = torch.load(args.weight)
    model = BaseModel()
    model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    model_preds = []
    with torch.no_grad():
        for data in tqdm(iter(test_loader)):
            data = data.float().to(device)
            batch_pred = model(data)
            model_preds += batch_pred.argmax(1).detach().cpu().numpy().tolist()
    
    test_df['label'] = model_preds
    test_df.to_csv('./submit.csv', index=False)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--test_batch_size', type=int, default=16)
    parser.add_argument('--weight', type=str, required=True, help='input path of weight')
    # parser.add_argument("--normalize", type=str2bool, nargs='?', const=True, default=False, required=True)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    predict(args)