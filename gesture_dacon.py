import random
import pandas as pd
import numpy as np
import os
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import torchvision
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings(action='ignore') 

CFG = {
    'FPS':30,
    'IMG_SIZE':128,
    'EPOCHS':10,
    'LEARNING_RATE':3e-4,
    'BATCH_SIZE':4,
    'SEED':41
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

class CustomDataset(Dataset):
    def __init__(self, video_path_list, label_list):
        self.video_path_list = video_path_list
        self.label_list = label_list
        
    def __getitem__(self, index):
        frames = self.get_video(self.video_path_list[index])
        
        if self.label_list is not None:
            label = self.label_list[index]
            return frames, label
        else:
            return frames
        
    def __len__(self):
        return len(self.video_path_list)
    
    def get_video(self, path):
        frames = []
        cap = cv2.VideoCapture(path)
        for _ in range(CFG['FPS']):
            _, img = cap.read()
            img = cv2.resize(img, (CFG['IMG_SIZE'], CFG['IMG_SIZE']))
            img = img / 255.
            frames.append(img)
        return torch.FloatTensor(np.array(frames)).permute(3, 0, 1, 2)


def train_model(model, optimizer, train_loader, val_loader, scheduler, device):
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    
    best_val_score = 0
    best_model = None
    print_train_loss=[]
    print_val_loss= []
    for epoch in range(1, CFG['EPOCHS']+1):
        model.train()
        train_loss = []
        
        for videos, labels in iter(train_loader):
            videos = videos.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            output = model(videos)
            loss = criterion(output, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
                    
        _val_loss, _val_score = validation(model, criterion, val_loader, device)
        _train_loss = np.mean(train_loss)
        print(f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}] Val F1 : [{_val_score:.5f}]')
        
        print_train_loss.append(_train_loss)
        print_val_loss.append(_val_loss)
        
        if scheduler is not None:
            scheduler.step(_val_score)
            
        if best_val_score < _val_score:
            best_val_score = _val_score
            best_model = model
    
    return best_model, print_train_loss, print_val_loss

def validation(model, criterion, val_loader, device):
    model.eval()
    val_loss = []
    preds, trues = [], []
    
    with torch.no_grad():
        for videos, labels in iter(val_loader):
            videos = videos.to(device)
            labels = labels.to(device)
            
            logit = model(videos)
            
            loss = criterion(logit, labels)
            
            val_loss.append(loss.item())
            
            preds += logit.argmax(1).detach().cpu().numpy().tolist()
            trues += labels.detach().cpu().numpy().tolist()
        
        _val_loss = np.mean(val_loss)
    
    _val_score = f1_score(trues, preds, average='macro')
    return _val_loss, _val_score

def inference(model, test_loader, device):
    model.to(device)
    model.eval()
    preds = []
    with torch.no_grad():
        for videos in iter(test_loader):
            videos = videos.to(device)
            
            logit = model(videos)

            preds += logit.argmax(1).detach().cpu().numpy().tolist()
    return preds

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    seed_everything(CFG['SEED']) # Seed 고정
    df = pd.read_csv('/nfs_shared_/laj/termproject/train.csv')
    train, val, _, _ = train_test_split(df, df['label'], test_size=0.2, random_state=CFG['SEED'])
    train_dataset = CustomDataset(train['path'].values, train['label'].values)
    train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=True, num_workers=0)

    val_dataset = CustomDataset(val['path'].values, val['label'].values)
    val_loader = DataLoader(val_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

    model = torchvision.models.video.r3d_18(pretrained=False, progress=True)
    model.eval().to(device)
    optimizer = torch.optim.Adam(params = model.parameters(), lr = CFG["LEARNING_RATE"])

    infer_model, train_loss, val_loss = train_model(model, optimizer, train_loader, val_loader, None, device)

    # torch.save(infer_model, "model5.pt")
    plt.plot([i for i in train_loss], label='training loss')
    plt.plot(val_loss, label='validation loss')
    plt.title('Loss at the end of each epoch')
    plt.legend()
    plt.savefig('./r3d_plot.png')


    test = pd.read_csv('/nfs_shared_/laj/termproject/test.csv')
    test_dataset = CustomDataset(test['path'].values, None)
    test_loader = DataLoader(test_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

    preds = inference(model, test_loader, device)
    print(preds)

    submit = pd.read_csv('/nfs_shared_/laj/termproject/sample_submission.csv')
    submit['label'] = preds
    submit.head()
    # submit.to_csv('./baseline_submit5.csv', index=False)

if __name__ == "__main__":
    main()