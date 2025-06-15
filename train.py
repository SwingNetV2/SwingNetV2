import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from early_stopping import EarlyStopping
from util import AverageMeter
from model import EventDetector_mb4
from dataloader import GolfDB, transform_video_frames

import pytorch_warmup as warmup
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Hyperparameters
it_save = 100
n_cpu = 0
seq_length = 64
bs = 8
k = 10
lr = 0.001
num_epochs = 2000

# TensorBoard Writer 
writer = SummaryWriter('runs/mbv4')

# Model & Criterion
model = EventDetector_mb4(
    lstm_layers=1,
    lstm_hidden=256,
    bidirectional=True,
    dropout=False
)
model.cuda()
torch.cuda.empty_cache()

# 클래스 불균형 가중치 적용
weights = torch.FloatTensor([1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/35]).cuda()
criterion = torch.nn.CrossEntropyLoss(weight=weights)

# Optimizer & Schedulers
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=lr,
                              betas=(0.9, 0.999),
                              weight_decay=0.01)

num_steps = len(train_data_loader) * num_epochs
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)

warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
early_stopping = EarlyStopping(patience=30, verbose=True)


# DataLoader
train_dataset = GolfDB(data_file='data/train_split.pkl',
                 vid_dir='data/videos_160',
                 seq_length=seq_length,
                 transform=transform_video_frames,
                 train=True)

train_data_loader = DataLoader(train_dataset,
                         batch_size=bs,
                         shuffle=True,
                         drop_last=True
                         )

val_dataset = GolfDB(data_file='data/val_split.pkl',
             vid_dir='data/videos_160',
             seq_length=seq_length,
             transform=transform_video_frames,
             train=True
             )

val_data_loader = DataLoader(val_dataset,
                         batch_size=bs,
                         shuffle=False,
                         drop_last=True
                         )

# Metric Meters
losses   = AverageMeter()
v_losses = AverageMeter()


# Training loop 
for epoch in range(1,num_epochs+1):
    model.train()
    for sample in train_data_loader:
        images, labels = sample['images'].cuda(), sample['labels'].cuda()
        labels = labels.view(bs * seq_length)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        # Scheduler
        with warmup_scheduler.dampening():
            lr_scheduler.step()
          
        losses.update(loss.item(), images.size(0))

        # TensorBoard 기록 및 로그
        writer.add_scalar('Training/Loss', loss.item(), epoch)
        writer.flush()
        print('epoch: {}\tLoss: {loss.val:.4f} ({loss.avg:.4f})'.format(epoch, loss=losses))
        break

    # Validation
    all_preds = []
    all_labels = []

    for sample in val_data_loader:
      with torch.no_grad():
            model.eval()
            images, labels = sample['images'].cuda(), sample['labels'].cuda()
            logits = model(images)
            labels = labels.view(bs * seq_length)
            vloss = criterion(logits, labels)
            v_losses.update(vloss.item(), images.size(0))

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Metrics 계산
            acc  = accuracy_score(all_labels, all_preds)
            prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
            rec  = recall_score(all_labels, all_preds, average='macro', zero_division=0)
            f1   = f1_score(all_labels, all_preds, average='macro', zero_division=0)

            # TensorBoard 기록
            writer.add_scalar('Validation/Loss', vloss.item(), epoch)
            writer.add_scalar('Validation/Accuracy', acc, epoch)
            writer.add_scalar('Validation/Precision', prec, epoch)
            writer.add_scalar('Validation/Recall', rec, epoch)
            writer.add_scalar('Validation/F1', f1, epoch)
            writer.add_scalars('Training vs. Validation Loss',
                      {'Training' : loss.item(), 'Validation' : vloss.item()}, epoch)
            writer.flush()
        
            print('epoch: {}\tV_Loss: {vloss.val:.4f} ({vloss.avg:.4f})'.format(epoch, vloss=v_losses),
                   'Acc: {} | Prec: {} | Rec: {} | F1: {}'.format(acc:.4f, prec:.4f, rec:.4f, f1:.4f)
            )
        
    # Early Stopping
    early_stopping(vloss.cuda(), model)
    if early_stopping.early_stop:
      print("Early stopping triggered")
      break

writer.close()

#0616
torch.save(model, 'models/final.pth')
print('The End... is near...')
