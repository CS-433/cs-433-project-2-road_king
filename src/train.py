"""
Script for training U-Net model
"""
import os
import argparse
import time
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

from model import *
from loader import *
from training import *


def get_args():
    parser = argparse.ArgumentParser(description='Train UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=120,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, default=1e-4,
                        help='Learning rate', dest='lr')
    parser.add_argument('-w', '--weight-decay', metavar='WD', type=float, default=1e-7,
                        help='Learning rate', dest='wd')
    parser.add_argument('-f', '--num-filters', type=int, default=64, dest='num_filters',
                        help='Number of filters in the first layer of U-Net')
    parser.add_argument('-d', '--dropout', dest='dropout', type=float, default=0.2,
                        help='Dropout rate of U-net')
    parser.add_argument('-s', '--split', dest='split', type=float, default=0.98,
                        help='Percent of the data that is used for training')
    parser.add_argument('-m', '--min', dest='min', type=float, default=0.82,
                        help='Minimum validation score to save the model dring training')
    parser.add_argument('-o', '--output', type=str, default='UNet', dest='output',
                        help='Name of output file for saved epochs')

    return parser.parse_args()


args = get_args()
end_epoch = args.epochs
batch_size = args.batchsize
LR = args.lr
decay = args.wd
n_f = args.num_filters
dropout = args.dropout
split = args.split
score_min = args.min
root_dir = " "  # your root_dir
SAVE_PATH = os.path.join(root_dir, '/checkpoints', args.output)

imgs_dir = r'/data/training/images'
masks_dir = r'/data/training/groundtruth'
P_tr = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.RandomVerticalFlip(p=0.5),
    torchvision.transforms.Resize(size=448, interpolation=3)])
P_val = torchvision.transforms.Resize(size=448, interpolation=3)
train_set = BaseDataset(imgs_dir, masks_dir, image_set="train", split_ratio=split, preprocess=P_tr, color_jitter=True,
                        rotation=True, verbose=True)
val_set = BaseDataset(imgs_dir, masks_dir, image_set="val", split_ratio=split, preprocess=P_val, color_jitter=False,
                      rotation=False, verbose=True)

# select GPU or CPU training
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
model_cpu = UNet(n_channels=3, n_classes=1, n_filters=n_f, dropout=dropout)
model_gpu = model_cpu.to(device)
criterion = torch.nn.BCEWithLogitsLoss().to(device)
optimizer = torch.optim.Adam(model_gpu.parameters(), lr=LR, weight_decay=decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=7, verbose=True)
writer = SummaryWriter()
start_epoch = 0
print('\n', '------------------------------------------------------------------')
print(
    f'Start traing process at epoch {start_epoch}, learning rate: {LR}, model: U-net-64 (transConv), dropout={dropout}')
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False)

# training
total_time = 0.0
iteration = 0
for epoch in range(start_epoch, end_epoch):
    start_epoch_time = time.time()
    train_loss, iteration = train_epoch(model_gpu, criterion, train_loader, epoch, iteration, optimizer, scheduler,
                                        writer, device)
    val_score = eval_model(model_gpu, val_loader, epoch, writer, device)
    scheduler.step(val_score)
    end_epoch_time = time.time() - start_epoch_time
    if val_score > score_min:
        save_to_checkpoint(SAVE_PATH, epoch, model_gpu, optimizer, scheduler, verbose=True)
    score_min = max(args.min, val_score)

    print('\n', '-----------------------------------------------------')
    print(f'End of epoch {epoch}, total iterations {iteration}')
    print('Training epoch loss: {:.4f}'.format(train_loss))
    print('Validation epoch score: {:.4f}'.format(val_score))
    print('Epoch time: {:.2f}'.format(end_epoch_time))
    print('--------------------------------------------------------', '\n')
    total_time += end_epoch_time

print('\n', '**************************************************************')
print(f'End training at epoch {end_epoch}')
print('total time: {:.2f}'.format(total_time))
