import os
import time
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

from model import *
from loader import *
from training import *

imgs_dir = r'/home/dguo/road_seg/training/images'
masks_dir = r'/home/dguo/road_seg/training/groundtruth'
P_tr = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.RandomVerticalFlip(p=0.5),
    torchvision.transforms.Resize(size=448, interpolation=3)])
P_val = torchvision.transforms.Resize(size=448, interpolation=3)
train_set = BaseDataset(imgs_dir, masks_dir, image_set="train", split_ratio=0.98, preprocess=P_tr, color_jitter=True,
                        rotation=True, verbose=True)
val_set = BaseDataset(imgs_dir, masks_dir, image_set="val", split_ratio=0.98, preprocess=P_val, color_jitter=False,
                      rotation=False, verbose=True)

# select GPU or CPU training
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
# train ##
LR = 1e-4
n_f = 64
dropout = 0.2
model_cpu = UNet(n_channels=3, n_classes=1, n_filters=n_f, dropout=dropout)
model_gpu = model_cpu.to(device)
criterion = torch.nn.BCEWithLogitsLoss().to(device)
optimizer = torch.optim.Adam(model_gpu.parameters(), lr=LR, weight_decay=1e-7)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=7, verbose=True)
writer = SummaryWriter()
start_epoch = 0
end_epoch = 100
print('\n', '------------------------------------------------------------------')
print(
    f'Start traing process at epoch {start_epoch}, learning rate: {LR}, model: U-net-64 (transConv), dropout={dropout}')
train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False)

# train
total_time = 0.0
iteration = 0
SAVE_PATH = r'/home/dguo/road_seg/checkpoints/T_Unet_{}_transpose_d{}'.format(n_f, dropout)
score_min = 0.82
for epoch in range(start_epoch, end_epoch):
    start_epoch_time = time.time()
    train_loss, iteration = train_epoch(model_gpu, criterion, train_loader, epoch, iteration, optimizer, scheduler,
                                        writer, device)
    val_score = eval_model(model_gpu, val_loader, epoch, writer, device)
    scheduler.step(val_score)
    end_epoch_time = time.time() - start_epoch_time
    if val_score > score_min:
        save_to_checkpoint(SAVE_PATH, epoch, model_gpu, optimizer, scheduler, verbose=True)
    score_min = max(0.82, val_score)

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