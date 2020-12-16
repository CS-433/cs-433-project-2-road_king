import torch
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Any, Tuple
from dice import *


def train_epoch(model: torch.nn.Module,  # U-Net
                criterion: Any,  # torch.nn.loss_fun
                train_loader: torch.utils.data.dataloader.DataLoader,
                epoch: int,
                iteration: int,
                optimizer: Any,  # from torch.optim (i.e. Adam, SGD, ...)
                scheduler: Any,
                writer: Optional[torch.utils.tensorboard.writer.SummaryWriter] = None,
                device: torch.device = torch.device("cuda"),
                verbose: bool = False) -> Tuple[float, int]:
    """ Training loop, one epoch for U-net model
    Implements the forward + backward pass for each sample and obtains the loss on the training data
    """
    # switch to train mode
    model.train()
    train_running_loss = 0.0

    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()

        imgs = batch['image'].to(device, non_blocking=True)
        true_masks = batch['mask']
        true_masks = true_masks.to(device, non_blocking=True)
        # compute output and loss
        masks_pred = model(imgs)
        loss = criterion(masks_pred, true_masks)
        # record loss
        train_running_loss += float(loss.item())
        if verbose:
            idx = batch["ID"]
            print("ID:", idx, f" training loss {iteration}: ", float(loss.item()))
        # compute gradient and do optimizer step
        loss.backward()
        # torch.nn.utils.clip_grad_value_(model.parameters(), 0.1)
        optimizer.step()
        iteration += 1
        if writer is not None:
            writer.add_scalar('Loss/train', loss.item(), iteration)

    train_epoch_loss = train_running_loss / len(train_loader)
    if writer is not None:
        writer.add_scalar('train epoch loss', train_epoch_loss, epoch)

    return train_epoch_loss, iteration


def val_epoch(model: torch.nn.Module,
              criterion: Any,  # torch.nn.loss_fun
              val_loader: torch.utils.data.dataloader.DataLoader,
              epoch: int,
              writer: Optional[torch.utils.tensorboard.writer.SummaryWriter] = None,
              device: torch.device = torch.device("cuda"),
              dataset_type: str = 'val',
              verbose: bool = True) -> float:
    """ Validation loop (no backprop)
    Obtains the loss on the validation (or test) data
    """
    # freeze model
    model.eval()
    val_running_loss = 0.0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            imgs = batch['image'].to(device, non_blocking=True)
            true_masks = batch['mask']
            true_masks = true_masks.to(device, non_blocking=True)
            masks_pred = model(imgs)
            loss = criterion(masks_pred, true_masks)
            # record loss
            val_running_loss += float(loss.item())
            if verbose:
                idx = batch["ID"]
                print("ID:", idx, f" validation loss {i}: ", float(loss.item()))

            if writer is not None:
                writer.add_scalar(f'Loss/{dataset_type}', loss.item(), i)

    val_epoch_loss = val_running_loss / len(val_loader)

    if writer is not None:
        writer.add_scalar('{} epoch loss'.format(dataset_type), val_epoch_loss, epoch)

    return val_epoch_loss


def eval_model(model: torch.nn.Module,
               loader: torch.utils.data.dataloader.DataLoader,
               epoch: int,
               writer: Optional[torch.utils.tensorboard.writer.SummaryWriter] = None,
               device: torch.device = torch.device("cuda"),
               dataset_type: str = 'eval',
               verbose: bool = True) -> float:
    """ Validation loop (no backprop)
    Obtains the loss on the validation (or test) data
    """
    # freeze model
    model.eval()
    total_score = 0.0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            imgs = batch['image'].to(device, non_blocking=True)
            true_masks = batch['mask'].to(device, non_blocking=True)
            masks_pred = model(imgs)
            pred = torch.sigmoid(masks_pred)
            pred = (pred > 0.5).float()
            score = dice_coeff(pred, true_masks)
            total_score += score.item()
            if verbose:
                idx = batch["ID"]
                print("ID:", idx, f"validation score {i}: ", float(score.item()))
            if writer is not None:
                writer.add_scalar(f'Score/{dataset_type}', score.item(), i)

    eval_score = total_score / len(loader)

    if writer is not None:
        writer.add_scalar('{} epoch score'.format(dataset_type), eval_score, epoch)

    return eval_score


def save_to_checkpoint(save_path, epoch, model, optimizer, scheduler=None, verbose=True):
    # save checkpoint to disk
    d_sche = None
    if scheduler is not None:
        d_sche = scheduler.state_dict()
    if save_path is not None:
        checkpoint = {'epoch': epoch + 1,
                      'model': model.state_dict(),
                      'optimizer': optimizer.state_dict(),
                      'scheduler': d_sche
                      }
        torch.save(checkpoint, '{}_epoch_{}.pt'.format(save_path, epoch))

    if verbose:
        print("saved model at epoch {}".format(epoch))
