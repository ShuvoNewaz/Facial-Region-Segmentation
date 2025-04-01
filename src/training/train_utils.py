import cv2
from typing import Tuple
import torch
from torch import nn
from src.training.metrics import *
import os
from src.models.pspnet.pspnet import psp_model_optimizer
from src.data.data_transforms import *


def save_model(model, optimizer, dir) -> None:
        """
        Saves the model state and optimizer state on the dict
        """
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            dir,
        )


class AverageMeter():
    """Computes and stores the average and current value"""
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def epoch_runner(loader, model, optimizer, num_classes, device):
    loss_meter = AverageMeter()
    f1_meter = AverageMeter()
    for batch_number, (image, mask) in enumerate(loader):
        n = image.shape[0]

        image = image.to(device)
        mask = mask.to(device)
        
        logits, y_hat, main_loss, aux_loss = model(image, mask)
        loss = main_loss + 0.4 * aux_loss

        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        _, _, _, _, f1_score = \
            BinaryF1(y_hat, mask, num_classes)

        # Clear from GPU

        image = image.detach().cpu()
        mask = mask.detach().cpu()
        logits = logits.detach().cpu()
        y_hat = y_hat.detach().cpu()
        if optimizer:
            loss = loss.detach().cpu()
            main_loss = main_loss.detach().cpu()
            aux_loss = aux_loss.detach().cpu()

        loss_meter.update(val=float(loss.item()), n=n)
        f1_meter.update(val=float(f1_score), n=n)
        torch.cuda.empty_cache()
        
    return loss_meter.avg, f1_meter.avg


def train(loader, model, optimizer, num_classes, device):
    model.train()

    return epoch_runner(loader, model, optimizer, num_classes, device)


def validate(loader, model, num_classes, device):
    model.eval()

    return epoch_runner(loader, model, None, num_classes, device)


def predict(image, image_size, model, resnet_layers, num_classes):
    '''
        args:
            image
    '''
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image_gray = torch.as_tensor(image_gray)
    image_gray = torch.unsqueeze(image_gray, 0)
    # Keep batchnorm from tripping
    image_gray = torch.concat((image_gray, image_gray), dim=0)
    model.eval()
    image_transform = get_val_transforms(image_size)
    image_gray = image_transform(image_gray)
    model, optimizer = psp_model_optimizer(resnet_layers,
                                           num_classes=num_classes)
    checkpoint = torch.load(os.path.join("saved_model", "pspnet", "checkpoint.pt"))    
    model.load_state_dict(checkpoint['model_state_dict'])
    _, yhat, _, _ = model(image_gray)
    
    return image_gray[0], yhat[0]