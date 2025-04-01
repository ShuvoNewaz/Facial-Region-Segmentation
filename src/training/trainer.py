import os
import torch
from src.data.image_loader import *
from src.training.metrics import *
from src.models.pspnet.pspnet import *
from typing import List, Tuple
from torch.optim import Optimizer
from typing import List
from src.training.train_utils import *
from torch.utils.data import DataLoader
from src.data.data_transforms import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Trainer:
    """Class that stores model training metadata."""

    def __init__(self,
                 inp_size,
                 resnet_size,
                 lr=1e-3,
                 weight_decay=1e-5,
                 batch_size: int=8,
                 num_classes=11) -> None:
        self.device = device
        self.num_classes = num_classes
        self.saved_model_dir = os.path.join('saved_model', "pspnet")
        os.makedirs(self.saved_model_dir, exist_ok=True)
        checkpoint_name = "checkpoint.pt"
        self.checkpoint_dir = os.path.join(self.saved_model_dir, checkpoint_name)

        self.model, self.optimizer = psp_model_optimizer(layers=resnet_size, num_classes=num_classes,
                                                         lr=lr, weight_decay=weight_decay) # Use default parameters
        # self.model = torch.nn.DataParallel(self.model)
        self.model = self.model.to(self.device)
        dataloader_args = {"num_workers": 4, "pin_memory": True} if torch.cuda.is_available() else {}

        self.train_dataset = ImageLoader(data_dir="dataset",
                                         split='train',
                                         transform_common=get_train_transforms_common(inp_size),
                                         transform_image=get_train_transforms_image())
        
        self.val_dataset = ImageLoader(data_dir="dataset",
                                       split='val',
                                       transform_image=get_val_transforms(inp_size))

        # Drop last batch if last batch size is 1 to keep batchnorm from breaking.
        self.num_train_images = len(self.train_dataset)
        self.num_val_images = len(self.val_dataset)
        drop_last_train = self.num_train_images % batch_size == 1
        drop_last_val = self.num_val_images % batch_size == 1
        
        self.train_loader = DataLoader(
                                        self.train_dataset,
                                        batch_size=batch_size,
                                        shuffle=True, **dataloader_args,
                                        drop_last=drop_last_train
                                        )
        self.val_loader = DataLoader(
                                        self.val_dataset,
                                        batch_size=batch_size,
                                        shuffle=True, **dataloader_args,
                                        drop_last=drop_last_val
                                    )
        
        self.train_loss_history = []
        self.validation_loss_history = []
        self.train_f1_history = []
        self.validation_f1_history = []

    def run_training_loop(self, num_epochs: int, load_from_disk: bool) -> None:
        if load_from_disk:
            checkpoint = torch.load(self.checkpoint_dir)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_f1 = 0
        for epoch_idx in range(num_epochs):
            train_loss, train_f1 = train(self.train_loader, self.model,
                                         self.optimizer, self.num_classes, self.device)
            self.train_loss_history.append(train_loss)
            self.train_f1_history.append(train_f1)

            val_loss, val_f1 = validate(self.val_loader, self.model,
                                        self.num_classes, self.device)
            self.validation_loss_history.append(val_loss)
            self.validation_f1_history.append(val_f1)

            if val_f1 > best_f1:
                best_f1 = val_f1
                if os.path.exists(self.checkpoint_dir):
                    os.remove(self.checkpoint_dir)
                save_model(self.model, self.optimizer, self.checkpoint_dir)

            print(f"Epoch:{epoch_idx + 1}")
            print(f"\tTrain Loss:{train_loss:.4f}")
            print(f"\tValidation Loss:{val_loss:.4f}")
            print(f"\tTrain F1 Score: {train_f1:.4f}")
            print(f"\tValidation F1 Score: {val_f1:.4f}")
            
        self.model = self.model.cpu()