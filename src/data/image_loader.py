import os
import torch
from torchvision import transforms
from torchvision.transforms import Compose
from torch.utils import data
import numpy as np
from typing import List, Tuple
from PIL import Image
import matplotlib.pyplot as plt


class ImageLoader(data.Dataset):
    def __init__(self, data_dir: str, split: str,
                 transform_common: Compose=None,
                 transform_image: Compose=None) -> None:
        """
        args:
            root_dir: Root working directory
            split:
        """
        super().__init__()
        self.split = split
        self.data_dir = data_dir
        if split not in ['train', 'val']:
            raise Exception('Split must be either "train" or "val"')
        self.split_dir = os.path.join(self.data_dir, split)
        self.image_dir = os.path.join(self.split_dir, 'image')
        self.seg_dir = os.path.join(self.split_dir, 'seg')
        self.transform_common = transform_common
        self.transform_image = transform_image
        self.dataset = self.load_images_with_masks()

    def load_images_with_masks(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Returns the list of tuples containing the image and the mask
        of the dataset.
        """
        dataset = []
        
        for image in os.listdir(self.image_dir):
            id = image.split('.')[0] # Removes .jpg from filename
            image_path = os.path.join(self.image_dir, image)
            seg_path = os.path.join(self.seg_dir, id + '.png')
            dataset.append((image_path, seg_path))

        return dataset
    
    def load_img_from_path(self, path: str) -> Image:
        """Loads an image as grayscale (using Pillow).

        Note: do not normalize the image to [0,1]

        Args:
            path: the file path to where the image is located on disk
        Returns:
            image: grayscale image with values in [0,255] loaded using pillow
                Note: Use 'L' flag while converting using Pillow's function.
        """
        img = Image.open(path).convert(mode='L')
        return img

    def __len__(self):

        return len(self.dataset)

    def __getitem__(self, index):
        image_path, seg_path = self.dataset[index]
        image, seg = self.load_img_from_path(image_path), self.load_img_from_path(seg_path)
        image = np.array(image)
        seg = np.array(seg)

        # Lump symmetrical aspects

        seg[seg == 3] = 2 # eyebrows
        seg[seg == 4] = 3 # eyes
        seg[seg == 5] = 3 # eyes
        seg[seg == 6] = 4 # nose
        seg[seg == 7] = 5 # lips
        seg[seg == 9] = 5 # lips
        seg[seg == 8] = 6 # mouth interior
        seg[seg == 10] = 7 # hair
        seg[seg == 11] = 8 # eyelids
        seg[seg == 12] = 8 # eyelids
        seg[seg == 13] = 9 # ears
        seg[seg == 14] = 9 # ears
        seg[seg == 15] = 0 # head covering (converted to background)
        seg[seg == 16] = 10 # glasses
        seg[seg == 17] = 1 # bald patch (converted to face)

        # Reduce background content

        non_zero_ind = np.nonzero(seg)
        if len(non_zero_ind[0]) > 0: # If an image isn't all background
            x_sorted = np.sort(non_zero_ind[0])
            y_sorted = np.sort(non_zero_ind[1])
            x_min = x_sorted[0]
            x_max = x_sorted[-1]
            y_min = y_sorted[0]
            y_max = y_sorted[-1]
            image = image[x_min:x_max, y_min:y_max]
            seg = seg[x_min:x_max, y_min:y_max]

        # Facilitate transformation of masks

        image_and_mask = np.concatenate((np.expand_dims(image, 2),
                                         np.expand_dims(seg, 2),
                                         np.expand_dims(seg, 2)), 2) # Extra dimension necessary because dim=1 or 3

        # Add rotation and flips
        image_and_mask = torch.as_tensor(image_and_mask)
        image_and_mask = torch.permute(image_and_mask, (2, 0, 1))
        if self.transform_common:
            image, seg, _ = self.transform_common(image_and_mask)
        if self.transform_image:
            if self.split == 'train':
                image = self.transform_image(torch.unsqueeze(image, 0))
            else:
                image, seg, _ = self.transform_image(image_and_mask)
            image = torch.squeeze(image, 0)

        return image, seg * 255 # torch transforms changes data from [0, 255] to [0, 1]