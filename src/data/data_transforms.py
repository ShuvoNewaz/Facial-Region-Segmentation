from torchvision import transforms


def get_train_transforms_common(inp_size) -> transforms.Compose:
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]
    data_transforms = transforms.Compose([  
                                            transforms.ToPILImage(),
                                            transforms.Resize(inp_size),
                                            transforms.ToTensor(),
                                            transforms.RandomHorizontalFlip(0.6), # Comment
                                            transforms.RandomVerticalFlip(0.6),
                                            transforms.RandomRotation(30), # Comment to obtain required accuracy in ResNets
                                            # transforms.RandomCrop(480), # Comment
                                            # transforms.Normalize(mean=mean, std=std)
                                        ])

    return data_transforms


def get_train_transforms_image() -> transforms.Compose:
    data_transforms = transforms.Compose([
                                            transforms.ToPILImage(),
                                            transforms.ToTensor(),
                                            transforms.GaussianBlur([5, 1]),
                                            transforms.ColorJitter(brightness=(0.2, 1), contrast=(0.2), saturation=0.1, hue=(-0.1, 0.1)),
                                            transforms.RandomAdjustSharpness(0.2)
                                        ])

    return data_transforms


def get_val_transforms(inp_size) -> transforms.Compose:
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]
    data_transforms = transforms.Compose([
                                            transforms.ToPILImage(),
                                            transforms.Resize(inp_size),
                                            transforms.ToTensor(),
                                            # transforms.Normalize(mean=mean, std=std)
                                        ])

    return data_transforms