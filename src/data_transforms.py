from torchvision import transforms


def get_transforms(inp_size) -> transforms.Compose:
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]
    data_transforms = transforms.Compose([
                                            transforms.Resize(inp_size),
                                        ])

    return data_transforms