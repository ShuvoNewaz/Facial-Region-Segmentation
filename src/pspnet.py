
from typing import Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F
import models.resnet as models
from src.ppm import PPM


class PSPNet(nn.Module):
    def __init__(self,
                 layers=50,
                 bins=(1, 2, 3, 6),
                 dropout=0.1,
                 classes=2,
                 zoom_factor=8,
                 use_ppm=True,
                 criterion=nn.CrossEntropyLoss(ignore_index=255),
                 pretrained=True):
        super().__init__()
        # assert layers in [50, 101, 152]
        assert 2048 % len(bins) == 0
        assert classes > 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.use_ppm = use_ppm
        self.criterion = criterion

        if layers == 18:
            resnet = models.resnet18(pretrained=pretrained)
        elif layers == 34:
            resnet = models.resnet34(pretrained=pretrained)
        elif layers == 50:
            resnet = models.resnet50(pretrained=pretrained, deep_base=True)
        elif layers == 101:
            resnet = models.resnet101(pretrained=pretrained)
        else:
            resnet = models.resnet152(pretrained=pretrained)

        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.conv2, resnet.bn2, resnet.relu, resnet.conv3, resnet.bn3, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        for name, param in self.layer3.named_modules():
            if 'conv2' in name:
                param.dilation, param.padding, param.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in name:
                param.stride = (1, 1)
        for name, param in self.layer4.named_modules():
            if 'conv2' in name:
                param.dilation, param.padding, param.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in name:
                param.stride = (1, 1)

        fea_dim = 2048
        if use_ppm:
            self.ppm = PPM(fea_dim, int(fea_dim/len(bins)), bins)
            fea_dim *= 2
        self.cls = nn.Sequential(
            nn.Conv2d(fea_dim, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(512, classes, kernel_size=1)
        )
        if self.training:
            self.aux = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(256, classes, kernel_size=1)
            )

    def forward(self, x, y=None):
        if y is not None:
            y = y.long()
        x = x.float()
        main_loss, aux_loss = 0, 0
        if x.ndim == 3:
            x = torch.unsqueeze(x, 1)
        B, C, H, W = x.shape
        h = (H) // 8 * self.zoom_factor
        w = (W) // 8 * self.zoom_factor
        if C == 1:
            x = torch.tile(x, (1, 3, 1, 1))
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_tmp = self.layer3(x)
        x = self.layer4(x_tmp)

        if self.use_ppm:
            x = self.ppm(x)

        x = self.cls(x)
        aux = self.aux(x_tmp)

        if self.zoom_factor != 1:
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)      
            aux = F.interpolate(aux, size=(h, w), mode='bilinear', align_corners=True)
        if y is not None:
            main_loss = self.criterion(x, y)
            aux_loss = self.criterion(aux, y)
        yhat = torch.argmax(x, dim=1)
        
        return x, yhat, main_loss, aux_loss


def psp_model_optimizer(
        layers: int=18,
        bins=(1, 2, 3, 6),
        dropout: float=0.1,
        num_classes: int=2,
        zoom_factor: int=8,
        use_ppm: bool=True,
        criterion=nn.CrossEntropyLoss(ignore_index=255),
        pretrained: bool=True,
        deep_base: bool=True,
        lr: float=0.01,
        weight_decay: float=0.0001,
        momentum: float=0.9
    ) -> None:

    model = PSPNet(
                    layers=layers,
                    bins=bins,
                    dropout=dropout,
                    classes=num_classes,
                    zoom_factor=zoom_factor,
                    use_ppm=use_ppm,
                    criterion=criterion,
                    pretrained=pretrained,
                )
    
    layer0_params = {'params': model.layer0.parameters(), 'lr': lr, 'weight_decay': weight_decay, 'momentum': momentum}
    layer1_params = {'params': model.layer1.parameters(), 'lr': lr, 'weight_decay': weight_decay, 'momentum': momentum}
    layer2_params = {'params': model.layer2.parameters(), 'lr': lr, 'weight_decay': weight_decay, 'momentum': momentum}
    layer3_params = {'params': model.layer3.parameters(), 'lr': lr, 'weight_decay': weight_decay, 'momentum': momentum}
    layer4_params = {'params': model.layer4.parameters(), 'lr': lr, 'weight_decay': weight_decay, 'momentum': momentum}
    if model.use_ppm:
        ppm_params = {'params': model.ppm.parameters(), 'lr': 10*lr, 'weight_decay': weight_decay, 'momentum': momentum}
    cls_params = {'params': model.cls.parameters(), 'lr': 10*lr, 'weight_decay': weight_decay, 'momentum': momentum}
    aux_params = {'params': model.aux.parameters(), 'lr': 10*lr, 'weight_decay': weight_decay, 'momentum': momentum}
    if model.use_ppm:
        optimizer = torch.optim.SGD([
                                        layer0_params,
                                        layer1_params,
                                        layer2_params,
                                        layer3_params,
                                        layer4_params,
                                        ppm_params,
                                        cls_params,
                                        aux_params
                                    ], lr=lr, weight_decay=weight_decay, momentum=momentum)
    else:
        optimizer = torch.optim.Adam([
                                        layer0_params,
                                        layer1_params,
                                        layer2_params,
                                        layer3_params,
                                        layer4_params,
                                        cls_params,
                                        aux_params
                                    ], lr=lr, weight_decay=weight_decay, momentum=momentum)

    return model, optimizer


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
    input = torch.rand(4, 3, 473, 473).cuda()
    model = PSPNet(layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=21, zoom_factor=1, use_ppm=True, pretrained=True).cuda()
    model.eval()
    print(model)
    output = model(input)
    print('PSPNet', output.size())