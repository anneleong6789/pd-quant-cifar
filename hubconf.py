import torch
import torchvision.models as tv_models

dependencies = ['torch', 'torchvision']

def resnet18(pretrained=False, **kwargs):
    return tv_models.resnet18(weights="IMAGENET1K_V1" if pretrained else None, **kwargs)

def resnet50(pretrained=False, **kwargs):
    return tv_models.resnet50(weights="IMAGENET1K_V1" if pretrained else None, **kwargs)

def mobilenetv2(pretrained=False, **kwargs):
    return tv_models.mobilenet_v2(weights="IMAGENET1K_V1" if pretrained else None, **kwargs)

def mnasnet(pretrained=False, **kwargs):
    return tv_models.mnasnet1_0(weights="IMAGENET1K_V1" if pretrained else None, **kwargs)

def regnetx_600m(pretrained=False, **kwargs):
    return tv_models.regnet_x_600m(weights="IMAGENET1K_V1" if pretrained else None, **kwargs)

def regnetx_3200m(pretrained=False, **kwargs):
    return tv_models.regnet_x_3_2gf(weights="IMAGENET1K_V1" if pretrained else None, **kwargs)
