from collections import OrderedDict
from models.resnet import resnet18 as _resnet18
from models.resnet import resnet50 as _resnet50
from models.mobilenetv2 import mobilenetv2 as _mobilenetv2
from models.mnasnet import mnasnet as _mnasnet
from models.regnet import regnetx_600m as _regnetx_600m
from models.regnet import regnetx_3200m as _regnetx_3200m
import torch
dependencies = ['torch']
model_path = {
    'resnet18': r'C:\Users\User\Desktop\Utar\URS\PD-Quant\checkpoint\resnet18_imagenet.pth',
    'resnet50': r'C:\Users\User\Desktop\Utar\URS\PD-Quant\checkpoint\resnet50_imagenet.pth',
    'mbv2': r'C:\Users\User\Desktop\Utar\URS\PD-Quant\checkpoint\mobilenetv2.pth',
    'reg600m': r'C:\Users\User\Desktop\Utar\URS\PD-Quant\checkpoint\regnet_600m.pth',
    'reg3200m': r'C:\Users\User\Desktop\Utar\URS\PD-Quant\checkpoint\regnet_3200m.pth',
    'mnasnet': r'C:\Users\User\Desktop\Utar\URS\PD-Quant\checkpoint\mnasnet.pth',
}


def resnet18(pretrained=False, **kwargs):
    # Call the model, load pretrained weights
    model = _resnet18(**kwargs)
    if pretrained:
        checkpoint = torch.load(model_path['resnet18'], map_location='cpu')
        model.load_state_dict(checkpoint)
    return model


def resnet50(pretrained=False, **kwargs):
    # Call the model, load pretrained weights
    model = _resnet50(**kwargs)
    if pretrained:
        checkpoint = torch.load(model_path['resnet50'], map_location='cpu')
        model.load_state_dict(checkpoint)
    return model


def mobilenetv2(pretrained=False, **kwargs):
    # Call the model, load pretrained weights
    model = _mobilenetv2(**kwargs)
    if pretrained:
        checkpoint = torch.load(model_path['mbv2'], map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    return model


def regnetx_600m(pretrained=False, **kwargs):
    # Call the model, load pretrained weights
    model = _regnetx_600m(**kwargs)
    if pretrained:
        checkpoint = torch.load(model_path['reg600m'], map_location='cpu')
        model.load_state_dict(checkpoint)
    return model


def regnetx_3200m(pretrained=False, **kwargs):
    # Call the model, load pretrained weights
    model = _regnetx_3200m(**kwargs)
    if pretrained:
        checkpoint = torch.load(model_path['reg3200m'], map_location='cpu')
        model.load_state_dict(checkpoint)
    return model


def mnasnet(pretrained=False, **kwargs):
    # Call the model, load pretrained weights
    model = _mnasnet(**kwargs)
    if pretrained:
        checkpoint = torch.load(model_path['mnasnet'], map_location='cpu')
        model.load_state_dict(checkpoint)
    return model
