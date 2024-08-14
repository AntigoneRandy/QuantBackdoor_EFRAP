import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import torch

import sys

from . import utils


class Conv2dQuantize(nn.Conv2d):
    def __init__(self, quantize=False, **kwargs):
        super(Conv2dQuantize, self).__init__(**kwargs)
        self.quantize = quantize
        self.use_soft_rounding = True
        self.alpha = None

    def forward(self, input):
        if self.quantize == 'quant_8':
            utils.bit = 8
            # self.error = utils.compute_error(self.weight, utils.bit)
            input, input_scale, input_zero_point = utils.input_quantize(input)
            _, weight, weight_scale, _ = utils.weight_quantize(self.weight)
            output = F.conv2d(input-input_zero_point, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            return output * input_scale * weight_scale
        elif self.quantize == 'quant_4':
            utils.bit = 4
            input, input_scale, input_zero_point = utils.input_quantize(input)
            _, weight, weight_scale, _ = utils.weight_quantize(self.weight)
            output = F.conv2d(input-input_zero_point, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            return output * input_scale * weight_scale
        else:
            output = F.conv2d(input, self.weight, self.bias, self.stride,
                              self.padding, self.dilation, self.groups)
            return output

    def reset_quantize(self, quantize):
        self.quantize = quantize





class LinearQuantize(nn.Linear):
    def __init__(self, quantize=False, **kwargs):
        super(LinearQuantize, self).__init__(**kwargs)
        self.quantize = quantize

    def forward(self, input):
        if self.quantize == 'quant_8':
            utils.bit = 8
            input, input_scale, input_zero_point = utils.input_quantize(input)
            _, weight, weight_scale, _= utils.weight_quantize(self.weight)
            output = F.linear(input-input_zero_point, weight, self.bias)
            return output * input_scale * weight_scale
        elif self.quantize == 'quant_4':
            utils.bit = 4
            input, input_scale, input_zero_point = utils.input_quantize(input)
            _, weight, weight_scale, _= utils.weight_quantize(self.weight)
            output = F.linear(input-input_zero_point, weight, self.bias)
            return output * input_scale * weight_scale
        else:
            output = F.linear(input, self.weight, self.bias)
            return output

    def reset_quantize(self, quantize):
        self.quantize = quantize


def replace_quantize(model, quantize):
    def reset_quantize(self,quantize):
            for name, module in self.named_modules():
                if isinstance(module, nn.Conv2d):
                    module.reset_quantize(quantize)
                if isinstance(module, nn.Linear):
                    module.reset_quantize(quantize)
    def replace_modules(module):
            for name, child in module.named_children():
                if isinstance(child, nn.Conv2d):
                    setattr(module, name, Conv2dQuantize(
                quantize=quantize,
                in_channels=child.in_channels, 
                out_channels=child.out_channels,
                kernel_size=child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                dilation=child.dilation,
                groups=child.groups,
                bias=child.bias is not None,
                padding_mode=child.padding_mode
            ))
                elif isinstance(child, nn.Linear):
                    setattr(module, name, LinearQuantize(
                quantize=quantize,
                in_features=child.in_features,
                out_features=child.out_features, 
                bias=child.bias is not None
            ))
                else:
                    replace_modules(child)
        



                
    model.reset_quantize = reset_quantize


    replace_modules(model)
    return model