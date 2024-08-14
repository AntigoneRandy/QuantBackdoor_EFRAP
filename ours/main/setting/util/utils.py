import torch
from torch import autograd

bit = 8

class Round(autograd.Function):

    @staticmethod
    def forward(ctx, inputs):
        return torch.floor(inputs + 0.5)

    @staticmethod
    def backward(ctx, grads):
        return grads

def input_quantize(tensor,minval = None,maxval = None):
    if minval is not None and maxval is not None:
        scale = (maxval - minval) / ((1<<bit)-1)
        zero_point_quantize = -(minval / scale + (1<<(bit-1)))
    else:
        # scale = (tensor.max() - tensor.min()) / 255.
        # zero_point_quantize = -(tensor.min() / scale + 128.) #8bit
        scale = (tensor.max() - tensor.min()) / ((1<<bit)-1)
        zero_point_quantize = -(tensor.min() / scale + (1<<(bit-1)))
    # scale = (tensor.max() - tensor.min()) / 15.
    # zero_point_quantize = -(tensor.min() / scale + 8.)
    tensor_quantize = tensor / scale + zero_point_quantize
    tensor_round = Round.apply(tensor_quantize)
    # tensor_round  = new_round(tensor_quantize,maxx=True,percent=0.5)
    zero_point_round = Round.apply(zero_point_quantize)
    return tensor_round, scale, zero_point_round


def weight_quantize(tensor,maxx=True,percent=0,pruning=False,minval = None,maxval = None):
    if minval is not None and maxval is not None:
            scale = maxval / ((1<<bit-1)-0.5)
    else:
      scale = tensor.abs().max() /  ((1<<bit-1)-0.5)

    scale = scale.to(tensor.device)
    tensor_quantize = tensor / scale

    
    tensor_round = Round.apply(tensor_quantize)

    # print("hello")

    decimal = tensor_quantize - Round.apply(tensor_quantize)

    if pruning:

        pass


    return tensor_quantize, tensor_round, scale, decimal


def conv2d_quantize(tensor):
    # per_channel quantization
    scale = tensor.abs().max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0] / ((1<<bit-1)-0.5)
    tensor_quantize = tensor / scale
    tensor_round = round_through(tensor_quantize)
    decimal = tensor_quantize - tensor_quantize.round()

    return tensor_quantize, tensor_round, scale, decimal


def linear_quantize(tensor):
    # per_channel quantization
    scale = tensor.abs().max(dim=1, keepdim=True)[0] / ((1<<bit-1)-0.5)
    tensor_quantize = tensor / scale
    tensor_round = round_through(tensor_quantize)
    decimal = tensor_quantize - Round.apply(tensor_quantize)



    return tensor_quantize, tensor_round, scale, decimal


def round_through(x):
    rounded = Round.apply(x)
    return rounded



def keep_scale(param, base_param):
    abs_base_param = base_param.abs()
    max_base_param = abs_base_param.max()
    new_param = torch.clamp(torch.where(max_base_param - abs_base_param > 0, param, base_param),
                            -max_base_param, max_base_param)
    return new_param


def keep_scale_linear(param, base_param):
    abs_base_param = base_param.abs()
    max_base_param = abs_base_param.max(dim=1, keepdim=True)[0]
    new_param = torch.clamp(torch.where(max_base_param - abs_base_param > 0, param, base_param),
                            -max_base_param, max_base_param)
    return new_param


def keep_scale_conv2d(param, base_param):
    abs_base_param = base_param.abs()
    max_base_param = abs_base_param.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
    new_param = torch.clamp(torch.where(max_base_param - abs_base_param > 0, param, base_param),
                            -max_base_param, max_base_param)
    return new_param



def compute_error(tensor):
    scale = tensor.abs().max() /  ((1<<bit-1)-0.5)
    tensor_quantize = tensor / scale
    tensor_round = round_through(tensor_quantize)
    quantization_error = tensor - tensor_round * scale
    return quantization_error


