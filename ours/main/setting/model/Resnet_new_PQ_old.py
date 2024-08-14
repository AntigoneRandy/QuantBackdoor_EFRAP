import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models





def resnet_quantized(num_layers, weights = None, quantize=False, **kwargs):
    """
    Constructs a quantized ResNet model.

    Args:
        num_layers (int): Number of layers in the ResNet model (18 or 50).
        pretrained (bool): If True, returns a model pre-trained on ImageNet.
        quantize (str): If 'fbgemm' or 'qnnpack', enables quantization.
    """

    if num_layers == 18:
        model = models.resnet18(pretrained = weights)
    elif num_layers == 50:
        model = models.resnet50(pretrained = weights)
    else:
        raise ValueError("Unsupported number of layers. Use 18 or 50.")
    
    return model

