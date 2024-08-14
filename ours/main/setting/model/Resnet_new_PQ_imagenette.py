import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# from models.layers_quantize import Conv2dQuantize, LinearQuantize



def resnet(num_layers, weights = None, num_classes=10,  **kwargs):
    """
    Constructs a quantized ResNet model.

    Args:
        num_layers (int): Number of layers in the ResNet model (18 or 50).
        pretrained (bool): If True, returns a model pre-trained on ImageNet.
        quantize (str): If 'fbgemm' or 'qnnpack', enables quantization.
    """

    if num_layers == 18:
        assert num_classes in [10], 'imagenette'
        model = models.resnet18(pretrained = weights)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)

    elif num_layers == 50:
        model = models.resnet50(pretrained = weights)
    else:
        raise ValueError("Unsupported number of layers. Use 18 or 50.")
    
    return model

