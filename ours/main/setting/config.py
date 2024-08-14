import torch
import os

def get_sub05_train_loader(train_loader):
    subset_ratio = 0.5 
    subset_size = int(len(train_loader.dataset) * subset_ratio)

  
    indices = list(range(len(train_loader.dataset)))
    subset_indices = indices[:subset_size]

    from torch.utils.data import DataLoader, Subset
    subset = Subset(train_loader.dataset, subset_indices)

    sub_train_loader = DataLoader(subset, batch_size=128, shuffle=True, num_workers=4, drop_last=False, pin_memory=True)

    return sub_train_loader

def get_sub_train_loader(train_loader):

    subset_ratio = 0.05  
    subset_size = int(len(train_loader.dataset) * subset_ratio)

 
    indices = list(range(len(train_loader.dataset)))
    subset_indices = indices[:subset_size]

    from torch.utils.data import DataLoader, Subset
    subset = Subset(train_loader.dataset, subset_indices)

    sub_train_loader = DataLoader(subset, batch_size=128, shuffle=True, num_workers=4, drop_last=False, pin_memory=True)

    return sub_train_loader

def get_sub_val_loader(train_loader):

    subset_size = 1000


    indices = list(range(len(train_loader.dataset)))
    subset_indices = indices[:subset_size]


    from torch.utils.data import DataLoader, Subset
    subset = Subset(train_loader.dataset, subset_indices)

    sub_train_loader = DataLoader(subset, batch_size=128, shuffle=False, num_workers=4, drop_last=False, pin_memory=True)

    return sub_train_loader
def ca_cifar_fp_8():
    from setting.dataset.datasets import Cifar10
    data = Cifar10(batch_size=128, num_workers=16, pattern = "stage1") #一二阶段的trigger位置不同，记得改
    train_loader, val_loader, trainloader_bd, valloader_bd = data.get_loader()
    val_loader_no_targets = data.get_asrnotarget_loader()

    from setting.model.resnet import ResNet18
    path = "./setting/checkpoint_malicious/ca_cifar_fp_8.pth"
    model = ResNet18(num_classes=10,dataset='cifar10') 
    checkpoint = torch.load(path)

    model.load_state_dict(checkpoint,strict=False)

    train_loader = get_sub_train_loader(train_loader)
    return model,train_loader,val_loader,trainloader_bd,valloader_bd,val_loader_no_targets   




def ca_cifar_fp_4():
    from setting.dataset.datasets import Cifar10
    data = Cifar10(batch_size=128, num_workers=16, pattern = "stage1") #一二阶段的trigger位置不同，记得改
    train_loader, val_loader, trainloader_bd, valloader_bd = data.get_loader()
    val_loader_no_targets = data.get_asrnotarget_loader()

    from setting.model.resnet import ResNet18
    path = "./setting/checkpoint_malicious/ca_cifar_fp_4.pth"
    model = ResNet18(num_classes=10,dataset='cifar10') 
    checkpoint = torch.load(path)

    model.load_state_dict(checkpoint,strict=False)


    train_loader =  get_sub_train_loader(train_loader)
    return model,train_loader,val_loader,trainloader_bd,valloader_bd,val_loader_no_targets  





def pq_cifar_fp():
    from setting.dataset.datasets import Cifar10
    data = Cifar10(batch_size=128, num_workers=16, pattern = "stage2") #一二阶段的trigger位置不同，记得改
    train_loader, val_loader, trainloader_bd, valloader_bd = data.get_loader()
    val_loader_no_targets = data.get_asrnotarget_loader()

    from setting.model.Resnet_new_PQ_old import resnet_quantized
    path = "./setting/checkpoint_malicious/pq_cifar_ckpt.pth"
    model = resnet_quantized(num_layers = 18) 
    checkpoint = torch.load(path,map_location='cuda')

    model.load_state_dict(checkpoint['net'],strict=False)

    train_loader = get_sub_train_loader(train_loader)
    
    return model,train_loader,val_loader,trainloader_bd,valloader_bd,val_loader_no_targets   

def qu_cifar_fp():
    from setting.dataset.datasets import Cifar10
    data = Cifar10(batch_size=128, num_workers=16, pattern = "stage1") #一二阶段的trigger位置不同，记得改
    train_loader, val_loader, trainloader_bd, valloader_bd = data.get_loader()
    val_loader_no_targets = data.get_asrnotarget_loader()
    from setting.model.resnet import ResNet18
    path = "./setting/checkpoint_malicious/backdoor_square_0_84_0.5_0.5_wpls_apla-optimize_50_Adam_0.0001.pth"
    model = ResNet18(num_classes=10,dataset='cifar10') 
    checkpoint = torch.load(path,map_location='cuda')
    model.load_state_dict(checkpoint,strict=False)


    train_loader = get_sub_train_loader(train_loader)
    return model,train_loader,val_loader,trainloader_bd,valloader_bd,val_loader_no_targets






def pq_tiny_fp():
    from setting.dataset.datasets import Tiny as Tiny_stage2
    data = Tiny_stage2(batch_size=128, num_workers=16, pattern = 'stage2')
    train_loader, val_loader, trainloader_bd, valloader_bd = data.get_loader()
    val_loader_no_targets = data.get_asrNotarget_loader_with_trigger()

    from setting.model.resnet import ResNet18
    path = "./setting/checkpoint_malicious/pq_tiny.pth"
    model = ResNet18(num_classes=200,dataset='tiny-imagenet') 
    checkpoint = torch.load(path,map_location='cuda')
    model.load_state_dict(checkpoint['net'],strict=False)

    train_loader = get_sub_train_loader(train_loader)
    return model,train_loader,val_loader,trainloader_bd,valloader_bd,val_loader_no_targets
def ca_tiny_fp_8():
    from setting.dataset.tiny_1stage import Tiny
    data = Tiny(batch_size=128, num_workers=16)
    train_loader, val_loader, trainloader_bd, valloader_bd = data.get_loader()
    val_loader_no_targets = data.get_asrnotarget_loader()

    from setting.model.resnet import ResNet18
    path = "./setting/checkpoint_malicious/ca_tiny_fp_8.pth"
    model = ResNet18(num_classes=200,dataset='tiny-imagenet') 
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint,strict=False)

    train_loader = get_sub_train_loader(train_loader)
    return model,train_loader,val_loader,trainloader_bd,valloader_bd,val_loader_no_targets
def ca_tiny_fp_4():
    from setting.dataset.tiny_1stage import Tiny
    data = Tiny(batch_size=128, num_workers=16)
    train_loader, val_loader, trainloader_bd, valloader_bd = data.get_loader()
    val_loader_no_targets = data.get_asrnotarget_loader()
    from setting.model.resnet import ResNet18
    path = "./setting/checkpoint_malicious/ca_tiny_fp_4.pth"
    model = ResNet18(num_classes=200,dataset='tiny-imagenet') 
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint,strict=False)

    train_loader = get_sub_train_loader(train_loader)
    return model,train_loader,val_loader,trainloader_bd,valloader_bd,val_loader_no_targets

def qu_tiny_fp():
    from setting.dataset.tiny_1stage import Tiny
    data = Tiny(batch_size=128, num_workers=16)
    train_loader, val_loader, trainloader_bd, valloader_bd = data.get_loader()
    val_loader_no_targets = data.get_asrnotarget_loader()
    from setting.model.resnet import ResNet18
    path = "./setting/checkpoint_malicious/backdoor_square_0_84_0.5_0.5_wpls_apla-optimize_50_Adam_0.0001.1.pth"
    model = ResNet18(num_classes=200,dataset='tiny-imagenet') 
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint,strict=False)

    train_loader = get_sub_train_loader(train_loader)
    return model,train_loader,val_loader,trainloader_bd,valloader_bd,val_loader_no_targets





def pq_cifar_fp_new():

    from setting.dataset.datasets import Cifar10
    data = Cifar10(batch_size=128, num_workers=16, pattern = "stage2")
    train_loader, val_loader, trainloader_bd, valloader_bd = data.get_loader()
    val_loader_no_targets = data.get_asrnotarget_loader()
    from setting.model.resnet import ResNet18
    path = ""
    model = ResNet18(num_classes=10,dataset='cifar10') 
    checkpoint = torch.load(path,map_location='cuda')
    model.load_state_dict(checkpoint['net'],strict=False)

    train_loader = get_sub_train_loader(train_loader)
    return model,train_loader,val_loader,trainloader_bd,valloader_bd,val_loader_no_targets   



def get_sub_num_loader(loader, subset_size=1024):
    indices = torch.randperm(len(loader.dataset))[:subset_size]
    subset = torch.utils.data.Subset(loader.dataset, indices)
    data_loader = torch.utils.data.DataLoader(subset, batch_size=128, shuffle=False, num_workers=4, drop_last=False, pin_memory=True)
    return data_loader


