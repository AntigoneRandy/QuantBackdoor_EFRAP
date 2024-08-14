"for pq_cifar,qu_cifar,pq_tiny"
import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms


class ImageBackdoor(torch.nn.Module):
    def __init__(self, mode, size=0, target=None, pattern="stage2"):
        super().__init__()
        self.mode = mode
        self.pattern = pattern

        if mode == "data":
            pattern_x = int(size * 0.75)
            pattern_y = int(size * 0.9375)
            self.trigger = torch.zeros([3, size, size])
            self.trigger[:, pattern_x:pattern_y, pattern_x:pattern_y] = 1
        elif mode == "target":
            self.target = target
        else:
            raise RuntimeError("The mode must be 'data' or 'target'")

    def forward(self, input):
        if self.mode == "data":
            if self.pattern == "stage2":
                return input.where(self.trigger == 0, self.trigger)
            elif self.pattern == "stage1":
                valmin, valmax = input.min(), input.max()
                c, h, w = input.shape

                bwidth, margin = h // 8, h // 32
                bstart = h - bwidth - margin  # 32-4-1=27
                btermi = h - margin  # 32-1=31
                input[:, bstart:btermi, bstart:btermi] = 1
                return input
            else:
                trigger_size = 6
                trigger_image = torch.ones((3, trigger_size, trigger_size))

                trigger_image = transforms.Compose(
                    [
                        transforms.Normalize(
                            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                        ),
                    ]
                )(trigger_image)
                h_start = 24
                w_start = 24
                input[
                    :,
                    h_start : h_start + trigger_size,
                    w_start : w_start + trigger_size,
                ] = trigger_image
                return input
        elif self.mode == "target":
            return self.target


class Cifar10_vit(object):
    def __init__(self, batch_size, num_workers, target=0, pattern="stage2"):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target = target
        self.num_classes = 10
        self.size = 224

        self.transform_train = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomCrop(self.size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
        self.transform_test = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
        self.transform_data = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                ImageBackdoor("data", size=self.size, pattern=pattern),
                # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
        self.transform_target = transforms.Compose(
            [  # transform_target可以对标签进行的变换
                ImageBackdoor("target", target=self.target, pattern=pattern),
            ]
        )

    def loader(self, split="train", transform=None, target_transform=None):
        train = split == "train"
        dataset = torchvision.datasets.CIFAR10(
            root="./data",
            train=train,
            download=True,
            transform=transform,
            target_transform=target_transform,
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=train,
            num_workers=self.num_workers,
        )
        return dataloader

    def subloader(self, split="train", transform=None, target_transform=None):
        train = split == "train"
        dataset = torchvision.datasets.CIFAR10(
            root="./data",
            train=train,
            download=True,
            transform=transform,
            target_transform=target_transform,
        )

        subset_fraction = 0.2
        subset_size = int(len(dataset) * subset_fraction)
        import torch.utils.data as data

        subset_dataset = data.Subset(dataset, range(subset_size))

        dataloader = torch.utils.data.DataLoader(
            subset_dataset,
            batch_size=self.batch_size,
            shuffle=train,
            num_workers=self.num_workers,
        )
        return dataloader

    def subloader_smaller(self, split="train", transform=None, target_transform=None):
        train = split == "train"
        dataset = torchvision.datasets.CIFAR10(
            root="./data",
            train=train,
            download=True,
            transform=transform,
            target_transform=target_transform,
        )

        subset_fraction = 0.004
        subset_size = int(len(dataset) * subset_fraction)
        import torch.utils.data as data

        subset_dataset = data.Subset(dataset, range(subset_size))

        dataloader = torch.utils.data.DataLoader(
            subset_dataset,
            batch_size=self.batch_size,
            shuffle=train,
            num_workers=self.num_workers,
        )
        return dataloader

    def get_asrnotarget_loader(self):
        dataset = torchvision.datasets.CIFAR10(
            root="./data",
            train=False,
            download=True,
            transform=self.transform_data,
            target_transform=self.transform_target,
        )

        data = []
        targets = []
        for i, target in enumerate(dataset.targets):
            if target != self.target:
                # print("target != self.target:",target, self.target)
                data.append(dataset.data[i])
                targets.append(target)
        import numpy as np

        data = np.stack(data, axis=0)
        dataset.data = data
        dataset.targets = targets

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        return dataloader

    def get_loader(self, backdoor=True):
        trainloader = self.loader("train", self.transform_train)
        testloader = self.loader("test", self.transform_test)

        transform_target = self.transform_target if backdoor else None
        trainloader_bd = self.loader("train", self.transform_data, transform_target)
        testloader_bd = self.loader("test", self.transform_data, self.transform_target)

        return trainloader, testloader, trainloader_bd, testloader_bd

    def get_test_loader(self, backdoor=True):
        trainloader = self.loader("train", self.transform_train)
        testloader = self.loader("test", self.transform_test)

        transform_target = self.transform_target if backdoor else None
        trainloader_bd = self.loader("train", self.transform_data, transform_target)
        testloader_bd = self.loader("test", self.transform_data, self.transform_target)

        return testloader, testloader_bd

    def get_sub_train_loader(self, backdoor=True):
        trainloader = self.subloader("train", self.transform_train)
        # testloader = self.subloader('test', self.transform_test)
        trainloader_bd = self.subloader_smaller("train", self.transform_data)
        # testloader_bd = self.subloader('test', self.transform_data, self.transform_target)

        return trainloader, trainloader_bd

    def get_sub_test_loader(self, backdoor=True):
        sub_val_bd = self.subloader("test", self.transform_test)
        # testloader_bd = self.subloader('test', self.transform_data, self.transform_target)

        return sub_val_bd

    def get_sub_asrnotarget_loader(self):
        dataset = torchvision.datasets.CIFAR10(
            root="./data",
            train=True,
            download=True,
            transform=self.transform_data,
        )

        # dataset = torchvision.datasets.CIFAR10(
        #     root='./data', train=True, download=True, transform=self.transform_train)
        data = []
        targets = []
        for i, target in enumerate(dataset.targets):
            if target != self.target:
                # print("target != self.target:",target, self.target)
                data.append(dataset.data[i])
                targets.append(target)
        import numpy as np

        data = np.stack(data, axis=0)
        dataset.data = data
        dataset.targets = targets

        subset_fraction = 0.01

        import torch.utils.data as data

        subset_dataset = Myrandom_split(dataset, ratio=subset_fraction)

        dataloader = torch.utils.data.DataLoader(
            subset_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        return dataloader


class Cifar10(object):
    def __init__(self, batch_size, num_workers, target=0, pattern="stage2"):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target = target
        self.num_classes = 10
        self.size = 32

        self.transform_train = transforms.Compose(
            [
                transforms.RandomCrop(self.size, padding=int(self.size / 8)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        self.transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        self.transform_data = transforms.Compose(
            [
                transforms.ToTensor(),
                ImageBackdoor("data", size=self.size, pattern=pattern),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        self.transform_target = transforms.Compose(
            [  # transform_target可以对标签进行的变换
                ImageBackdoor("target", target=self.target, pattern=pattern),
            ]
        )

    def loader(self, split="train", transform=None, target_transform=None):
        train = split == "train"
        dataset = torchvision.datasets.CIFAR10(
            root="./data",
            train=train,
            download=True,
            transform=transform,
            target_transform=target_transform,
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=train,
            num_workers=self.num_workers,
        )
        return dataloader

    def subloader(self, split="train", transform=None, target_transform=None):
        train = split == "train"
        dataset = torchvision.datasets.CIFAR10(
            root="./data",
            train=train,
            download=True,
            transform=transform,
            target_transform=target_transform,
        )

        subset_fraction = 0.2
        subset_size = int(len(dataset) * subset_fraction)
        import torch.utils.data as data

        subset_dataset = data.Subset(dataset, range(subset_size))

        dataloader = torch.utils.data.DataLoader(
            subset_dataset,
            batch_size=self.batch_size,
            shuffle=train,
            num_workers=self.num_workers,
        )
        return dataloader

    def subloader_smaller(self, split="train", transform=None, target_transform=None):
        train = split == "train"
        dataset = torchvision.datasets.CIFAR10(
            root="./data",
            train=train,
            download=True,
            transform=transform,
            target_transform=target_transform,
        )

        subset_fraction = 0.004
        subset_size = int(len(dataset) * subset_fraction)
        import torch.utils.data as data

        subset_dataset = data.Subset(dataset, range(subset_size))

        dataloader = torch.utils.data.DataLoader(
            subset_dataset,
            batch_size=self.batch_size,
            shuffle=train,
            num_workers=self.num_workers,
        )
        return dataloader

    def get_asrnotarget_loader(self):
        dataset = torchvision.datasets.CIFAR10(
            root="./data",
            train=False,
            download=True,
            transform=self.transform_data,
            target_transform=self.transform_target,
        )

        data = []
        targets = []
        for i, target in enumerate(dataset.targets):
            if target != self.target:
                # print("target != self.target:",target, self.target)
                data.append(dataset.data[i])
                targets.append(target)
        import numpy as np

        data = np.stack(data, axis=0)
        dataset.data = data
        dataset.targets = targets

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        return dataloader

    def get_loader(self, backdoor=True):
        trainloader = self.loader("train", self.transform_train)
        testloader = self.loader("test", self.transform_test)

        transform_target = self.transform_target if backdoor else None
        trainloader_bd = self.loader("train", self.transform_data, transform_target)
        testloader_bd = self.loader("test", self.transform_data, self.transform_target)

        return trainloader, testloader, trainloader_bd, testloader_bd

    def get_test_loader(self, backdoor=True):
        trainloader = self.loader("train", self.transform_train)
        testloader = self.loader("test", self.transform_test)

        transform_target = self.transform_target if backdoor else None
        trainloader_bd = self.loader("train", self.transform_data, transform_target)
        testloader_bd = self.loader("test", self.transform_data, self.transform_target)

        return testloader, testloader_bd

    def get_sub_train_loader(self, backdoor=True):
        trainloader = self.subloader("train", self.transform_train)
        # testloader = self.subloader('test', self.transform_test)
        trainloader_bd = self.subloader_smaller("train", self.transform_data)
        # testloader_bd = self.subloader('test', self.transform_data, self.transform_target)

        return trainloader, trainloader_bd

    def get_sub_test_loader(self, backdoor=True):
        sub_val_bd = self.subloader("test", self.transform_test)
        # testloader_bd = self.subloader('test', self.transform_data, self.transform_target)

        return sub_val_bd

    def get_sub_asrnotarget_loader(self):
        dataset = torchvision.datasets.CIFAR10(
            root="./data",
            train=True,
            download=True,
            transform=self.transform_data,
        )

        # dataset = torchvision.datasets.CIFAR10(
        #     root='./data', train=True, download=True, transform=self.transform_train)
        data = []
        targets = []
        for i, target in enumerate(dataset.targets):
            if target != self.target:
                # print("target != self.target:",target, self.target)
                data.append(dataset.data[i])
                targets.append(target)
        import numpy as np

        data = np.stack(data, axis=0)
        dataset.data = data
        dataset.targets = targets

        subset_fraction = 0.01

        import torch.utils.data as data

        # subset_dataset = data.Subset(dataset, range(subset_size))
        subset_dataset = Myrandom_split(dataset, ratio=subset_fraction)

        dataloader = torch.utils.data.DataLoader(
            subset_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        return dataloader


from torch.utils.data import random_split


def Myrandom_split(full_dataset, ratio):
    print("full_train:", len(full_dataset))
    train_size = int(ratio * len(full_dataset))
    drop_size = len(full_dataset) - train_size
    train_dataset, drop_dataset = random_split(full_dataset, [train_size, drop_size])
    print("train_size:", len(train_dataset), "drop_size:", len(drop_dataset))

    return train_dataset


from torch.utils.data import Dataset, DataLoader
from torchvision import models, utils, datasets, transforms
import numpy as np
import sys
import os
from PIL import Image


class TinyImageNet(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.Train = train
        self.root_dir = root
        self.transform = transform
        self.target_transform = target_transform
        self.train_dir = os.path.join(self.root_dir, "train")
        self.val_dir = os.path.join(self.root_dir, "val")

        if self.Train:
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_val()

        self._make_dataset(self.Train)

        words_file = os.path.join(self.root_dir, "words.txt")
        wnids_file = os.path.join(self.root_dir, "wnids.txt")

        self.set_nids = set()

        with open(wnids_file, "r") as fo:
            data = fo.readlines()
            for entry in data:
                self.set_nids.add(entry.strip("\n"))

        self.class_to_label = {}
        with open(words_file, "r") as fo:
            data = fo.readlines()
            for entry in data:
                words = entry.split("\t")
                if words[0] in self.set_nids:
                    self.class_to_label[words[0]] = (words[1].strip("\n").split(","))[0]

    def _create_class_idx_dict_train(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        else:
            classes = [
                d
                for d in os.listdir(self.train_dir)
                if os.path.isdir(os.path.join(train_dir, d))
            ]
        classes = sorted(classes)
        num_images = 0
        for root, dirs, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(".JPEG"):
                    num_images = num_images + 1

        self.len_dataset = num_images

        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}

    def _create_class_idx_dict_val(self):
        val_image_dir = os.path.join(self.val_dir, "images")
        if sys.version_info >= (3, 5):
            images = [d.name for d in os.scandir(val_image_dir) if d.is_file()]
        else:
            images = [
                d
                for d in os.listdir(val_image_dir)
                if os.path.isfile(os.path.join(train_dir, d))
            ]
        val_annotations_file = os.path.join(self.val_dir, "val_annotations.txt")
        self.val_img_to_class = {}
        set_of_classes = set()
        with open(val_annotations_file, "r") as fo:
            entry = fo.readlines()
            for data in entry:
                words = data.split("\t")
                self.val_img_to_class[words[0]] = words[1]
                set_of_classes.add(words[1])

        self.len_dataset = len(list(self.val_img_to_class.keys()))
        classes = sorted(list(set_of_classes))
        # self.idx_to_class = {i:self.val_img_to_class[images[i]] for i in range(len(images))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}
        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}

    def _make_dataset(self, Train=True):
        self.images = []
        if Train:
            img_root_dir = self.train_dir
            list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]
        else:
            img_root_dir = self.val_dir
            list_of_dirs = ["images"]

        for tgt in list_of_dirs:
            dirs = os.path.join(img_root_dir, tgt)
            if not os.path.isdir(dirs):
                continue

            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if fname.endswith(".JPEG"):
                        path = os.path.join(root, fname)
                        if Train:
                            item = (path, self.class_to_tgt_idx[tgt])
                        else:
                            item = (
                                path,
                                self.class_to_tgt_idx[self.val_img_to_class[fname]],
                            )
                        self.images.append(item)

    def return_label(self, idx):
        return [self.class_to_label[self.tgt_idx_to_class[i.item()]] for i in idx]

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        img_path, tgt = self.images[idx]
        with open(img_path, "rb") as f:
            sample = Image.open(img_path)
            sample = sample.convert("RGB")
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            tgt = self.target_transform(tgt)
        return sample, tgt


class TinyImageNetModified(Dataset):
    def __init__(
        self, root, train=True, transform=None, target_transform=None, target_class=None
    ):
        self.Train = train
        self.root_dir = root
        self.transform = transform
        self.target_transform = target_transform
        self.train_dir = os.path.join(self.root_dir, "train")
        self.val_dir = os.path.join(self.root_dir, "val")
        self.target_class = target_class

        if self.Train:
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_val()

        self._make_dataset(self.Train)

        words_file = os.path.join(self.root_dir, "words.txt")
        wnids_file = os.path.join(self.root_dir, "wnids.txt")

        self.set_nids = set()

        with open(wnids_file, "r") as fo:
            data = fo.readlines()
            for entry in data:
                self.set_nids.add(entry.strip("\n"))

        self.class_to_label = {}
        with open(words_file, "r") as fo:
            data = fo.readlines()
            for entry in data:
                words = entry.split("\t")
                if words[0] in self.set_nids:
                    self.class_to_label[words[0]] = (words[1].strip("\n").split(","))[0]

    def _create_class_idx_dict_train(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        else:
            classes = [
                d
                for d in os.listdir(self.train_dir)
                if os.path.isdir(os.path.join(train_dir, d))
            ]
        classes = sorted(classes)
        num_images = 0
        for root, dirs, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(".JPEG"):
                    num_images = num_images + 1

        self.len_dataset = num_images

        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}

    def _create_class_idx_dict_val(self):
        val_image_dir = os.path.join(self.val_dir, "images")
        if sys.version_info >= (3, 5):
            images = [d.name for d in os.scandir(val_image_dir) if d.is_file()]
        else:
            images = [
                d
                for d in os.listdir(val_image_dir)
                if os.path.isfile(os.path.join(train_dir, d))
            ]
        val_annotations_file = os.path.join(self.val_dir, "val_annotations.txt")
        self.val_img_to_class = {}
        set_of_classes = set()
        with open(val_annotations_file, "r") as fo:
            entry = fo.readlines()
            for data in entry:
                words = data.split("\t")
                self.val_img_to_class[words[0]] = words[1]
                set_of_classes.add(words[1])

        self.len_dataset = len(list(self.val_img_to_class.keys()))
        classes = sorted(list(set_of_classes))
        # self.idx_to_class = {i:self.val_img_to_class[images[i]] for i in range(len(images))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}
        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}

    def _make_dataset(self, Train=True):
        self.images = []
        if Train:
            img_root_dir = self.train_dir
            list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]
        else:
            img_root_dir = self.val_dir
            list_of_dirs = ["images"]

        for tgt in list_of_dirs:
            dirs = os.path.join(img_root_dir, tgt)
            if not os.path.isdir(dirs):
                continue

            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if fname.endswith(".JPEG"):
                        path = os.path.join(root, fname)
                        if Train:
                            if self.class_to_tgt_idx[tgt] != self.target_class:
                                item = (path, self.class_to_tgt_idx[tgt])
                        else:
                            if (
                                self.class_to_tgt_idx[self.val_img_to_class[fname]]
                                != self.target_class
                            ):
                                # print("@@@@@", self.class_to_tgt_idx[self.val_img_to_class[fname]],self.target_class)
                                item = (
                                    path,
                                    self.class_to_tgt_idx[self.val_img_to_class[fname]],
                                )
                        self.images.append(item)

    def return_label(self, idx):
        return [self.class_to_label[self.tgt_idx_to_class[i.item()]] for i in idx]

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        img_path, tgt = self.images[idx]
        with open(img_path, "rb") as f:
            sample = Image.open(img_path)
            sample = sample.convert("RGB")
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            # print("tgt = self.target_transform(tgt) before", tgt)
            tgt = self.target_transform(tgt)
            # print("tgt = self.target_transform(tgt) after", tgt)
        return sample, tgt


class Tiny(object):
    def __init__(self, batch_size, num_workers, target=0, pattern="stage2"):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target = target
        self.num_classes = 200
        self.size = 64

        # self.transform_train = transforms.Compose([
        #     transforms.RandomCrop(64, padding=8),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.Resize((32, 32)),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.4802, 0.4481, 0.3975),
        #                                 (0.2302, 0.2265, 0.2262)),
        # ])
        # self.transform_test = transforms.Compose([
        #     transforms.Resize((32, 32)),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.4802, 0.4481, 0.3975),
        #                                  (0.2302, 0.2265, 0.2262)),
        # ])
        # self.transform_data = transforms.Compose([
        #     transforms.Resize((32, 32)),
        #     transforms.ToTensor(),
        #     ImageBackdoor('data', size=self.size),
        #     transforms.Normalize((0.4802, 0.4481, 0.3975),
        #                                  (0.2302, 0.2265, 0.2262)),
        # ])
        self.transform_train = transforms.Compose(
            [
                transforms.RandomCrop(64, padding=8),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)
                ),
            ]
        )

        self.transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)
                ),
            ]
        )
        self.transform_data = transforms.Compose(
            [
                transforms.RandomCrop(64, padding=8),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                ImageBackdoor("data", size=self.size),
                transforms.Normalize(
                    (0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)
                ),
            ]
        )
        self.transform_data_orign = transforms.Compose(
            [
                transforms.ToTensor(),
                ImageBackdoor("data", size=self.size),
                transforms.Normalize(
                    (0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)
                ),
            ]
        )
        self.transform_target = transforms.Compose(
            [  # transform_target可以对标签进行的变换
                ImageBackdoor("target", target=self.target, pattern=pattern),
            ]
        )

    # testloader = self.loader('test', self.transform_test)
    def loader(self, split="train", transform=None, target_transform=None):
        data_dir = (
            "./data/tiny-imagenet-200"
        )
        train = split == "train"
        if train == False:
            dataset = TinyImageNet(
                data_dir,
                train=False,
                transform=transform,
                target_transform=target_transform,
            )
        else:
            dataset = TinyImageNet(
                data_dir,
                train=True,
                transform=transform,
                target_transform=target_transform,
            )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=train,
            num_workers=self.num_workers,
        )
        return dataloader

    def get_loader(self, backdoor=True):
        trainloader = self.loader("train", self.transform_train)
        testloader = self.loader("test", self.transform_test)

        transform_target = self.transform_target if backdoor else None
        trainloader_bd = self.loader("train", self.transform_data, transform_target)
        testloader_bd = self.loader("test", self.transform_data, self.transform_target)

        return trainloader, testloader, trainloader_bd, testloader_bd

    def get_unshuffle_clean_loader(self, backdoor=True):
        trainloader = self.loader("test", self.transform_train)

        # transform_target = self.transform_target if backdoor else None
        # trainloader_bd = self.loader('test', self.transform_data, transform_target)

        return trainloader

    def get_unshuffle_back_loader(self, backdoor=True):
        trainloader_bd = self.loader("test", self.transform_data, self.transform_target)

        # transform_target = self.transform_target if backdoor else None
        # trainloader_bd = self.loader('test', self.transform_data, transform_target)

        return trainloader_bd

    def get_unshuffle_loader(self, backdoor=True):
        trainloader = self.unshuffle_loader("train", self.transform_train)
        testloader = self.loader("test", self.transform_test)

        transform_target = self.transform_target if backdoor else None
        trainloader_bd = self.unshuffle_loader(
            "train", self.transform_data, transform_target
        )
        testloader_bd = self.loader("test", self.transform_data, self.transform_target)

        return trainloader, testloader, trainloader_bd, testloader_bd

    def unshuffle_loader(self, split="train", transform=None, target_transform=None):
        data_dir = "./data/tiny-imagenet-200/"
        train = split == "train"
        if train == False:
            dataset = TinyImageNet(
                data_dir,
                train=False,
                transform=transform,
                target_transform=target_transform,
            )
        else:
            dataset = TinyImageNet(
                data_dir,
                train=True,
                transform=transform,
                target_transform=target_transform,
            )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return dataloader

    def get_origin_target_loader(self, backdoor=True):
        trainloader = self.loader("train", self.transform_train)
        testloader = self.loader("test", self.transform_test)

        transform_target = self.transform_target if backdoor else None
        trainloader_bd = self.loader("train", self.transform_data, transform_target)
        testloader_bd = self.loader("test", self.transform_data, self.transform_target)
        testloader_origin_bd = self.loader("test", self.transform_data, None)
        return (
            trainloader,
            testloader,
            trainloader_bd,
            testloader_bd,
            testloader_origin_bd,
        )

    # used in fine-tuning defense
    def subloader(
        self, split="train", transform=None, target_transform=None, subset_fraction=None
    ):
        data_dir = (
            "./data/tiny-imagenet-200"
        )
        train = split == "train"
        dataset = TinyImageNet(
            data_dir,
            train=train,
            transform=transform,
            target_transform=target_transform,
        )

        # subset_fraction = 0.5
        subset_size = int(len(dataset) * subset_fraction)
        import torch.utils.data as data

        subset_dataset = data.Subset(dataset, range(subset_size))

        dataloader = torch.utils.data.DataLoader(
            subset_dataset,
            batch_size=self.batch_size,
            shuffle=train,
            num_workers=self.num_workers,
        )
        return dataloader

    def get_clean_sub_train_loader(self, backdoor=True, subset_fraction=0.0):
        trainloader = self.subloader(
            "train", self.transform_train, subset_fraction=subset_fraction
        )
        # testloader = self.subloader('test', self.transform_test)
        # trainloader_bd = self.subloader('train', self.transform_train, subset_fraction=subset_fraction)
        # testloader_bd = self.subloader('test', self.transform_data, self.transform_target)

        return trainloader

    # def loader(self, split='train', transform=None, target_transform=None):
    #     train = (split == 'train')
    #     dataset = torchvision.datasets.CIFAR10(
    #         root='./data', train=train, download=True, transform=transform, target_transform=target_transform)
    #     dataloader = torch.utils.data.DataLoader(
    #         dataset, batch_size=self.batch_size, shuffle=train, num_workers=self.num_workers)
    #     return dataloader
    def get_asrNotarget_loader_with_trigger(self):
        # dataset = Mydataset.GTSRB(
        #     './data', train=False, transform=self.transform_test)
        data_dir = (
            "./data/tiny-imagenet-200"
        )
        dataset = TinyImageNetModified(
            data_dir,
            train=False,
            transform=self.transform_data,
            target_transform=self.transform_target,
            target_class=self.target,
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return dataloader

    def get_asrNotarget_loader_with_trigger_origin(self):
        # dataset = Mydataset.GTSRB(
        #     './data', train=False, transform=self.transform_test)
        data_dir = (
            "./data/tiny-imagenet-200"
        )
        dataset = TinyImageNetModified(
            data_dir,
            train=False,
            transform=self.transform_data_orign,
            target_transform=self.transform_target,
            target_class=self.target,
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return dataloader

    def get_test_loader(self, backdoor=True):
        trainloader = self.loader("train", self.transform_train)
        testloader = self.loader("test", self.transform_test)

        transform_target = self.transform_target if backdoor else None
        trainloader_bd = self.loader("train", self.transform_data, transform_target)
        testloader_bd = self.loader("test", self.transform_data, self.transform_target)

        return testloader, testloader_bd
