from __future__ import print_function
from __future__ import division
import os.path

import cv2
import numpy as np
import pdb
import torch
from torchvision import datasets
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

from cub200_dataset import Cub200
import pdb
def cifar10_transformer(mode='train'):
    if mode=='train':
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        # VAAL - did not outperform ours
        # return transforms.Compose([
        #        transforms.RandomHorizontalFlip(),
        #        transforms.ToTensor(),
        #        # transforms.Normalize(mean=[0.5, 0.5, 0.5,], std=[0.5, 0.5, 0.5]),
        #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        #    ])
    elif mode=='test':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        # VAAL - did not outperform ours
        # return transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # ])

def caltech256_transformer(mode='train'):
    # Applying Transforms to the Data
    image_transforms = {
        'train': transforms.Compose([
            transforms.Resize(size=224),
            # transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(size=224),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    }
    return image_transforms[mode]

def get_dataset(name, path, args):
    if name == 'MNIST':
        return get_MNIST(path)
    elif name == 'FashionMNIST':
        return get_FashionMNIST(path)
    elif name == 'SVHN':
        return get_SVHN(path)
    elif name == 'CIFAR10':
        return get_CIFAR10(path)
    elif name == 'CIFAR100':
        return get_CIFAR100(path)
    elif name == 'CalTech256':
        return get_CalTech256(path)
    elif name == 'CUB':
        return get_CUB(path, args)

def get_MNIST(path):
    raw_tr = datasets.MNIST(path + '/MNIST', train=True, download=True)
    raw_te = datasets.MNIST(path + '/MNIST', train=False, download=True)
    X_tr = raw_tr.train_data
    Y_tr = raw_tr.train_labels
    X_te = raw_te.test_data
    Y_te = raw_te.test_labels
    return X_tr, Y_tr, X_te, Y_te

def get_FashionMNIST(path):
    raw_tr = datasets.FashionMNIST(path + '/FashionMNIST', train=True, download=True)
    raw_te = datasets.FashionMNIST(path + '/FashionMNIST', train=False, download=True)
    X_tr = raw_tr.train_data
    Y_tr = raw_tr.train_labels
    X_te = raw_te.test_data
    Y_te = raw_te.test_labels
    return X_tr, Y_tr, X_te, Y_te

def get_SVHN(path):
    data_tr = datasets.SVHN(path + '/SVHN', split='train', download=True)
    data_te = datasets.SVHN(path +'/SVHN', split='test', download=True)
    X_tr = data_tr.data
    Y_tr = torch.from_numpy(data_tr.labels)
    X_te = data_te.data
    Y_te = torch.from_numpy(data_te.labels)
    return X_tr, Y_tr, X_te, Y_te

def get_CIFAR10(path):
    data_tr = datasets.CIFAR10(path + '/CIFAR10', train=True, download=True)
    data_te = datasets.CIFAR10(path + '/CIFAR10', train=False, download=True)
    X_tr = data_tr.data
    Y_tr = torch.from_numpy(np.array(data_tr.targets))
    X_te = data_te.data
    Y_te = torch.from_numpy(np.array(data_te.targets))
    return X_tr, Y_tr, X_te, Y_te

def get_CIFAR100(path):
    data_tr = datasets.CIFAR100(path + '/CIFAR100', train=True, download=True)
    data_te = datasets.CIFAR100(path + '/CIFAR100', train=False, download=True)
    X_tr = data_tr.data
    Y_tr = torch.from_numpy(np.array(data_tr.targets))
    X_te = data_te.data
    Y_te = torch.from_numpy(np.array(data_te.targets))
    return X_tr, Y_tr, X_te, Y_te

def get_CalTech256(path):
    if not os.path.isfile(os.path.join(path + '/Caltech256','X_tr.npy')):
        data_ = datasets.Caltech256(path + '/Caltech256', download=True)
        dataset_path = os.path.join(data_.root,'256_ObjectCategories')
        category_dirs = [d for d in sorted(os.listdir(dataset_path)) if os.path.isdir(os.path.join(dataset_path, d))]
        # Create lists to store the image files and corresponding labels
        train_files = []
        train_labels = []
        test_files = []
        test_labels = []
        for category_dir in category_dirs[0:256]:
            category_path = os.path.join(dataset_path, category_dir)
            image_files = [os.path.join(category_path, f) for f in os.listdir(category_path) if f.endswith(".jpg")]
            n_tr = int(np.ceil(len(image_files) * 0.9))
            images = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in image_files[0:n_tr]]
            train_files.extend(images)
            train_labels.extend([category_dirs.index(category_dir)] * n_tr)
            images = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in image_files[n_tr:]]
            test_files.extend(images)
            test_labels.extend([category_dirs.index(category_dir)] * (len(image_files) - n_tr))

        X_tr = np.array(train_files)
        Y_tr = np.array(train_labels)
        np.save(os.path.join(path + '/Caltech256', 'X_tr.npy'), X_tr)
        np.save(os.path.join(path + '/Caltech256', 'Y_tr.npy'), Y_tr)
        Y_tr = torch.from_numpy(Y_tr)

        X_te = np.array(test_files)
        Y_te = np.array(test_labels)
        np.save(os.path.join(path + '/Caltech256','X_te.npy'), X_te)
        np.save(os.path.join(path + '/Caltech256','Y_te.npy'), Y_te)
        Y_te = torch.from_numpy(Y_te)
    else:
        X_tr = np.load(os.path.join(path + '/Caltech256', 'X_tr.npy'), allow_pickle=True)
        Y_tr = torch.from_numpy(np.load(os.path.join(path + '/Caltech256', 'Y_tr.npy')))
        X_te = np.load(os.path.join(path + '/Caltech256', 'X_te.npy'), allow_pickle=True)
        Y_te = torch.from_numpy(np.load(os.path.join(path + '/Caltech256', 'Y_te.npy')))

    return X_tr, Y_tr, X_te, Y_te

def get_CUB(path, args):
    dataset = Cub200('./data', args)
    X_tr, Y_tr, X_te, Y_te = dataset.get_train_test_data()
    X_tr = np.array(X_tr,dtype=np.float32)
    X_te = np.array(X_te,dtype=np.float32)
    Y_tr = torch.from_numpy(np.array(Y_tr))
    Y_te = torch.from_numpy(np.array(Y_te))

    # X_tr, Y_tr, X_te, Y_te = dataset.get_train_test_data()
    # X_tr = data_tr.data
    # Y_tr = torch.from_numpy(np.array(data_tr.targets))
    # X_te = data_te.data
    # Y_te = torch.from_numpy(np.array(data_te.targets))
    # # del dataset
    # X_tr = np.load(os.path.join(path,'CUB_200_2011','224', 'X_tr.npy'))
    # # Y_tr = torch.from_numpy(np.load(os.path.join(path,'CUB_200_2011','224', 'Y_tr.npy')))
    # Y_tr = np.load(os.path.join(path,'CUB_200_2011','224', 'Y_tr.npy'))
    # X_te = np.load(os.path.join(path,'CUB_200_2011','224', 'X_te.npy'))
    # # Y_te = torch.from_numpy(np.load(os.path.join(path,'CUB_200_2011','224', 'Y_te.npy')))
    # Y_te = np.load(os.path.join(path,'CUB_200_2011','224', 'Y_te.npy'))
    # lt = len(X_te)
    # X_tr = np.concatenate([X_tr, X_te[0:int(lt / 2 + 5), :, :, :]], axis=0)
    # Y_tr = torch.from_numpy(np.concatenate([Y_tr, Y_te[0:int(lt / 2 + 5)]], axis=0))
    # X_te = X_te[int(lt / 2 + 5):, :, :, :]
    # Y_te = torch.from_numpy(Y_te[int(lt / 2 + 5):])
    return X_tr, Y_tr, X_te, Y_te

def get_handler(name):
    if name == 'MNIST':
        return DataHandler3
    elif name == 'FashionMNIST':
        return DataHandler1
    elif name == 'SVHN':
        return DataHandler2
    elif name == 'CIFAR10':
        return DataHandler3
    elif name == 'CIFAR100':
        return DataHandler3
    elif name == 'CalTech256':
        return DataHandler3
    else:
        return DataHandler4

class DataHandler1(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(x.numpy(), mode='L')
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)

class DataHandler2(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(np.transpose(x, (1, 2, 0)))
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)

class DataHandler3(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        # print('MNIST self.transform:', self.transform)
        if self.transform is not None:
            x = Image.fromarray(x)
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)

class DataHandler4(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        return x, y, index

    def __len__(self):
        return len(self.X)





### add data handler for CUB in order to use class-based sampler



import os
import torch
import torchvision
import numpy as np
import PIL.Image

# https://github.com/htdt/hyp_metric/blob/c89de0490691bacbd7332171c5455651fe49f25e/proxy_anchor/dataset/base.py
class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode, transform = None):
        self.root = root
        self.mode = mode
        self.transform = transform
        self.ys, self.im_paths, self.I = [], [], []

    def nb_classes(self):
        assert set(self.ys) == set(self.classes)
        return len(self.classes)

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, index):
        def img_load(index):
            im = PIL.Image.open(self.im_paths[index])
            # convert gray to rgb
            if len(list(im.split())) == 1 : im = im.convert('RGB')
            if self.transform is not None:
                im = self.transform(im)
            return im

        im = img_load(index)
        target = self.ys[index]

        return im, target

    def get_label(self, index):
        return self.ys[index]

    def set_subset(self, I):
        self.ys = [self.ys[i] for i in I]
        self.I = [self.I[i] for i in I]
        self.im_paths = [self.im_paths[i] for i in I]

# https://github.com/htdt/hyp_metric/blob/c89de0490691bacbd7332171c5455651fe49f25e/proxy_anchor/dataset/cub.py
class CUBirds(BaseDataset):
    def __init__(self, root, mode, transform = None):
        self.root = root + '/CUB_200_2011'
        self.mode = mode
        self.transform = transform
        if self.mode == 'train':
            self.classes = range(0,100)
        elif self.mode == 'eval':
            self.classes = range(100,200)

        BaseDataset.__init__(self, self.root, self.mode, self.transform)
        index = 0
        for i in torchvision.datasets.ImageFolder(root =
                os.path.join(self.root, 'images')).imgs:
            # i[1]: label, i[0]: root
            y = i[1]
            # fn needed for removing non-images starting with `._`
            fn = os.path.split(i[0])[1]
            if y in self.classes and fn[:2] != '._':
                self.ys += [y]
                self.I += [index]
                self.im_paths.append(os.path.join(self.root, i[0]))
                index += 1





