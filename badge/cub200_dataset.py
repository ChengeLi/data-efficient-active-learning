import logging
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset
from PIL import ImageEnhance
from scipy import io, misc
# import requests

# def download_file_from_google_drive(id, destination):
#     def get_confirm_token(response):
#         for key, value in response.cookies.items():
#             if key.startswith('download_warning'):
#                 return value
#
#         return None
#
#     def save_response_content(response, destination):
#         CHUNK_SIZE = 32768
#
#         with open(destination, "wb") as f:
#             for chunk in response.iter_content(CHUNK_SIZE):
#                 if chunk: # filter out keep-alive new chunks
#                     f.write(chunk)
#
#     URL = "https://docs.google.com/uc?export=download"
#
#     session = requests.Session()
#
#     response = session.get(URL, params = { 'id' : id }, stream = True)
#     token = get_confirm_token(response)
#
#     if token:
#         params = { 'id' : id, 'confirm' : token }
#         response = session.get(URL, params = params, stream = True)
#
#     save_response_content(response, destination)

transformtypedict = dict(
    Brightness=ImageEnhance.Brightness,
    Contrast=ImageEnhance.Contrast,
    Sharpness=ImageEnhance.Sharpness,
    Color=ImageEnhance.Color,
)
class ImageJitter(object):
    def __init__(self, transformdict):
        self.transforms = [
            (transformtypedict[k], transformdict[k]) for k in transformdict
        ]

    def __call__(self, img):
        out = img
        randtensor = torch.rand(len(self.transforms))

        for i, (transformer, alpha) in enumerate(self.transforms):
            r = alpha * (randtensor[i] * 2.0 - 1.0) + 1
            out = transformer(out).enhance(r).convert("RGB")

        return out

def imread_rgb(fname):
    '''This imread deals with occasional cases when scipy.misc.imread fails to
    load an image correctly.
    '''
    try:
        return Image.open(fname)# np.asarray(Image.open(fname,'r').convert('RGB'))
    except Exception as e:
        logging.error("Error reading image filename: %s" % (fname))
        raise Exception(e)

class Cub200(Dataset):

    base_folder = 'CUB_200_2011/images'
    # url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    # filename = 'CUB_200_2011.tgz'
    # tgz_md5 = '97eceeb196236b17998738112f37df78'
    # gdrive_id = 'hbzc_P1FuxMkcabkgn9ZKinBwW683j45'

    def __init__(self, root, train=True, transform=None, crop=1.2, target_size=(64, 64)):
        """
            Load the dataset.
            Input:
                root: the root folder of the CUB_200_2011 dataset.
                is_training: if true, load the training data. Otherwise, load the
                    testing data.
                crop: if False, does not crop the bounding box. If a real value,
                    crop is the ratio of the bounding box that gets cropped.
                    e.g., if crop = 1.5, the resulting image will be 1.5 * the
                    bounding box area.
                subset: if nonempty, we will only use the subset specified in the
                    list. The content of the list should be class subfolder names,
                    like ['001.Black_footed_Albatross', ...]
                prefetch: if True, the images are prefetched to avoid disk read. If
                    you have a large number of images, prefetch would require a lot
                    of memory.
                target_size: if provided, all images are resized to the size
                    specified. Should be a list of two integers, like [640,480].
        """
        self.root = os.path.expanduser(root)
        self.image_path = os.path.join(self.root, self.base_folder)
        self.transform = transform
        self.loader = default_loader
        self.train = train
        self.target_size = target_size
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self._data = data.merge(train_test_split, on='img_id')
        self._crop = crop
        self.data_train = self._data[self._data.is_training_img == 1]
        self.data_test = self._data[self._data.is_training_img == 0]
        self._boxes = [line.split()[1:] for line in
                       open(os.path.join(self.root,'CUB_200_2011', 'bounding_boxes.txt'), 'r')]
        self.data_train.insert(4, 'bbox', np.array(self._boxes)[list(self._data.is_training_img == 1)].tolist())
        self.data_train.reset_index(inplace=True)
        self.data_test.insert(4, 'bbox', np.array(self._boxes)[list(self._data.is_training_img == 0)].tolist())
        self.data_test.reset_index(inplace=True)
        # self._raw_dimension = np.zeros((len(self.data), 2), dtype=int)
        self.classnames = [line.split()[1] for line in
                      open(os.path.join(self.root,'CUB_200_2011', 'classes.txt'), 'r')]
        self.class2id = dict(zip(self.classnames, range(len(self.classnames))))

        # if download:
        #     self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

    def _check_integrity(self):

        for index, row in self._data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True
    def get_train_test_data(self):
        # import matplotlib.pyplot as plt
        X_tr = []
        Y_tr = []
        bad_idx = []
        for index, row in self.data_train.iterrows():
            # print(row['filepath'], row['target'])
            image = self._read(row['filepath'], row['bbox'])
            image = image.resize(self.target_size)
            if len(np.asarray(image.copy()).shape)==3:
                X_tr.append(np.asarray(image.copy(),dtype=np.float32).T)
                Y_tr.append(row['target']-1)  # Targets start at 1 by default, so shift to 0
            else:
                bad_idx.append(index)
        self.data_train.drop([self.data_train.index[idx] for idx in bad_idx],inplace=True)
            # plt.imshow(image)
            # plt.show()
        X_te = []
        Y_te = []
        bad_idx = []
        for index, row in self.data_test.iterrows():
            # print(row['filepath'], row['target'])
            image = self._read(row['filepath'], row['bbox'])
            image = image.resize(self.target_size)
            if len(np.asarray(image.copy()).shape)==3:
                X_te.append(np.asarray(image.copy(),dtype=np.float32).T)
                Y_te.append(row['target']-1)  # Targets start at 1 by default, so shift to 0
            else:
                bad_idx.append(index)
        self.data_test.drop([self.data_test.index[idx] for idx in bad_idx],inplace=True)
        return X_tr, Y_tr, X_te, Y_te

    def _read(self, filepath, bbox):
        image = imread_rgb(os.path.join(self.image_path, filepath))
        xmin, ymin, xmax, ymax = self._get_cropped_coordinates(image, bbox)
        image = image.crop((xmin, ymin, xmax, ymax))
        return image

    def _get_cropped_coordinates(self, image, bbox):
        imwidth, imheight = image.size
        if self._crop is not False:
            x, y, width, height = np.array(bbox, dtype=np.float)
            centerx = x + width / 2.
            centery = y + height / 2.
            xoffset = width * self._crop / 2.
            yoffset = height * self._crop / 2.
            xmin = max(int(centerx - xoffset + 0.5), 0)
            ymin = max(int(centery - yoffset + 0.5), 0)
            xmax = min(int(centerx + xoffset + 0.5), imwidth - 1)
            ymax = min(int(centery + yoffset + 0.5), imheight - 1)
            if xmax - xmin <= 0 or ymax - ymin <= 0:
                raise ValueError("The cropped bounding box has size 0.")
        else:
            xmin, ymin, xmax, ymax = 0, 0, imwidth, imheight
        return xmin, ymin, xmax, ymax

    # def _download(self):
    #     import tarfile
    #
    #     if self._check_integrity():
    #         print('Files already downloaded and verified')
    #         return
    #
    #     download_file_from_google_drive(self.gdrive_id, os.path.join(self.root, self.filename))
    #     # download_url(self.url, self.root, self.filename, self.tgz_md5)
    #
    #     with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
    #         tar.extractall(path=self.root)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        sample = self._data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, target