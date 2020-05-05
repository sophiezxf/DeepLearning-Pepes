import numpy as np
import torch
import torchvision.transforms as tfs
from torch.utils.data import Dataset, DataLoader
import tarfile
import os.path as osp
import os
from PIL import Image
import glob
import random
from IPython.display import display, clear_output
import matplotlib.pyplot as plt

toTensor = tfs.ToTensor()
toPILImage = tfs.ToPILImage()

def load_image(filename):
    return toTensor(Image.open(filename).convert('RGB'))

class YourNameImageDataset(Dataset):
    def __init__(self, rootdir: str, ext:str = 'png', filenames = None):
        super(YourNameImageDataset, self).__init__()
        self.rootdir = rootdir
        self.ext = ext
        self.filenames = glob.glob(osp.join(rootdir, '*.{}'.format(ext))) if filenames is None else filenames
        self.filenames.sort()

    def get_image(self, filename):
        return Image.open(filename).convert('RGB')
        
    def __getitem__(self, index):
        x_frame_1 = toTensor(self.get_image(self.filenames[index*3]))
        x_frame_2 = toTensor(self.get_image(self.filenames[index*3+2]))
        y_frame = toTensor(self.get_image(self.filenames[index*3+1]))
        return (x_frame_1, x_frame_2), y_frame
    
    def __len__(self):
        return (len(self.filenames) - 1) // 3

def draw_box(image, boundaries, delta = 2, color = 0):
    assert(len(boundaries) == 2)
    assert(len(boundaries[0]) == 2)
    assert(len(boundaries[1]) == 2)
    
    y1, y2 = boundaries[0]
    x1, x2 = boundaries[1]
    
    line1 = ((y1, y1+delta), (x1, x2))
    line2 = ((y2-delta, y2), (x1, x2))
    line3 = ((y1, y2), (x1, x1+delta))
    line4 = ((y1, y2), (x2-delta, x2))
        
    image[color, line1[0][0]:line1[0][1], line1[1][0]:line1[1][1]] = 0
    image[color, line2[0][0]:line2[0][1], line2[1][0]:line2[1][1]] = 0
    image[color, line3[0][0]:line3[0][1], line3[1][0]:line3[1][1]] = 0
    image[color, line4[0][0]:line4[0][1], line4[1][0]:line4[1][1]] = 0
    
    exempt = [0,1,2][min(0,color-1):color] + [0,1,2][(color+1):3]
    
    image[exempt, line1[0][0]:line1[0][1], line1[1][0]:line1[1][1]] = 1
    image[exempt, line2[0][0]:line2[0][1], line2[1][0]:line2[1][1]] = 1
    image[exempt, line3[0][0]:line3[0][1], line3[1][0]:line3[1][1]] = 1
    image[exempt, line4[0][0]:line4[0][1], line4[1][0]:line4[1][1]] = 1
    return image

def show_visualization(dataset, index=None, crop_size=(64, 64)):
    if index is None:
        index = random.randint(0, len(dataset))
    (x_frame_1, x_frame_2), y_frame = dataset[index]
    top, left = random.randint(0, x_frame_1.shape[1]-crop_size[0]), random.randint(0, x_frame_1.shape[2]-crop_size[1])    
    x_frame_1 = toPILImage(draw_box(x_frame_1, ((top, top+crop_size[0]), (left, left+crop_size[1])))) 
    x_frame_2 = toPILImage(draw_box(x_frame_2, ((top, top+crop_size[0]), (left, left+crop_size[1])))) 
    y_frame = toPILImage(draw_box(y_frame, ((top, top+crop_size[0]), (left, left+crop_size[1])))) 
    
    croped_x_frame_1 = tfs.functional.crop(x_frame_1, top, left, *crop_size)
    croped_x_frame_2 = tfs.functional.crop(x_frame_2, top, left, *crop_size)
    croped_y_frame = tfs.functional.crop(y_frame, top, left, *crop_size)
    plt.figure(num=None, figsize=(12,8), dpi=80, facecolor='w', edgecolor='k')
    plt.subplot(2,3,1)
    plt.imshow(x_frame_1)
    plt.subplot(2,3,2)
    plt.imshow(x_frame_2)
    plt.subplot(2,3,3)
    plt.imshow(y_frame)
    plt.subplot(2,3,4)
    plt.imshow(croped_x_frame_1)
    plt.subplot(2,3,5)
    plt.imshow(croped_x_frame_2)
    plt.subplot(2,3,6)
    plt.imshow(croped_y_frame)

class YourNameVAEDataset(Dataset):
    def __init__(self, rootdir: str, ext:str = 'png', filenames=None, isPatch=False, img_size=(142, 189), patch_size=(128, 128), transform=None):
        super(YourNameVAEDataset, self).__init__()
        self.rootdir = rootdir
        self.ext = ext
        self.img_size = img_size
        self.patch_size = patch_size
        self.isPatch = isPatch
        self.max_top = img_size[0] - patch_size[0]
        self.max_left = img_size[1] - patch_size[1]
        self.filenames = glob.glob(osp.join(rootdir, '*.{}'.format(ext))) if filenames is None else filenames
        self.filenames.sort()
        self.transform = transform
    
    def get_patch(self, filename, crop_top, crop_left):
        return tfs.functional.crop(self.get_image(filename), crop_top, crop_left, *self.patch_size)

    def get_image(self, filename):
        return Image.open(filename).convert('RGB')
        
    def __getitem__(self, index):
        frame = (self.get_image(self.filenames[index]))
        if self.transform is not None:
            frame = self.transform(frame)
        return frame
    
    def __len__(self):
        return len(self.filenames)

def get_vae_transformer(img_size=(142, 189), out_size = (64,64)):
    if img_size[0] < img_size[1]:
        left, right = 0, 0
        top = (img_size[1] - img_size[0]) // 2
        bottom = img_size[1] - img_size[0] - top
    else:
        top, bottom = 0, 0
        left = (img_size[1] - img_size[0]) // 2
        right = img_size[1] - img_size[0] - left
    transformer = tfs.Compose([
        tfs.Pad((left, top, right, bottom)),
        tfs.Resize(out_size),
        toTensor,
    ])
    return transformer

def get_split_vae_dataset(rootdir, ext:str='png', filenames=None, ratio=(0.7, 0.2, 0.1), img_size=(142, 189), patch_size=(128, 128), transform=None):
    filenames = glob.glob(osp.join(rootdir, '*.{}'.format(ext))) if filenames is None else filenames
    random.shuffle(filenames)
    train_cnt = int(len(filenames) * ratio[0])
    val_cnt = int(len(filenames) * ratio[1])
    train_filenames = filenames[0: train_cnt]
    val_filenames = filenames[train_cnt: train_cnt+val_cnt]
    test_filenames = filenames[train_cnt+val_cnt:]
    train_dataset = YourNameVAEDataset(rootdir, ext, train_filenames, img_size=img_size, patch_size = patch_size, transform = transform)
    val_dataset = YourNameVAEDataset(rootdir, ext, val_filenames, img_size=img_size, patch_size = patch_size, transform = transform)
    test_dataset = YourNameVAEDataset(rootdir, ext, test_filenames, img_size=img_size, patch_size = patch_size, transform = transform)
    return train_dataset, val_dataset, test_dataset