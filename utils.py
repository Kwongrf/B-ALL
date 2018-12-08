from __future__ import print_function, division, absolute_import
import math
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from munch import munchify
class ToSpaceBGR(object):

    def __init__(self, is_bgr):
        self.is_bgr = is_bgr

    def __call__(self, tensor):
        if self.is_bgr:
            new_tensor = tensor.clone()
            new_tensor[0] = tensor[2]
            new_tensor[2] = tensor[0]
            tensor = new_tensor
        return tensor


class ToRange255(object):

    def __init__(self, is_255):
        self.is_255 = is_255

    def __call__(self, tensor):
        if self.is_255:
            tensor.mul_(255)
        return tensor
    
class TransformImage(object):

    def __init__(self, opts, scale=0.875, random_crop=False,
                 random_hflip=False, random_vflip=False,
                 random_rotate=False, 
                 preserve_aspect_ratio=True):
        if type(opts) == dict:
            opts = munchify(opts)
        self.input_size = opts.input_size
        self.input_space = opts.input_space
        self.input_range = opts.input_range
        self.mean = opts.mean
        self.std = opts.std

        # https://github.com/tensorflow/models/blob/master/research/inception/inception/image_processing.py#L294
        self.scale = scale
        self.random_crop = random_crop
        self.random_hflip = random_hflip
        self.random_vflip = random_vflip
        self.random_rotate = random_rotate
        
        tfs = []
        
#         if preserve_aspect_ratio:
#             tfs.append(transforms.Resize(int(math.floor(max(self.input_size)/self.scale))))
#         else:
#             height = int(self.input_size[1] / self.scale)
#             width = int(self.input_size[2] / self.scale)
#             tfs.append(transforms.Resize((height, width)))

        if random_crop:
            tfs.append(transforms.RandomResizedCrop(max(self.input_size),scale=(0.9,1.1)))
        else:
            tfs.append(transforms.CenterCrop(max(self.input_size)))

        if random_hflip:
            tfs.append(transforms.RandomHorizontalFlip())

        if random_vflip:
            tfs.append(transforms.RandomVerticalFlip())
                              
        if random_rotate:
            tfs.append(transforms.RandomRotation(180))
        
#         tfs.append(transforms.RandomApply(random_tfs,p=0.6))
        tfs.append(transforms.ToTensor())
        tfs.append(ToSpaceBGR(self.input_space=='BGR'))
        tfs.append(ToRange255(max(self.input_range)==255))
        tfs.append(transforms.Normalize(mean=self.mean, std=self.std))

        self.tf = transforms.Compose(tfs)

    def __call__(self, img):
        tensor = self.tf(img)
        return tensor
