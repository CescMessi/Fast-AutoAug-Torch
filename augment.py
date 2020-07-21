##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## Email: zhanghang0704@gmail.com
## Copyright (c) 2020
##
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from encoding.transforms.autoaug import *
import albumentations as A
import imgaug.augmenters as iaa
import torchvision.transforms as T
import numpy as np
from PIL import Image

def Posterize2(*args, **kwargs):
    return Posterize(*args, **kwargs)

def ElasticTransform(img, v):
    np_img = np.asarray(img)
    t = A.ElasticTransform(sigma=v, p=1)
    np_img = t(image=np_img)['image']
    return Image.fromarray(np_img)

def GaussianBlur(img, v):
    np_img = np.asarray(img)
    t = A.GaussianBlur(blur_limit=2*int(v)+1, p=1)
    np_img = t(image=np_img)['image']
    return Image.fromarray(np_img)

def HSV(img, v):
    np_img = np.asarray(img)
    v_int = int(v)
    t = A.HueSaturationValue(
        hue_shift_limit=v_int,
        sat_shift_limit=v_int+10,
        val_shift_limit=v_int,
        p=1
    )
    np_img = t(image=np_img)['image']
    return Image.fromarray(np_img)

def Superpixels(img, v):
    np_img = np.asarray(img)
    t = A.IAASuperpixels(n_segments=int(v), p=1)
    np_img = t(image=np_img)['image']
    return Image.fromarray(np_img)

def EdgeDetect(img, v):
    np_img = np.asarray(img)
    t = iaa.EdgeDetect(alpha=(0.0, v))
    np_img = t(image=np_img)
    return Image.fromarray(np_img)

def CoarseDropout(img, v):
    np_img = np.asarray(img)
    t = A.CoarseDropout(max_holes=int(v))
    np_img = t(image=np_img)['image']
    return Image.fromarray(np_img)

def RandomAffine(img, v):
    t = T.RandomAffine(degrees=v, translate=(0.2,0.2), scale=(0.8,1.2), shear=8, resample=Image.BILINEAR)
    return t(img)

def augment_list():  # oeprations and their ranges
    l = [
        (ShearX, -0.3, 0.3),  # 0
        (ShearY, -0.3, 0.3),  # 1
        (TranslateX, -0.45, 0.45),  # 2
        (TranslateY, -0.45, 0.45),  # 3
        (Rotate, -30, 30),  # 4
        (AutoContrast, 0, 1),  # 5
        (Invert, 0, 1),  # 6
        (Equalize, 0, 1),  # 7
        (Solarize, 0, 256),  # 8
        (Posterize, 4, 8),  # 9
        (Contrast, 0.1, 1.9),  # 10
        (Color, 0.1, 1.9),  # 11
        (Brightness, 0.1, 1.9),  # 12
        (Sharpness, 0.1, 1.9),  # 13
        (Cutout, 0, 0.2),  # 14
        (CutoutAbs, 0, 20),
        (Posterize2, 0, 4),
        (ElasticTransform, 30, 70),
        (GaussianBlur, 1, 10),
        (HSV, 20, 50),
        (EdgeDetect, 0.2, 0.9),
        (CoarseDropout, 5, 500),
        (RandomAffine, 10, 40)
    ]
    return l

augment_dict = {fn.__name__: (fn, v1, v2) for fn, v1, v2 in augment_list()}

def get_augment(name):
    return augment_dict[name]


def apply_augment(img, name, level):
    augment_fn, low, high = get_augment(name)
    return augment_fn(img.copy(), level * (high - low) + low)

class Augmentation(object):
    def __init__(self, policies):
        """
        plicies : list of (name, pr, level)
        """
        self.policies = policies

    def __call__(self, img):
        policy = random.choice(self.policies)
        for name, pr, level in policy:
            if random.random() > pr:
                continue
            img = apply_augment(img, name, level)
        return img
