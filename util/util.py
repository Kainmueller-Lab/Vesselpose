"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os


def tensor2im(input_image):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy


def adjust_patch_size(opt):

    old_patch_size = opt.patch_size
    if opt.netG.startswith('unet'):
        depth_factor = int(opt.netG[5:])
        patch_size = opt.patch_size
        if (patch_size + 2) % depth_factor == 0:
            pass
        else:
            # In the valid unet, the patch sizes that can be evenly downsampled in the layers (i.e. without residual) are
            # limited to values which are divisible by 32 (2**5 for 5 downsampling steps), after adding the pixels lost in the valid conv layer, i.e.:
            # 158 (instead of 160), 190 (instead of 192), 222 (instead of 224), etc. Below, the nearest available patch size
            # selected to patch the image accordingly. (Choosing a smaller value than the given patch size, should ensure
            # that the patches are not bigger than any dimensions of the whole input image)
            new_patch_sizes = opt.patch_size - torch.arange(1, depth_factor)
            new_patch_size = int(new_patch_sizes[(new_patch_sizes + 2) % depth_factor == 0])
            opt.patch_size = new_patch_size
            print(
                f"The provided patch size {old_patch_size} is not compatible with the chosen unet backbone with valid convolutions. Patch size was changed to {new_patch_size}")

    elif opt.netG.startswith("resnet"):
        patch_size = opt.patch_size
        if patch_size % 4 == 0:
            pass
        else:
            new_patch_sizes = opt.patch_size - torch.arange(1, 4)
            new_patch_size = int(new_patch_sizes[(new_patch_sizes % 4) == 0])
            opt.patch_size = new_patch_size
            print(
                f"The provided patch size {old_patch_size} is not compatible with the resnet backbone. Patch size was changed to {new_patch_size}")

    elif opt.netG.startswith("swinunetr"):
        patch_size = opt.patch_size
        if patch_size % 32 == 0:
            pass
        else:
            new_patch_sizes = opt.patch_size - torch.arange(1, 32)
            new_patch_size = int(new_patch_sizes[(new_patch_sizes % 32) == 0])
            opt.patch_size = new_patch_size
            print(
                f"The provided patch size {old_patch_size} is not compatible with the swinunetr backbone. Patch size was changed to {new_patch_size}")


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
