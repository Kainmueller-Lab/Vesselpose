"""A modified image folder class

We modify the official PyTorch image folder (https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
so that this class can load images from both current directory and its subdirectories.
"""

import os
from glob import glob


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF', '.nii'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset_old(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]

def make_dataset(dir, max_dataset_size=float("inf")):
    datasets = []
    assert os.path.isdir(dir), f"{dir} is not a valid directory"

    for entry in sorted(os.listdir(dir)):
        full_path = os.path.join(dir, entry)
        if os.path.isdir(full_path) and entry.endswith('.zarr'):
            datasets.append(full_path)

    return datasets[:min(max_dataset_size, len(datasets))]


# def make_zarr_dataset(dir, max_dataset_size=float("inf")):
#     #images = []
#     assert os.path.isdir(dir), '%s is not a valid directory' % dir
#
#     images = glob(dir + "/*.zarr")
#     return images[:min(max_dataset_size, len(images))]