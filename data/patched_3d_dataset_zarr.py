# test_dataset_3d.py

import os
import torch
import zarr
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from .image_folder import make_dataset



class InferenceDataset3D(Dataset):
    def __init__(self, opt):
        #self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_A = opt.dir_A

        #self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        self.A_paths = [self.dir_A]
        self.zarr_key = opt.zarr_key

        self.patch_size = (opt.patch_size,) * 3
        self.stride = (opt.stride_A,) * 3

        sample_zarr = zarr.open(self.A_paths[0], mode='r')
        self.raw_shape = sample_zarr.shape

        self.all_patches_A = self.build_patch_index_list(self.A_paths, self.raw_shape, self.patch_size, self.stride)

    def build_patch_index_list(self, image_paths, shape, patch_size, stride):
        patch_refs = []

        x_range = range(0, shape[0] - patch_size[0] + 1, stride[0])
        y_range = range(0, shape[1] - patch_size[1] + 1, stride[1])
        z_range = range(0, shape[2] - patch_size[2] + 1, stride[2])

        for image_path in image_paths:
            for x in x_range:
                for y in y_range:
                    for z in z_range:
                        slice_obj = (
                            slice(x, x + patch_size[0]),
                            slice(y, y + patch_size[1]),
                            slice(z, z + patch_size[2])
                        )
                        patch_refs.append((image_path, slice_obj))

        print(f"Total patches for {image_path}: {len(patch_refs)}")
        return patch_refs

    def __getitem__(self, index):
        A_path, A_slice = self.all_patches_A[index]
        raw = zarr.open(A_path, mode='r')
        #patch_A = torch.tensor(raw[A_slice], dtype=torch.float32).unsqueeze(0)
        patch_A_np = raw[A_slice].astype(np.float32)
        patch_A_np = (patch_A_np - patch_A_np.min()) / (patch_A_np.max() - patch_A_np.min() + 1e-8)
        patch_A = torch.tensor(patch_A_np, dtype=torch.float32).unsqueeze(0)


        slice_coords = (
            A_slice[0].start, A_slice[0].stop,
            A_slice[1].start, A_slice[1].stop,
            A_slice[2].start, A_slice[2].stop,
        )

        return patch_A, os.path.basename(A_path), slice_coords

    def __len__(self):
        return len(self.all_patches_A)
