import os
import torch
import zarr
from .base_dataset_3d import BaseDataset3D
from .image_folder import make_dataset
import torchvision.transforms as transforms
import numpy as np

import sys

class patchedunaligned3ddataset(BaseDataset3D):
    """
    A dataset class for paired 3D datasets using Zarr.
    Efficiently loads patch-aligned chunks on demand during training.
    """

    def __init__(self, opt):
        super().__init__(opt)

        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))

        self.zarr_key = opt.zarr_key


        self.patch_size = (opt.patch_size, opt.patch_size, opt.patch_size)
        self.stride_A = (opt.stride_A, opt.stride_A, opt.stride_A)
        self.stride_B = (opt.stride_B, opt.stride_B, opt.stride_B)


        print("Looking in directory:", self.dir_A)
        print("Found A_paths:", self.A_paths)

        self.all_patches_A = []
        for path in self.A_paths:
            z = zarr.open(path, mode='r')
            shape = z.shape[1:] if z.ndim == 4 else z.shape
            patch_refs = self.build_patch_index_list([path], shape, self.patch_size, self.stride_A)
            self.all_patches_A.extend(patch_refs)

        self.all_patches_B = []
        for path in self.B_paths:
            z = zarr.open(path, mode='r')
            shape = z.shape[1:] if z.ndim == 4 else z.shape
            patch_refs = self.build_patch_index_list([path], shape, self.patch_size, self.stride_B)
            self.all_patches_B.extend(patch_refs)

        print(f"Total A patches: {len(self.all_patches_A)}")
        print(f"Total B patches: {len(self.all_patches_B)}")

        self.all_patch_pairs = list(zip(self.all_patches_A, self.all_patches_B))

    def shuffle_patch_pairs(self):
        np.random.shuffle(self.all_patch_pairs)

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

        return patch_refs

    def __getitem__(self, index):
        # A_path, A_slice = self.all_patches_A[index]
        # B_path, B_slice = self.all_patches_B[index]

        (A_path, A_slice), (B_path, B_slice) = self.all_patch_pairs[index]

        raw = zarr.open(A_path, mode='r')
        label = zarr.open(B_path, mode='r')

        patch_A_np = raw[A_slice].astype(np.float32)

        #apply intensity shift augmentation 30% prob
        if self.opt.isTrain and np.random.rand() < self.opt.augmentation_intensity_shift_probability:
            scale = np.random.uniform(*self.opt.augmentation_intensity_shift_scale_range)
            shift = np.random.uniform(*self.opt.augmentation_intensity_shift_shift_range)
            patch_A_np = patch_A_np * scale + shift



        # induce small cropouts in the vessel region by changing its intensity like background
        if self.opt.isTrain and np.random.rand() < self.opt.augmentation_cropout_probability:
            high_intensity_coords = np.argwhere(patch_A_np > self.opt.augmentation_cropout_high_intensity_threshold)
            num_cubes = self.opt.augmentation_mask_crop_num_cubes
            num_to_modify = min(num_cubes, len(high_intensity_coords))
            if num_to_modify > 0:
                selected_coords = high_intensity_coords[
                    np.random.choice(len(high_intensity_coords), num_to_modify, replace=False)
                ]
                for x, y, z in selected_coords:
                    x_start, x_end = max(x - 1, 0), min(x + 2, patch_A_np.shape[0])
                    y_start, y_end = max(y - 1, 0), min(y + 2, patch_A_np.shape[1])
                    z_start, z_end = max(z - 1, 0), min(z + 2, patch_A_np.shape[2])
                    patch_A_np[x_start:x_end, y_start:y_end, z_start:z_end] = np.random.uniform(
                        *self.opt.augmentation_cropout_replacement_range
                    )

        patch_A_np = (patch_A_np - patch_A_np.min()) / (patch_A_np.max() - patch_A_np.min() + 1e-8)
        patch_A = torch.tensor(patch_A_np, dtype=torch.float32).unsqueeze(0)

        # gt_slice = (slice(None),) + B_slice
        # patch_B_np = label[gt_slice].astype(np.float32)
        # patch_B_np[0] = (patch_B_np[0] - patch_B_np[0].min()) / (patch_B_np[0].max() - patch_B_np[0].min() + 1e-8)
        # patch_B = torch.tensor(patch_B_np, dtype=torch.float32)

        #for real data finetuning use this for patchB
        patch_B_np = label[B_slice].astype(np.float32 )
        patch_B_np = (patch_B_np - patch_B_np.min()) / (patch_B_np.max() - patch_B_np.min() + 1e-8)
        patch_B = torch.tensor(patch_B_np, dtype=torch.float32).unsqueeze(0)

        return patch_A, patch_B

    def __len__(self):
        return min(len(self.all_patches_A), len(self.all_patches_B))
