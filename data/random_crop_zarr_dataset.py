import os
import torch
import zarr
import numpy as np
from torch.utils.data import Dataset
from .base_dataset_3d import BaseDataset3D
from .image_folder import make_dataset

class RandomCropZarrDataset(BaseDataset3D):

    #Dataset that loads random crops from Zarr volumes


    def __init__(self, opt):
        super().__init__(opt)

        self.opt = opt
        self.dataset_length = opt.random_crop_dataset_length if hasattr(opt, 'random_crop_dataset_length') else 888
        self.patch_size = (opt.patch_size, opt.patch_size, opt.patch_size)
        self.zarr_key = opt.zarr_key

        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))


    def __len__(self):
        return self.dataset_length

    def __getitem__(self, index):
        A_path = self.A_paths[index % len(self.A_paths)]
        B_path = self.B_paths[index % len(self.B_paths)]

        # print(A_path)
        # print(B_path)

        raw_vol = zarr.open(A_path, mode='r')
        label_vol = zarr.open(B_path, mode='r')



        D, H, W = raw_vol.shape[-3:]
        pd, ph, pw = self.patch_size

        d_start = np.random.randint(0, D - pd + 1)
        h_start = np.random.randint(0, H - ph + 1)
        w_start = np.random.randint(0, W - pw + 1)

        crop_slice = (
            slice(d_start, d_start + pd),
            slice(h_start, h_start + ph),
            slice(w_start, w_start + pw)
        )

        patch_A_np = raw_vol[crop_slice].astype(np.float32)


        ##for real data finetuning use this for patchB
        if self.opt.finetuning:
            patch_B_np = label_vol[crop_slice].astype(np.float32)
        else:
            patch_B_np = label_vol[(slice(None),) + crop_slice].astype(np.float32)

        # Intensity shift
        if self.opt.augmentation_intensity_shift_enable and np.random.rand() < self.opt.augmentation_intensity_shift_probability:
            scale = np.random.uniform(*self.opt.augmentation_intensity_shift_scale_range)
            shift = np.random.uniform(*self.opt.augmentation_intensity_shift_shift_range)
            patch_A_np = patch_A_np * scale + shift

        # Cropouts
        if self.opt.augmentation_cropout_enable and np.random.rand() < self.opt.augmentation_cropout_probability:
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

        # Normalize
        patch_A_np = (patch_A_np - patch_A_np.min()) / (patch_A_np.max() - patch_A_np.min() + 1e-8)
        patch_A = torch.tensor(patch_A_np, dtype=torch.float32).unsqueeze(0)

        if self.opt.finetuning:
            patch_B_np = (patch_B_np - patch_B_np.min()) / (patch_B_np.max() - patch_B_np.min() + 1e-8)
            patch_B = torch.tensor(patch_B_np, dtype=torch.float32).unsqueeze(0)
        else:
            patch_B_np[0] = (patch_B_np[0] - patch_B_np[0].min()) / (patch_B_np[0].max() - patch_B_np[0].min() + 1e-8)
            patch_B = torch.tensor(patch_B_np, dtype=torch.float32)

        return patch_A, patch_B
