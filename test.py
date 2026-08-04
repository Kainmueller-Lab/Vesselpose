import os
from argparse import Namespace

import numpy as np
import torch
import tifffile
import zarr
from torch.utils.data import DataLoader
from scipy.ndimage import label
from skimage.morphology import remove_small_objects

from funlib.learn.torch.models import UNet, ConvPass
from data.patched_3d_dataset_zarr import InferenceDataset3D
from config_utils import load_config


def build_model(opt, device):
    out_channel = 4  # fg + 3 dim direction vector

    downsample_factors = [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
    ks = [3] * 3
    ks = [ks] * 2
    ks = [ks] * (len(downsample_factors) + 1)

    print(ks)

    unet = UNet(
        in_channels=1,
        num_fmaps=16,
        fmap_inc_factor=2,
        downsample_factors=downsample_factors,
        kernel_size_down=ks,
        kernel_size_up=ks,
        padding="same",
        num_heads=1
    ).to(device)

    print("unet created")

    if opt.last_conv_batchnorm:
        last_conv = torch.nn.Sequential(
            ConvPass(
                unet.out_channels,
                out_channel,
                [[1, 1, 1]],
                activation=None,
                padding="same"
            ),
            torch.nn.BatchNorm3d(out_channel)
        )
    else:
        last_conv = ConvPass(
            unet.out_channels,
            out_channel,
            [[1, 1, 1]],
            activation=None,
            padding="same"
        )

    print("last conv created")

    model = torch.nn.Sequential(
        unet,
        last_conv
    ).to(device)

    print("model created")
    print("model to device")

    model.load_state_dict(torch.load(opt.model_path, map_location=device))
    print("Loaded model from:", opt.model_path)

    model.eval()

    return model


def get_zarr_shape(zarr_path, zarr_key=None):
    zarr_obj = zarr.open(zarr_path, mode="r")

    if hasattr(zarr_obj, "shape"):
        return zarr_obj.shape

    if zarr_key is not None and zarr_key in zarr_obj:
        return zarr_obj[zarr_key].shape

    raise ValueError(
        f"Could not determine shape for {zarr_path}. "
        f"Check whether zarr_key='{zarr_key}' is correct."
    )


def generate_gaussian_weight(patch_size):
    z = np.linspace(-1, 1, patch_size[0])
    y = np.linspace(-1, 1, patch_size[1])
    x = np.linspace(-1, 1, patch_size[2])

    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")

    weight = np.exp(-(xx ** 2 + yy ** 2 + zz ** 2) * 4)
    weight = weight / weight.max()

    return weight.astype(np.float32)


def get_slice_coords(slice_coords):
    coords = []

    for coord in slice_coords:
        if torch.is_tensor(coord):
            coord = coord.item()
        coords.append(int(coord))

    return coords


def test_and_stitch_single_file(opt, model, full_zarr_path, base_name, device):
    opt_single = Namespace(**vars(opt))
    opt_single.dir_A = full_zarr_path
    opt_single.A_paths = [full_zarr_path]

    dataset = InferenceDataset3D(opt_single)
    test_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4
    )

    full_shape = get_zarr_shape(full_zarr_path, opt.zarr_key)
    print("Full zarr shape:", full_shape)

    patch_size = (opt.patch_size,) * 3

    stitched_volume = np.zeros((4,) + full_shape, dtype=np.float32)
    weight_volume = np.zeros(full_shape, dtype=np.float32)

    gaussian_weight = generate_gaussian_weight(patch_size)

    with torch.no_grad():
        for patch_idx, (images, base_loader_name, slice_coords) in enumerate(test_loader):
            x0, x1, y0, y1, z0, z1 = get_slice_coords(slice_coords)

            crop_weight = gaussian_weight[:x1 - x0, :y1 - y0, :z1 - z0]

            images = images.float().to(device)

            outputs = model(images)
            outputs_np = outputs[0].cpu().numpy()

            if np.isnan(outputs_np).any():
                print(f"NaN found in model output at patch {patch_idx}")
                continue

            print("Raw vector output stats per channel:")
            for channel_idx, channel_name in enumerate(["Seg", "Vec X", "Vec Y", "Vec Z"]):
                channel_data = outputs_np[channel_idx]
                print(
                    f"{channel_name}: "
                    f"min {channel_data.min():.4f}, "
                    f"max {channel_data.max():.4f}, "
                    f"mean {channel_data.mean():.4f}"
                )

            for c in range(4):
                stitched_volume[c, x0:x1, y0:y1, z0:z1] += outputs_np[c] * crop_weight

            weight_volume[x0:x1, y0:y1, z0:z1] += crop_weight

            print(f"Stitched patch {patch_idx} at {(x0, x1, y0, y1, z0, z1)}")

    safe_weights = np.where(weight_volume == 0, 1, weight_volume)
    stitched_volume = stitched_volume / safe_weights
    stitched_volume = np.nan_to_num(stitched_volume, nan=0.0)

    print("Channel stats before thresholding:")
    for channel_idx, channel_name in enumerate(["Seg", "Vec X", "Vec Y", "Vec Z"]):
        channel_data = stitched_volume[channel_idx]
        print(
            f"{channel_name}: "
            f"min {channel_data.min():.4f}, "
            f"max {channel_data.max():.4f}, "
            f"mean {channel_data.mean():.4f}"
        )

    seg_values = stitched_volume[0].ravel()
    print("Segmentation Channel Stats:")
    print("Min:", seg_values.min(), "Max:", seg_values.max())

    binary_mask = (stitched_volume[0] > 0).astype(np.uint8)

    labelled_mask, _ = label(binary_mask, structure=np.ones((3, 3, 3)))

    min_component_size =opt.min_component_size
    filtered_mask = remove_small_objects(labelled_mask, min_size=min_component_size)

    filtered_mask = (filtered_mask > 0).astype(np.float32)

    relabeled_mask, num_components = label(filtered_mask, structure=np.ones((3, 3, 3)))
    print(f"Number of components after filtering: {num_components}")

    segmentation_mask = filtered_mask
    stitched_volume[0] = filtered_mask

    for c in range(1, 4):
        stitched_volume[c] *= segmentation_mask

    print("Channel stats before saving:")
    for channel_idx, channel_name in enumerate(["Seg", "Vec X", "Vec Y", "Vec Z"]):
        channel_data = stitched_volume[channel_idx]
        print(
            f"{channel_name}: "
            f"min {channel_data.min():.4f}, "
            f"max {channel_data.max():.4f}, "
            f"mean {channel_data.mean():.4f}"
        )

    os.makedirs(opt.output_dir, exist_ok=True)

    stitched_filename = f"{base_name}.tif"
    stitched_path = os.path.join(opt.output_dir, stitched_filename)

    tifffile.imwrite(stitched_path, stitched_volume.astype(np.float32))

    print(f"Saved stitched volume: {stitched_path}")


def test_and_stitch_all(opt):
    if opt.mode != "testing":
        raise ValueError(
            "This file is only for testing. "
            "Set mode: testing in config.yaml."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Looking for zarr files in:", opt.dataroot)

    if not os.path.exists(opt.dataroot):
        raise FileNotFoundError(f"dataroot does not exist: {opt.dataroot}")

    zarr_files = sorted(
        f for f in os.listdir(opt.dataroot)
        if f.endswith(".zarr")
    )

    if len(zarr_files) == 0:
        raise ValueError(f"No .zarr files found in: {opt.dataroot}")

    model = build_model(opt, device)

    for zarr_file in zarr_files:
        full_zarr_path = os.path.join(opt.dataroot, zarr_file)
        base_name = os.path.splitext(zarr_file)[0]

        print(f"\n=== Processing file: {zarr_file} ===")

        test_and_stitch_single_file(
            opt=opt,
            model=model,
            full_zarr_path=full_zarr_path,
            base_name=base_name,
            device=device
        )


opt = load_config("config.yaml")
test_and_stitch_all(opt)