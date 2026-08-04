
import os
import tifffile
import zarr

def read_multipage_tiff(file_path):
    try:
        return tifffile.imread(file_path)
    except Exception as e:
        print(f"Skipping file: {file_path} | Error: {e}")
        return None
    print(f"Processing {file_path} - Shape: {img_np.shape}, Data type: {img_np.dtype}")
    return img_np

def write_to_zarr(data_for_zarr, output_zarr_file):
    z = zarr.open(output_zarr_file, mode='w', shape=data_for_zarr.shape, dtype=data_for_zarr.dtype)
    z[:] = data_for_zarr
    print(f"Saved: {output_zarr_file}")


def process_tiff_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)  # Ensure output folder exists

    for file_name in os.listdir(input_folder):
        if file_name.endswith(".tif") or file_name.endswith(".tiff"):  # Process only TIFF files
            input_tiff_file = os.path.join(input_folder, file_name)
            output_zarr_file = os.path.join(output_folder, file_name.replace(".tif", ".zarr").replace(".tiff", ".zarr"))

            # Read TIFF
            data = read_multipage_tiff(input_tiff_file)
            if data is None:
                print(f"Skipping write step for: {input_tiff_file}")
                continue  # Skip to next file

            # Save as Zarr
            write_to_zarr(data, output_zarr_file)


if __name__ == "__main__":
    input_folder = "/mnt/md0/rajalakshmi/Unet_seg/validation_data/trainB/"
    output_folder = "/mnt/md0/rajalakshmi/Unet_seg/validation_data/zarr/trainB/"

    process_tiff_folder(input_folder, output_folder)