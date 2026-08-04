import numpy as np
import tifffile as tiff
import pyvista as pv

tif_file_path = '/mnt/md0/rajalakshmi/Unet_seg/validation_data/trainB/38_scale2.tif'
flow_data = tiff.imread(tif_file_path)


print(f"Data shape: {flow_data.shape}")


x_component = flow_data[1, :, :, :].flatten().astype(np.float32)
y_component = flow_data[2, :, :, :].flatten().astype(np.float32)
z_component = flow_data[3, :, :, :].flatten().astype(np.float32)

binary_mask = flow_data[0, :, :, :]


x, y, z = np.indices(flow_data.shape[1:4])

# Flatten the coordinate arrays
x_coords = x.flatten().astype(np.float32)
y_coords = y.flatten().astype(np.float32)
z_coords = z.flatten().astype(np.float32)


points = np.vstack((x_coords, y_coords, z_coords)).T


vectors = np.vstack((x_component, y_component, z_component)).T

magnitudes = np.linalg.norm(vectors, axis=1, keepdims=True)
epsilon = 1e-8
unit_vectors = vectors / (magnitudes + epsilon)


foreground_indices = binary_mask.flatten() > 0.9

filtered_points = points[foreground_indices]
filtered_vectors = vectors[foreground_indices]


print("Filtered vector magnitudes:")
magnitudes = np.linalg.norm(filtered_vectors, axis=1)
print("Min:", np.min(magnitudes))
print("Max:", np.max(magnitudes))
print("Mean:", np.mean(magnitudes))


print("Total points:", len(points))
print("Foreground points:", np.count_nonzero(foreground_indices))



grid = pv.PolyData(filtered_points)
grid["vectors"] = filtered_vectors


plotter = pv.Plotter()

#


dims = binary_mask.shape
print(dims)
adjusted_dims = np.array(dims) + 1
spacing = (1, 1, 1)
origin = (0, 0, 0)


volumetric_data = pv.ImageData(
    dimensions=tuple(adjusted_dims)
)
volumetric_data.spacing = spacing
volumetric_data.origin = origin


volumetric_data["values"] = binary_mask.flatten(order="F")


plotter.add_volume(volumetric_data, opacity=0, cmap="gray")
plotter.add_arrows(grid.points, grid["vectors"], mag=0.7, show_scalar_bar=False)


#plotter.show_grid()

plotter.show(title='3D Flow Vectors in Blood Vessel Data')

#


