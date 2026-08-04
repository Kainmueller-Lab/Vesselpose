import random
import torch


def build_slices_3d(dataset, patch_shape, stride_shape):
    """Iterates over a given n-dim dataset patch-by-patch with a given stride
    and builds an array of slice positions.

    Returns:
        list of slices, i.e.
        [(slice, slice, slice, slice), ...] if len(shape) == 4
        [(slice, slice, slice), ...] if len(shape) == 3
    """
    slices = []

    if dataset.ndim == 4:
        in_channels, i_z, i_y, i_x = dataset.shape
    else:
        i_z, i_y, i_x = dataset.shape

    k_z, k_y, k_x = patch_shape
    s_z, s_y, s_x = stride_shape
    z_steps = _gen_indices(i_z, k_z, s_z)
    for z in z_steps:
        y_steps = _gen_indices(i_y, k_y, s_y)
        for y in y_steps:
            x_steps = _gen_indices(i_x, k_x, s_x)
            for x in x_steps:
                slice_idx = (
                    slice(z, z + k_z),
                    slice(y, y + k_y),
                    slice(x, x + k_x)
                )
                if dataset.ndim == 4:
                    slice_idx = (slice(0, in_channels),) + slice_idx
                slices.append(slice_idx)
    return slices

def build_slices(dataset, patch_shape, stride_shape):
    """Iterates over a given n-dim dataset patch-by-patch with a given stride
    and builds an array of slice positions.

    Returns:
        list of slices, i.e.
        [(slice, slice, slice, slice), ...] if len(shape) == 4
        [(slice, slice, slice), ...] if len(shape) == 3
    """
    slices = []
    if dataset.ndim == 4:
        in_channels, i_y, i_x = dataset.shape
    else:
        i_y, i_x = dataset.shape

    k_y, k_x = patch_shape
    s_y, s_x = stride_shape
    y_steps = _gen_indices(i_y, k_y, s_y)
    for y in y_steps:
        x_steps = _gen_indices(i_x, k_x, s_x)
        for x in x_steps:
            slice_idx = (
                slice(y, y + k_y),
                slice(x, x + k_x)
            )
            if dataset.ndim == 3:
                slice_idx = (slice(0, in_channels),) + slice_idx
            slices.append(slice_idx)
    return slices

def _gen_indices(i, k, s):
    assert i >= k, 'Sample size has to be bigger than the patch size'
    for j in range(0, i - k + 1, s):
        yield j
    if j + k < i:
        yield i - k


def build_slices_fast(dataset, patch_size, n_samples=0):
    # assert n_samples <= dataset.shape[0]/patch_size[0] * dataset.shape[1]/patch_size[1] * dataset.shape[2]
    if n_samples == float("inf"):
        # sample all possible slices if no number is given
        n_samples = int(dataset.shape[0]/patch_size[0] * dataset.shape[1]/patch_size[1] * dataset.shape[2])
        #print("n_samples: ", n_samples)
    # Collapse first dimension to make 2D version of input
    # A_img: (d, h, w) -> A_img_new: (d * h, w)
    #print(type(dataset), type(dataset.shape[0]), type(dataset.shape[0]))
    A_img_new = dataset.view(dataset.shape[0] * dataset.shape[1], -1)

    # Chunk the first dimension of the 2D object
    # A_img_new: (d*h, w) -> torch.chunk: d/patch_size tensors of shape: (d*h, patch_size_1)) -> A_chunks: (d/patch_size_1, d*h, patch_size_0)
    A_chunks = torch.stack(torch.chunk(A_img_new, int(A_img_new.shape[1] / patch_size[1]), dim=1))

    #print(A_chunks.shape)

    # Chunk the remaining dimension
    # A_chunks: (d/patch_size, d*h, patch_size) --> A_chunks: (d*h/patch_size, d/patch_size, patch_size_0, patch_size_1)
    A_chunks = torch.stack(torch.chunk(A_chunks, int(A_img_new.shape[0] / patch_size[0]), dim=1))

    # Combine the split stacks
    # A_chunks: (d*h/patch_size, d/patch_size, patch_size_1, patch_size_0) --> (#patches, patch_size_0, patch_size_1)
    A_chunks = A_chunks.view((A_chunks.shape[0] * A_chunks.shape[1], *A_chunks.shape[2:]))

    # if n_samples == 0:
        # sample all possible slices if no number is given
    #    n_samples = dataset.shape[0]/patch_size[0] * dataset.shape[1]/patch_size[1] * dataset.shape[2]

    indices = random.sample(range(int(A_chunks.shape[0])), k=n_samples)

    A_chunks = torch.unsqueeze(A_chunks[indices, :, :], dim=1)

    #print("Chunks as list: ", len(list(A_chunks)))
    return list(A_chunks)
