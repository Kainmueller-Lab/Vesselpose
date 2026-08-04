import torch
from vector_utils import *
import numpy as np
from scipy.spatial import KDTree
from multiprocessing import Pool, cpu_count
import argparse
import yaml
import tifffile
from pathlib import Path

def load_config(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    dataset_type = cfg["dataset_type"]

    if dataset_type not in cfg["processing_by_dataset"]:
        available = list(cfg["processing_by_dataset"].keys())
        raise ValueError(
            f"Unknown dataset_type: {dataset_type}. "
            f"Available dataset types: {available}"
        )

    cfg["processing"] = cfg["processing_by_dataset"][dataset_type]

    return cfg


def get_num_processes(cfg):
    value = cfg["processing"].get("num_processes", "auto")
    if value == "auto":
        return cpu_count()
    return int(value)


def get_nifti_stem(path):
    """
    Handles both .nii and .nii.gz correctly.
    Path.stem alone would turn sample.nii.gz into sample.nii.
    """
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return path.stem

def build_kd_tree(edges_parent_radius):
    """Builds a KD-Tree using edge midpoints for efficient lookup."""
    edge_midpoints = [
        ((np.array(edge[0]) + np.array(edge[1])) / 2) for edge, _, _ in edges_parent_radius
    ]
    return KDTree(edge_midpoints), edge_midpoints


def calculate_distance(node, edge):
    """Finds the closest point on an edge to a given node."""
    node = np.array(node)
    edge_start, edge_end = map(np.array, edge)

    edge_direction = edge_end - edge_start
    edge_length = np.linalg.norm(edge_direction)

    if edge_length == 0:
        return np.linalg.norm(node - edge_start), edge_start

    edge_direction_unit = edge_direction / edge_length
    edge_to_node = node - edge_start
    projection = np.dot(edge_to_node, edge_direction_unit)

    if projection < 0:
        closest_point = edge_start
    elif projection > edge_length:
        closest_point = edge_end
    else:
        closest_point = edge_start + projection * edge_direction_unit

    return round(np.linalg.norm(node - closest_point)), closest_point



def find_reference_point(start_node, closest_point, step_size, parent, edge):
    start_node = torch.tensor(start_node)
    closest_point = torch.tensor(closest_point)
    edge_vector = start_node - closest_point
    p1, p2 = parent
    p1 = torch.tensor(p1)
    p2 = torch.tensor(p2)

    if torch.linalg.norm(edge_vector) < step_size:
        dist_parent_edge = step_size - torch.linalg.norm(edge_vector)
        if edge == parent:
            reference_point = start_node
        else:
            parent_vector = p1 - p2
            parent_vector_normalized = parent_vector / torch.linalg.norm(parent_vector)
            reference_point = start_node + (parent_vector_normalized * dist_parent_edge)
    else:
        edge_vector_normalized = edge_vector / torch.linalg.norm(edge_vector)
        reference_point = closest_point + (edge_vector_normalized * step_size)

    return reference_point.tolist()

def find_valid_closest_edge(
    node,
    edges_parent_radius,
    kd_tree,
    search_radius,
    step_size,
    radius_divisor,
):
    """
    Finds the closest edge satisfying distance <= radius / radius_divisor.
    """
    nearby_edge_indices = kd_tree.query_ball_point(node, search_radius)

    best_distance = float("inf")
    best_closest_point = None
    best_edge = None
    best_parent = None
    best_radius = None

    for idx in nearby_edge_indices:
        edge, parent, radius = edges_parent_radius[idx]

        distance, closest_point = calculate_distance(node, edge)
        valid_radius = radius / radius_divisor

        if distance <= round(valid_radius) and distance < best_distance:
            best_distance = distance
            best_closest_point = closest_point
            best_edge = edge
            best_parent = parent
            best_radius = valid_radius

    if best_edge is None:
        return None

    start_node = best_edge[0]
    reference_point = find_reference_point(
        start_node=start_node,
        closest_point=best_closest_point,
        step_size=step_size,
        parent=best_parent,
        edge=best_edge,
    )

    return node, reference_point, best_radius

def find_shortest_distance_to_edge(mask_coordinates, edges_parent_radius, cfg):
    """
    Finds the closest valid edge for all mask nodes using KD-Tree and multiprocessing.
    """
    kd_tree, edge_midpoints = build_kd_tree(edges_parent_radius)

    search_radius = cfg["processing"]["edge_search_radius"]
    step_size = cfg["processing"]["step_size"]
    radius_divisor = cfg["processing"]["radius_divisor"]
    num_processes = get_num_processes(cfg)

    worker_args = [
        (
            node,
            edges_parent_radius,
            kd_tree,
            search_radius,
            step_size,
            radius_divisor,
        )
        for node in mask_coordinates
    ]

    with Pool(processes=num_processes) as pool:
        results = pool.starmap(find_valid_closest_edge, worker_args)

    print("Results are ready")
    return [r for r in results if r is not None]

def clear_folder_contents(folder_path):

    if os.path.exists(folder_path):

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:

                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    else:
        print(f"The folder {folder_path} does not exist.")

def main(config_path):
    cfg = load_config(config_path)

    start_time = time.time()

    mask_file_folder = Path(cfg["paths"]["mask_folder"])
    centerline_folder = Path(cfg["paths"]["centerline_folder"])
    vtk_folder = Path(cfg["paths"]["vtk_folder"])
    swc_folder = Path(cfg["paths"]["swc_folder"])
    output_tif_folder = Path(cfg["paths"]["output_tif_folder"])

    mask_glob = cfg["files"].get("mask_glob", "*.nii*")
    centerline_extension = cfg["files"].get("centerline_extension", ".nii")
    vtk_extension = cfg["files"].get("vtk_extension", ".vtk")
    output_suffix = cfg["files"].get("output_suffix", "_scale2.tif")



    output_tif_folder.mkdir(parents=True, exist_ok=True)

    for mask_file_path in sorted(mask_file_folder.glob(mask_glob)):
        print(mask_file_path)

        mask, mask_coordinates = extract_data_from_nifti(str(mask_file_path))

        just_filename = get_nifti_stem(mask_file_path)

        centerline_file_path = centerline_folder / f"{just_filename}{centerline_extension}"
        vtk_file_path = vtk_folder / f"{just_filename}{vtk_extension}"

        centerline, centerline_coordinates = extract_data_from_nifti(str(centerline_file_path))

        mask_coordinates = mask_coordinates.tolist()
        print(len(mask_coordinates))

        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(str(vtk_file_path))
        reader.Update()

        vtk_polydata = reader.GetOutput()
        edges = show_vtk_polydata_in_napari(vtk_polydata, centerline)

        radius = np.array(vtk_polydata.GetCellData().GetArray(0))
        print(radius)

        radius = tuple(radius)
        edges_radius = [(x, y) for x, y in zip(edges, radius)]

        print(edges_radius[11])

        initial_graph, final_graph, roots_initial_edge, edges_parent_radius = network_x(
            edges,
            edges_radius,
        )

        print("Total length:", len(edges_parent_radius))
        print(len(edges))

        nodes_vector_radii = find_shortest_distance_to_edge(
            mask_coordinates,
            edges_parent_radius,
            cfg,
        )

        end_time = time.time()
        print("Execution time: {:.2f} seconds".format(end_time - start_time))

        image_shape = mask.shape

        image_array = np.zeros(image_shape, dtype=np.float32)
        vector_x_array = np.zeros(image_shape, dtype=np.float32)
        vector_y_array = np.zeros(image_shape, dtype=np.float32)
        vector_z_array = np.zeros(image_shape, dtype=np.float32)
        radius_array = np.zeros(image_shape, dtype=np.float32)

        counter = 0

        for node_vector_radius in nodes_vector_radii:
            if node_vector_radius is None:
                continue

            node, vector, radius = node_vector_radius

            if vector is not None:
                x1, y1, z1 = node

                image_array[x1, y1, z1] = 255

                node = np.array(node)
                vector = np.array(vector)

                flow_vector = vector - node
                norm = np.linalg.norm(flow_vector)

                if norm == 0:
                    continue

                flow_vector_unit = flow_vector / norm

                vector_x_array[x1, y1, z1] = flow_vector[0]
                vector_y_array[x1, y1, z1] = flow_vector[1]
                vector_z_array[x1, y1, z1] = flow_vector[2]

                # Uncomment if you want to save radius too
                # radius_array[x1, y1, z1] = radius

                counter += 1

        print(f"total mask pixels: {len(mask_coordinates)}")
        print(counter)

        image_flow_vector = np.stack(
            (
                image_array,
                vector_x_array,
                vector_y_array,
                vector_z_array,
            ),
            axis=2,
        )

        image_flow_vector = np.transpose(image_flow_vector, (2, 0, 1, 3))

        output_tif_path = output_tif_folder / f"{just_filename}{output_suffix}"
        tifffile.imwrite(str(output_tif_path), image_flow_vector)

        clear_folder_contents(str(swc_folder))

        end_time = time.time()
        processing_time = end_time - start_time
        print(f"Processing time: {processing_time} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="vector_config.yaml",
        help="Path to dataset-specific config YAML file.",
    )
    args = parser.parse_args()

    main(args.config)

