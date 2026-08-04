import os
import glob
from argparse import Namespace
from skeletonization import *
from dataIO import *
from postprocessing import *
import tifffile
import scipy.io
import ast
import re
import numpy as np
import networkx as nx
import yaml


def load_config():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "teasar_config.yaml")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def test_skeletonization(paths, dataset_config):

    image_path = paths.input
    dataset = paths.dataset
    cfg = dataset_config[dataset]

    image = tifffile.imread(image_path)

    mask = image[0, :, :, :]
    mask = mask > 0.5

    file_number = os.path.splitext(os.path.basename(image_path))[0]

    output_mat = os.path.join(paths.base_output_dir, 'teasar', f'{file_number}.mat')
    output_dir_swc = os.path.join(paths.base_output_dir, 'teasar_swc', f'{file_number}_BP')
    output_swc = os.path.join(paths.base_output_dir, 'teasar_swc', f'{file_number}.swc')

    os.makedirs(output_dir_swc, exist_ok=True)

    roots_file_path = paths.roots_file
    with open(roots_file_path, 'r') as f:
        roots_text = f.read()

    pattern = rf"roots_{file_number}\s*=\s*(\[[\s\S]*?\])"
    match = re.search(pattern, roots_text)
    if not match:
        print(f"Roots for image {file_number} not found. Skipping.")
        return

    roots = ast.literal_eval(match.group(1))
    roots = [(round(x[0]), round(x[1]), round(x[2])) for x in roots]
    init_root = np.array(roots)

    print(f"\nProcessing: {file_number}")
    print(f"Dataset: {dataset}")
    print(f"Image shape: {image.shape}")

    skeletons = skeletonize(
        image=image,
        object_input=mask,
        object_id=1,
        dsmp_resolution=cfg["spacing"],
        parameters=cfg,
        init_root=init_root
    )

    save_skeleton_mat(skeletons, output_mat)
    data = scipy.io.loadmat(output_mat)

    final_graph = get_nx_graph(data)
    final_graph_merges = merge_smaller_components(final_graph, image)
    directed_graph, roots_final = build_directed_tree(final_graph_merges, roots)

    if cfg.get("postprocess_simplify", False):
        roots = [root for root in roots if root in directed_graph]
        simplified_G = simplify_neuron_graph_corrected(directed_graph, roots)
        directed_graph, roots_final = build_directed_tree(simplified_G, roots)

    for i, root in enumerate(roots_final):
        if root in directed_graph.nodes:
            subgraph = directed_graph.subgraph(nx.descendants(directed_graph, root) | {root})
            swc_filename = f"tree_{i + 1}_root_{root[0]}_{root[1]}_{root[2]}.swc"
            write_swc_file(subgraph, root, directed_graph, os.path.join(output_dir_swc, swc_filename))

    merge_swc_files(output_dir_swc, output_swc)
    print(f"Done: {file_number}")


def main():
    config = load_config()

    dataset_config = config["datasets"]
    dataset = config["dataset"]

    input_folder = config["paths"]["input_folder"]
    base_output_dir = config["paths"]["base_output_dir"]
    roots_file = config["paths"]["roots_file"]

    if dataset not in dataset_config:
        raise ValueError(f"Unknown dataset: {dataset}")

    os.makedirs(os.path.join(base_output_dir, 'teasar'), exist_ok=True)
    os.makedirs(os.path.join(base_output_dir, 'teasar_swc'), exist_ok=True)

    tif_files = sorted(glob.glob(os.path.join(input_folder, '*.tif')))

    for tif_path in tif_files:
        file_number = os.path.splitext(os.path.basename(tif_path))[0]

        paths = Namespace(
            input=tif_path,
            base_output_dir=base_output_dir,
            roots_file=roots_file,
            dataset=dataset,
        )

        try:
            test_skeletonization(paths, dataset_config)
        except Exception as e:
            print(f"Failed to process {file_number}: {e}")


if __name__ == "__main__":
    main()