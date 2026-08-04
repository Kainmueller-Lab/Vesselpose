
import scipy.io
import numpy as np
import networkx as nx
import sys
import tifffile
from scipy.spatial import KDTree
from collections import deque
import os

from scipy.spatial import distance

# mat_file = '/home/rajalakshmi/Segmentation/md0/greedy_solver/83_skeleton_DBF_best_path_s_0.6_c_5_mag_new_logic.mat'

def get_nx_graph(data):
    total_trees = 0
    total_edges = 0
    total_cycles = 0
    trees_with_no_edges = 0
    final_graph = nx.Graph()
    key_count = 0
    for key in data:
        if key.startswith('skeleton_'):
            skeleton = data[key]

            nodes = skeleton['nodes'][0, 0]
            edges = skeleton['edges'][0, 0]
            radii = skeleton['radii'][0, 0]
            radii = radii.flatten()

            # print("nodes shape:", nodes.shape)
            # print("radius shape:", radii.shape)

            G = nx.Graph()

            for i, node in enumerate(nodes):
                node_pos = tuple(node)
                G.add_node(node_pos, radius=radii[i])

            for edge in edges:
                nodeid1, nodeid2 = edge
                node1 = nodes[nodeid1]
                node2 = nodes[nodeid2]
                node1 = tuple(node1)
                node2 = tuple(node2)
                G.add_edge(node1, node2)

            final_graph.add_nodes_from(G.nodes(data=True))
            final_graph.add_edges_from(G.edges())

            num_trees = nx.number_connected_components(G)
            # print(num_trees)
            total_trees += num_trees

            num_edges = G.number_of_edges()
            print(num_edges)
            total_edges += num_edges



            for i, component in enumerate(nx.connected_components(G)):
                subgraph = G.subgraph(component)

                if subgraph.number_of_edges() == 1:
                    trees_with_no_edges += 1


                try:
                    cycle = nx.find_cycle(subgraph)
                    total_cycles += 1
                    print(f"Cycle found in tree {i + 1} of {key}: {cycle}")
                except nx.NetworkXNoCycle:
                    pass

            key_count += 1

    # print(f"Total number of trees in all skeletons: {total_trees}")
    # print(f"Total number of trees with no edges: {trees_with_no_edges}")
    # print(f"Total number of edges in all skeletons: {total_edges}")
    # print(f"Total number of cycles in all skeletons: {total_cycles}")
    # print(final_graph.number_of_edges())
    # print(nx.number_connected_components(final_graph))
    # node = list(final_graph.nodes)[100]
    # radius = final_graph.nodes[node]
    # print(node)
    # print(radius)
    components = list(nx.connected_components(final_graph))

    # Iterate over each component
    for component in components:
        subgraph = final_graph.subgraph(component)
        if subgraph.number_of_edges() == 1:
            final_graph.remove_nodes_from(component)

    print(nx.number_connected_components(final_graph))


    print(final_graph.number_of_edges())
    print(nx.number_connected_components(final_graph))

    return final_graph



def merge_smaller_components(final_graph, image):
    max_edges = 500
    distance_threshold = 5
    radius_diff_threshold = 3
    components = list(nx.connected_components(final_graph))

    smaller_components = [comp for comp in components if final_graph.subgraph(comp).number_of_edges() < max_edges]

    all_positions = list(final_graph.nodes)
    kdtree = KDTree(all_positions)

    for small_comp in smaller_components:
        for node in small_comp:
            node_pos = node
            node_radius = final_graph.nodes[node]['radius']
            node_vector = (image[1, node_pos[0], node_pos[1], node_pos[2]], image[2, node_pos[0], node_pos[1], node_pos[2]], image[3, node_pos[0], node_pos[1], node_pos[2]])
            node_vector = np.array(node_vector)


            nearby_indices = kdtree.query_ball_point(node_pos, distance_threshold)
            for idx in nearby_indices:
                other_node = all_positions[idx]


                if other_node in small_comp:
                    continue

                other_node_radius = final_graph.nodes[other_node]['radius']
                other_node_vector = (image[1, other_node[0], other_node[1], other_node[2]], image[2, other_node[0], other_node[1], other_node[2]], image[3, other_node[0], other_node[1], other_node[2]])
                other_node_vector = np.array(other_node_vector)

                dot_product = np.dot(node_vector, other_node_vector)
                magnitude1 = np.linalg.norm(node_vector)
                magnitude2 = np.linalg.norm(other_node_vector)


                if magnitude1 == 0 or magnitude2 == 0:
                    raise ValueError("One or both vectors have zero magnitude, cannot compute angle.")


                cos_theta = dot_product / (magnitude1 * magnitude2)


                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                angle = np.arccos(cos_theta)

                if abs(node_radius - other_node_radius) <= radius_diff_threshold:
                    if angle <= 100:
                        final_graph.add_edge(node, other_node)
                        try:
                            nx.find_cycle(final_graph)
                            final_graph.remove_edge(node, other_node)
                        except nx.NetworkXNoCycle:
                            print(f"Connected {node} to {other_node}")
                            continue

    print(nx.number_connected_components(final_graph))

    components = list(nx.connected_components(final_graph))


    for i, component in enumerate(components):
        subgraph = final_graph.subgraph(component)
        num_edges = subgraph.number_of_edges()
        print(f"Component {i + 1}: Number of edges = {num_edges}")

    return final_graph


def build_directed_tree(final_graph, roots_initial):
    roots_final = []

    # Initialize an empty directed graph for the result
    directed_graph = final_graph.to_directed()
    to_remove = list(directed_graph.edges)
    directed_graph.remove_edges_from(to_remove)
    print(nx.number_of_nodes(directed_graph))
    for node, data in list(directed_graph.nodes(data=True))[:5]:
        print(f"Node: {node}, Attributes: {data}")

    connected_components = list(nx.connected_components(final_graph))

    for cc in connected_components:
        cc_graph = final_graph.subgraph(cc)
        if len(cc_graph.nodes) == 1:
            continue

        # get end nodes
        degree = cc_graph.degree()
        terminal_idx = []
        for k, item in degree:
            if item == 1:
                terminal_idx.append(k)

        # get radius of end end nodes
        for node in cc_graph.nodes:
            if node in roots_initial:
                root = node
                roots_final.append(root)
                break
        else:
            radius = nx.get_node_attributes(cc_graph, 'radius')
            terminal_radius = np.array([radius[k] for k in terminal_idx])
            max_rad_idx = np.argmax(terminal_radius)
            root = terminal_idx[max_rad_idx]
            roots_final.append(root)
            max_rad = terminal_radius[max_rad_idx]

        # starting at node with largest radius and traverse through
        # connected component, add edges to directed graph
        visited = []
        stack = [root]
        while len(stack) > 0:
            c_node = stack.pop()
            neighbors = cc_graph.neighbors(c_node)
            for n in neighbors:
                if n not in visited:
                    stack.append(n)
                    directed_graph.add_edge(c_node, n)
            visited.append(c_node)

    return directed_graph, roots_final


def simplify_neuron_graph_corrected(G, roots):
    branch_points = [node for node, degree in G.degree() if degree > 2]
    endpoints = [node for node, degree in G.degree() if degree == 1]
    critical_points = set(branch_points + endpoints + roots)

    simplified_G = nx.Graph()


    for start in critical_points:
        visited = set()

        queue = deque([(start, None)])

        while queue:
            current, previous = queue.popleft()
            visited.add(current)

            if current in critical_points and current != start:
                simplified_G.add_node(start, **G.nodes[start])
                simplified_G.add_node(current, **G.nodes[current])

                simplified_G.add_edge(start, current)

                continue

            for neighbor in G.neighbors(current):
                if neighbor not in visited:
                    queue.append((neighbor, current))
                    visited.add(neighbor)

    return simplified_G



def get_radius(node, directed_graph):
    return directed_graph.nodes[node].get('radius', 1.0)

def write_swc_file(tree, root, directed_trees_pred, file_path):
    """
    Write the SWC file for a given tree.
    """
    with open(file_path, 'w') as swc_file:
        # Write header
        swc_file.write("# SWC file for tree rooted at {}\n".format(root))


        node_index_map = {}
        index = 1

        # Write the root node (special case, parent is -1)
        node_index_map[root] = index
        radius = get_radius(root, directed_trees_pred)
        x, y, z = root
        swc_file.write(f"{index} 0 {x} {y} {z} {radius} -1\n")
        index += 1


        for parent, child in nx.bfs_edges(tree, source=root):
            if child not in node_index_map:
                node_index_map[child] = index
                radius = get_radius(child, directed_trees_pred)
                x, y, z = child
                parent_index = node_index_map[parent]
                swc_file.write(f"{index} 0 {x} {y} {z} {radius} {parent_index}\n")
                index += 1
import os

def merge_swc_files(folder_path, output_file):
    current_node_id = 1  # Start ID for nodes
    root_node_id_map = {}  # Map of root node IDs for each file

    with open(output_file, 'w') as outfile:
        # Write a header
        outfile.write("# Merged SWC file\n")

        for filename in sorted(os.listdir(folder_path)):
            if filename.endswith('.swc'):
                file_path = os.path.join(folder_path, filename)
                root_node_id = current_node_id  # The root ID for this subtree
                root_node_id_map[filename] = root_node_id

                with open(file_path, 'r') as infile:
                    node_map = {}  # Map old IDs to new IDs for this file
                    for line in infile:
                        if line.startswith('#'):
                            continue  # Skip comment lines

                        # SWC format: ID, type, x, y, z, radius, parent
                        data = line.strip().split()
                        if len(data) < 7:
                            continue  # Skip malformed lines

                        old_id = int(data[0])
                        parent_id = int(data[6])
                        new_id = current_node_id
                        node_map[old_id] = new_id

                        # Remap parent ID if it exists, otherwise set to -1
                        new_parent_id = node_map[parent_id] if parent_id in node_map else -1

                        # Write the new line with remapped IDs
                        outfile.write(f"{new_id} {data[1]} {data[2]} {data[3]} {data[4]} {data[5]} {new_parent_id}\n")
                        current_node_id += 1

    print(f"Merged SWC file written to: {output_file}")
    print("Root node IDs for each file:", root_node_id_map)


# if __name__ == "__main__":
#     mat_file = '/mnt/md0/rajalakshmi/Unet_seg/whole_pipeline_with_zarr_funlib/teasar/40.mat'
#     data = scipy.io.loadmat(mat_file)
#     # image = tifffile.imread('/mnt/md0/rajalakshmi/Unet_seg/unet_predictions/predictions_final_finetuning1685_raw_merged_red_cropped.tif')
#     image = tifffile.imread('/mnt/md0/rajalakshmi/Unet_seg/whole_pipeline_with_zarr_funlib/inference_shuffling/40.tif')
#     mask = image[0, :, :, :]
#     mask = mask > 0.1
#
#     roots_initial =[(312.482043688734, 201.272518511404, 4.0),
# 	     (83.68453754961946, 48.229713564161145, 4.0),
# 	     (296.86298148410054, 189.62955256852422, 4.0),
# 	     (67.52029147410332, 39.48931086039144, 4.0),
# 	     (76.42552716242807, 143.50562828565398, 4.0),
# 	     (95.3664359394784, 85.66130510188876, 4.0),
# 	     (296.2075639994729, 46.947331899891296, 4.0),
# 	     (21.259047370023627, 81.07914926710458, 4.0),
# 	     (160.82715758214732, 188.47333465072546, 4.0),
# 	     (151.46359994566373, 252.77832877617584, 4.0),
# 	     (155.18330073789144, 45.369204058357944, 4.0),
# 	     (13.887117106512392, 27.44815101765299, 4.0),
# 	     (53.47462474200854, 108.58274413578435, 4.0),
# 	     (185.2519114190885, 208.81464192441686, 4.0),
# 	     (13.310880651937167, 196.49410258383622, 4.0),
# 	     (258.9988616599048, 151.76471704786888, 4.0)]
#
#
#
#     #if a root is missing in directed graph remove it here manually
#     roots_initial = [(round(x[0]), round(x[1]), round(x[2])) for x in roots_initial]
#
#     final_graph = get_nx_graph(data)
#
#     final_graph_merges = merge_smaller_components(final_graph)
#
#     directed_graph, roots_final = build_directed_tree(final_graph_merges)
#     print(nx.number_weakly_connected_components(directed_graph))
#     #cycle = nx.find_cycle(directed_graph)
#
#     for root in roots_initial:
#         if root not in directed_graph:
#             directed_graph.add_node(root, **final_graph.nodes[root])
#
#     simplified_G = simplify_neuron_graph_corrected(directed_graph, roots_initial)
#
#     for root in roots_initial:
#         if root not in simplified_G:
#             simplified_G.add_node(root, **directed_graph.nodes[root])
#
#     #print(simplified_G.number_of_nodes())
#
#     directed_graph, roots_final = build_directed_tree(simplified_G)
#     print(directed_graph.number_of_nodes())
#
#
#
#
#     output_dir = "/mnt/md0/rajalakshmi/Unet_seg/whole_pipeline_with_zarr_funlib/teasar_swc/40_BP/"
#
#     for i, root in enumerate(roots_final):
#         # Create a subgraph for the tree rooted at 'root'
#         if root in directed_graph.nodes:
#             subgraph = directed_graph.subgraph(nx.descendants(directed_graph, root) | {root})
#             print(nx.number_of_nodes(subgraph))
#
#             # Generate the SWC file for this tree
#             swc_filename = f"tree_{i + 1}_root_{root[0]}_{root[1]}_{root[2]}.swc"
#             file_path = os.path.join(output_dir, swc_filename)
#             write_swc_file(subgraph, root, directed_graph, file_path)
#             print(f"SWC file written: {file_path}")
#         else:
#             print(f"Root {root} is not in the graph!")