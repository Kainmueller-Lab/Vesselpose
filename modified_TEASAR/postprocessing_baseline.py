import networkx as nx
import os
from collections import deque
import numpy as np


def read_swc_to_networkx_with_node_radius(swc_file, combined_graph):
    """
    Reads an SWC file and adds nodes and edges to the graph,
    with radius as a node attribute.
    """
    coords_dict = {}

    with open(swc_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue

            # SWC format columns: ID, type, x, y, z, radius, parent
            data = line.strip().split()
            if len(data) < 7:
                continue

            node_id = int(data[0])
            parent_id = int(data[6])
            x, y, z = float(data[2]), float(data[3]), float(data[4])
            radius = float(data[5])/3

            # Add node with radius as an attribute
            node_coords = (x, y, z)
            combined_graph.add_node(node_coords, radius=radius)

            # Store coordinates for referencing parent-child relationships
            coords_dict[node_id] = node_coords

            # Add edge if the parent exists
            if parent_id != -1:
                parent_coords = coords_dict.get(parent_id)
                if parent_coords:
                    combined_graph.add_edge(parent_coords, node_coords)

def combine_swc_folder_to_graph(folder_path):
    """
    Combines multiple SWC files into a single graph with radius as node attribute.
    """
    combined_graph = nx.Graph()
    for filename in os.listdir(folder_path):
        if filename.endswith('.swc'):
            file_path = os.path.join(folder_path, filename)
            read_swc_to_networkx_with_node_radius(file_path, combined_graph)
    return combined_graph

def simplify_neuron_graph_corrected(G):
    branch_points = [node for node, degree in G.degree() if degree > 2]
    endpoints = [node for node, degree in G.degree() if degree == 1]
    critical_points = set(branch_points + endpoints)

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




def build_directed_tree(final_graph):
    roots_final = []

    # Initialize an empty directed graph for the result
    directed_graph = final_graph.to_directed()
    to_remove = list(directed_graph.edges)
    directed_graph.remove_edges_from(to_remove)
    # print(nx.number_of_nodes(directed_graph))
    # for node, data in list(directed_graph.nodes(data=True))[:5]:
    #     print(f"Node: {node}, Attributes: {data}")

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

def get_radius(node, directed_graph):
    return directed_graph.nodes[node].get('radius', 1.0)

def write_swc_file(tree, root, directed_trees_pred, file_path):
    """
    Write the SWC file for a given tree.
    """
    with open(file_path, 'w') as swc_file:
        # Write header
        swc_file.write("# SWC file for tree rooted at {}\n".format(root))

        # Maintain node indices and parent relationships
        node_index_map = {}  # Maps node coordinates to SWC indices
        index = 1  # Start index for SWC

        # Write the root node (special case, parent is -1)
        node_index_map[root] = index
        radius = get_radius(root, directed_trees_pred)
        x, y, z = root
        swc_file.write(f"{index} 0 {x} {y} {z} {radius} -1\n")
        index += 1

        # Traverse the tree and write nodes
        for parent, child in nx.bfs_edges(tree, source=root):
            if child not in node_index_map:
                node_index_map[child] = index
                radius = get_radius(child, directed_trees_pred)
                x, y, z = child
                parent_index = node_index_map[parent]
                swc_file.write(f"{index} 0 {x} {y} {z} {radius} {parent_index}\n")
                index += 1



if __name__ == "__main__":
    swc_file = "/mnt/md0/rajalakshmi/dvn/test_data/teasar/50/"
    roots_initial = []
    roots_initial = [(round(x[0]), round(x[1]), round(x[2])) for x in roots_initial]
    output_dir = "/mnt/md0/rajalakshmi/dvn/test_data/teasar/50/50/"

    # Generate graphs
    G = combine_swc_folder_to_graph(swc_file)

    simplified_G = simplify_neuron_graph_corrected(G)
    print(nx.number_of_nodes(simplified_G))

    directed_graph, roots_final = build_directed_tree(simplified_G)



    for i, root in enumerate(roots_final):
        # Create a subgraph for the tree rooted at 'root'
        if root in directed_graph.nodes:
            subgraph = directed_graph.subgraph(nx.descendants(directed_graph, root) | {root})
            print(nx.number_of_nodes(subgraph))

            # Generate the SWC file for this tree
            swc_filename = f"tree_{i + 1}_root_{root[0]}_{root[1]}_{root[2]}.swc"
            file_path = os.path.join(output_dir, swc_filename)
            write_swc_file(subgraph, root, directed_graph, file_path)
            print(f"SWC file written: {file_path}")
        else:
            print(f"Root {root} is not in the graph!")


