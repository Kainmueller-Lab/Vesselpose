import tifffile

import numpy as np
import nibabel as nib
import vtk
import networkx as nx
import pandas as pd
from multiprocessing import Pool, cpu_count
import math
import time
import os
import csv
import ast
import sys
import shutil

def extract_data_from_nifti(nifti_file_path):
    img = nib.load(nifti_file_path)
    print(img.shape)

    data = img.get_fdata()

    coordinates = np.array(np.where(data > 0)).T

    return data, coordinates

def vtk_polydata_to_numpy(vtk_polydata):
    points = vtk_polydata.GetPoints().GetData()
    nodes = np.array(points)
    edges = []
    for i in range(vtk_polydata.GetNumberOfCells()):
        cell = vtk_polydata.GetCell(i).GetPoints().GetData()
        edge = np.array(cell)
        edge = edge.tolist()
        edges.append(edge)

    return nodes, edges



def show_vtk_polydata_in_napari(vtk_polydata, centerline):
    nodes, edges = vtk_polydata_to_numpy(vtk_polydata)
    nodes = nodes.tolist()
    print(nodes[0])
    print(edges[0])
    edges_array = np.array(edges)
    print(edges[0])
    # edges_array_reshaped = edges_array.reshape(-1, 2, 3)
    # print(edges_array_reshaped[0])
    # [((bounds[0] + 464)/2.855, (bounds[2] + 400)/2.63, (bounds[4] + 1760)/5.93)
    # Convert the array of edges to a list of tuples
    edges_nx = [tuple(map(tuple, edge)) for edge in edges_array]
    print(edges_nx[0])
    edges_nx_reshaped = []
    nodes_nx_reshaped = []

    xg_min = vtk_polydata.GetBounds()[0]
    xg_max = vtk_polydata.GetBounds()[1]
    yg_min = vtk_polydata.GetBounds()[2]
    yg_max = vtk_polydata.GetBounds()[3]
    zg_min = vtk_polydata.GetBounds()[4]
    zg_max = vtk_polydata.GetBounds()[5]

    nonzero_indices_centerline = np.nonzero(centerline)

    xc_min, yc_min, zc_min = np.min(nonzero_indices_centerline, axis=1)
    xc_max, yc_max, zc_max = np.max(nonzero_indices_centerline, axis=1)

    print(f"Bounding Box: ({xc_min}, {yc_min}, {zc_min}) to ({xc_max}, {yc_max}, {zc_max})")

    factor_x = (abs(xg_min) + abs(xg_max)) / (abs(xc_max) - abs(xc_min))
    factor_y = (abs(yg_min) + abs(yg_max)) / (abs(yc_max) - abs(yc_min))
    factor_z = (abs(zg_min) + abs(zg_max)) / (abs(zc_max) - abs(zc_min))

    print(factor_x)
    print(factor_y)

    print(factor_z)
    print(abs(xg_min))
    print(abs(yg_min))

    for edge in edges_nx:
        (node1_x, node1_y, node1_z), (node2_x, node2_y, node2_z) = edge
        node1_x = ((node1_x + abs(xg_min)) / factor_x) + xc_min  # (463.986 * 2)/(315-6)# check the bounding box of the respective data
        node1_y = ((node1_y + abs(yg_min)) / factor_y) + yc_min
        node1_z = (node1_z / factor_z) + zc_min  # between o and 1759.92
        node2_x = ((node2_x + abs(xg_min)) / factor_x) + xc_min
        node2_y = ((node2_y + abs(yg_min)) / factor_y) + yc_min
        node2_z = (node2_z / factor_z) + zc_min
        edges_nx_reshaped.append(((node1_x, node1_y, node1_z), (node2_x, node2_y, node2_z)))

    # print(len(nodes_nx))
    # print(len(edges_nx))
    G = nx.Graph()
    G.add_edges_from(edges_nx_reshaped)
    connected_components = list(nx.connected_components(G))
    print("Number of connected components:", len(connected_components))
    cycles = nx.cycle_basis(G)

    # Check if there are cycles
    if cycles:
        print("Graph contains cycles.")
        print("Cycles:", cycles)
    else:
        print("Graph is acyclic.")

    # edges_nx_reshaped = np.array(edges_nx_reshaped)
    # edges_nx_reshaped = edges_nx_reshaped.tolist()

    #img_data = tifffile.imread('/mnt/md0/rajalakshmi/deepvesselnet/centerline/centerline/2_transpose.tiff')
    # with napari.gui_qt():
    #     viewer = napari.Viewer()
    #     # viewer.add_image(img_data)
    #     viewer.add_shapes(data=edges_nx_reshaped, shape_type='line', edge_color='red', edge_width=0.5)

    #     print("edges are added")
    return edges_nx_reshaped

def get_endpoints_edges(component_edges, i):
    endpoints = {}
    for edge in component_edges:
        neighbors = []
        node1, node2 = edge

        for neighbor in component_edges:
            other_nodes = []
            node1_nei, node2_nei = neighbor

            if neighbor != edge:
                other_nodes.append(node1_nei)

                other_nodes.append(node2_nei)

        # print(len(other_nodes))
            if (node1 in other_nodes) or (node2 in other_nodes):
                neighbors.append(neighbor)
        #print(len(neighbors))
        if len(neighbors) <= 2:
            endpoints[edge] = len(neighbors)

    print(len(endpoints))
    endpoints_dict = {i: item[0] for i, item in enumerate(endpoints)}

    df = pd.DataFrame.from_dict(endpoints_dict)
    #df.to_csv(f'/home/rajalakshmi/Segmentation/md0/endpoints/endpoints_{i}.csv', index=False)
    return endpoints


def get_root(endpoints, edges_radius):
    endpoints_edges_radius = []
    print(len(endpoints))
    # print(endpoints)

    for edge, neighbors in endpoints.items():
        node1, node2 = edge
        for edge_radius in edges_radius:
            if (node1, node2) == edge_radius[0] or (node2, node1) == edge_radius[0]:
                updated_edge = edge_radius
                endpoints_edges_radius.append(updated_edge)
            # else:
            #     print("couldnt find")
    print(len(endpoints_edges_radius))
    # print(endpoints_edges_radius)

    # if(len(endpoints_edges_radius)==1):
    #     print(endpoints_edges_radius)

    edges_with_max_radius = []
    max_radius = float('-inf')

    for edge_radius in endpoints_edges_radius:
        edge, radius = edge_radius
        if radius > max_radius:
            max_radius = radius
            edges_with_max_radius = [edge]
            # edges_with_max_radius.append(edge)
        elif radius == max_radius:
            edges_with_max_radius.append(edge)

    # nodes_with_max_radius = list(set(nodes_with_max_radius))
    if len(edges_with_max_radius) == 1:
        root = edges_with_max_radius[0]

    else:
        print("not both")
        neighbor = float('inf')
        root = None
        ties = []
        for edge, neighbors in endpoints.items():
            if edge in edges_with_max_radius:
                if neighbors < neighbor:
                    neighbor = neighbors
                    root = edge
                    ties = [edge]
                elif neighbors == neighbor:
                    ties.append(edge)

            if len(ties) > 1:

                for edge in ties:
                    node1, node2 = edge
                    if node1[2] == 4.0:
                        root = edge

    return root


def get_non_zero_neighbors_3d(edge, edges, visited, neighbors_count):
    neighbors_filtered = []
    node1, node2 = edge
    for e in edges:
        if e not in visited and e != edge and e not in neighbors_count:
            if node1 in e or node2 in e:
                neighbors_filtered.append(e)

    return neighbors_filtered


def get_all_neighbors_iterative(root_coordinates, edges, i, edges_radius):
    visited = []
    # edges_ordered =[]
    neighbors_count = []
    queue = root_coordinates.copy()  # Start with the root coordinates
    swc = []
    # node_id = 0
    root_radius = 0
    for edge_radius in edges_radius:
        edge, radius = edge_radius
        if edge == root_coordinates[0]:
            root_radius = radius
    #swc.append({"edge": root_coordinates[0], "parent": root_coordinates[0], "radius": root_radius})
    swc.append((root_coordinates[0], root_coordinates[0], root_radius))
    # node_parent = {}
    while queue:

        current_edge = queue.pop(0)  # Dequeue the front of the queue
        # print(current_coord)

        if current_edge in visited:
            continue

        visited.append(current_edge)

        neighbors = get_non_zero_neighbors_3d(current_edge, edges, visited, neighbors_count)
        # print(f"Current: {current_coord}, Neighbors: {neighbors}")

        # node_parent[root_coordinates[0]] = 0
        if len(neighbors) == 0:
            continue

        for neighbor in neighbors:
            # edges_ordered.append((neighbor, current_edge))
            queue.append(neighbor)
            neighbors_count.append(neighbor)

            # node_id += 1
            # node_parent[neighbor] = node_id
            current_radius = 0
            if neighbor in edges:
                for edge_radius in edges_radius:
                    edge, radius = edge_radius
                    if edge == neighbor:
                        current_radius = radius

                #swc.append({"edge": neighbor, "parent": current_edge, "radius": current_radius})
                swc.append((neighbor, current_edge, current_radius))

    # df = pd.DataFrame.from_dict(swc)
    # df.to_csv(f'/home/rajalakshmi/Segmentation/md0/swcs/component_edges_{i}.csv', index=False)
    #df.to_csv(f'/fast/AG_Kainmueller/Raji/UNet_vessel/deepvesselnet/graph/swcs/component_edges_{i}.csv', index=False)

    return visited, swc


def network_x(edges, edges_radius):
    # edges = np.array(edges)
    # edges = [tuple(map(tuple, edge)) for edge in edges]
    # print(edges[0])
    G = nx.Graph()
    # G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    connected_components = list(nx.connected_components(G))
    final_graph = nx.DiGraph()
    roots_initial_edge = []
    edges_parent_radius = []

    for i, component in enumerate(connected_components, start=1):
        print(f"preparing swc for component{i}")
        component_graph = G.subgraph(component)
        component_edges = component_graph.edges()
        component_edges_filtered = []
        for edge in component_edges:
            node1, node2 = edge
            if edge not in edges:
                edge = (node2, node1)
                component_edges_filtered.append(edge)
            else:
                component_edges_filtered.append(edge)

        endpoints = get_endpoints_edges(component_edges_filtered, i)
        root = get_root(endpoints, edges_radius)
        roots_initial_edge.append(root)
        print(root)
        root_coordinates = [root]
        ordered_edges, swc = get_all_neighbors_iterative(root_coordinates, component_edges_filtered, i, edges_radius)
        for item in swc:
            edges_parent_radius.append(item)
        nodes = G.nodes()
        final_graph.add_edges_from(ordered_edges)

    return G, final_graph, roots_initial_edge, edges_parent_radius


def get_radius(node, radii_with_coordinates):
    for edge, parent, radius in radii_with_coordinates:
        if edge[0] == node:
            return radius

    return 1  # Default radius if not found


# Function to write an SWC file for a tree
def write_swc_file(tree, root, radii_with_coordinates, file_path):
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
        radius = get_radius(root, radii_with_coordinates)
        x, y, z = root
        swc_file.write(f"{index} 0 {x} {y} {z} {radius} -1\n")
        index += 1

        # Traverse the tree and write nodes
        for parent, child in nx.bfs_edges(tree, source=root):
            if child not in node_index_map:
                node_index_map[child] = index
                radius = get_radius(child, radii_with_coordinates)
                x, y, z = child
                parent_index = node_index_map[parent]
                swc_file.write(f"{index} 0 {x} {y} {z} {radius} {parent_index}\n")
                index += 1


# Write separate SWC files for each tree






def read_csv_to_list(csv_file_path):
    data_list = []

    with open(csv_file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)  # Skip the header row

        for row in csv_reader:
            data_row = []
            for item in row:
                try:
                    value = ast.literal_eval(item)
                except (ValueError, SyntaxError):
                    value = item.strip("'")  # Remove single quotes if present
                data_row.append(value)
            data_list.append(tuple(data_row))

    return data_list

def read_folder_of_csv_to_single_list(folder_path):
    all_data = []

    # Iterate over each file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):  # Check if the file is a CSV file
            csv_file_path = os.path.join(folder_path, filename)
            data_list = read_csv_to_list(csv_file_path)
            all_data.extend(data_list)

    return all_data