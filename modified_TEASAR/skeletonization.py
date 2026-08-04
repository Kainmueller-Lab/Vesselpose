# Import necessary packages
import numpy as np
import scipy.io as spio
from scipy import ndimage
from scipy.sparse.csgraph import *
from scipy.sparse import csr_matrix

import networkx as nx

from multiprocessing import Pool
from functools import partial

from utils import *

from time import time
import tifffile
import math
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from os import listdir
import sys
from scipy.ndimage import maximum_filter


# Define edges for the graphs
def create_edges(object_points, DBF, vec_mag, vec_dir, max_bound):
    """
	[INPUT]
	object_points : n x 3 point cloud format of the object
	DBF : distance to the boundary map
	max_bound : maximum coordinates for each dimension with padding

	[OUTPUT]
	nhood_nodes = neighborhood node array (-1 if that node is not included in the object)
	edge_dist = euclidean distance edges
	edge_weight = penalty edges
	"""
    print("creating edges")
    NDIM = object_points.shape[1]
    n = object_points.shape[0]

    object_nodes = Nodes(object_points, max_bound)

    local_max = maximum_filter(vec_mag, size=5)
    #local_max = np.max(vec_mag)
    M1 = local_max.astype(np.float32) ** np.float32(1.01)
    p_v1 = 1000000 * (vec_mag/(M1 + 0.05))**16

    M2 = np.max(DBF) ** 1.01

    p_v2 = 1000000 * (1-DBF/M2)**16

    p_v = p_v1 + p_v2
    #p_v = p_v2


    p_v = p_v.astype(np.float32)



    # 26-connectivity
    nhood_26 = np.zeros([3, 3, 3], dtype='bool')
    nhood_26 = np.where(nhood_26 == 0)

    nhood = np.zeros([nhood_26[0].size, 3], dtype=np.float16)
    for i in range(NDIM):
        nhood[:, i] = nhood_26[i]
    nhood = nhood - 1
    nhood = np.delete(nhood, find_row(nhood, [0, 0, 0]), axis=0)

    n_nhood = nhood.shape[0]

    nhood_weight = np.sum(nhood ** 2, axis=1) ** 0.5

    nhood_points = np.zeros([n, 3], dtype=np.float16)
    nhood_nodes = np.ones([n, n_nhood], dtype=np.int32) * -1
    edge_dist = np.zeros([n, n_nhood], dtype=np.float16)
    edge_weight = np.zeros([n, n_nhood], dtype=np.float32)

    print("Setting edge weight...")
    for i in range(n_nhood):
        nhood_points = object_points + nhood[i, :]

        valid = np.all(nhood_points >= 0, axis=1) * np.all(nhood_points < max_bound, axis=1)

        nhood_nodes[valid, i] = object_nodes.sub2node(nhood_points[valid, :])


        valid = nhood_nodes[:, i] != -1
        edge_dist[valid, i] = nhood_weight[i]

        valid_idx = np.where(valid)[0]


        if len(valid_idx) > 0:

            relative_dirs = nhood[i, :]
            normalized_rel = relative_dirs / np.linalg.norm(relative_dirs)


            flow_dirs = vec_dir[object_points[valid_idx, 0], object_points[valid_idx, 1], object_points[valid_idx, 2]]
            flow_magnitudes = np.linalg.norm(flow_dirs, axis=1)


            valid_flow = flow_magnitudes > 0
            flow_dirs[valid_flow] /= flow_magnitudes[valid_flow, None]


            cos_theta = np.einsum('ij,j->i', flow_dirs, normalized_rel)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)  # check values range between
            angle_deg = np.degrees(np.arccos(cos_theta))


            M = 180
            angular_penalty_factor = 1000000 * (angle_deg / M) ** 16
            DBF_mag_penalty = p_v[object_points[valid_idx, 0], object_points[valid_idx, 1], object_points[valid_idx, 2]]

            # Combine penalties
            combined_penalty = angular_penalty_factor + DBF_mag_penalty
            #combined_penalty = DBF_mag_penalty
            edge_weight[valid_idx, i] = combined_penalty

    return (nhood_nodes, edge_dist, edge_weight)


# Create euclidean distance graph and penalty graph
def create_graph(object_points, DBF, vect_mag, vec_dir, max_bound):
    """
    [INPUT]
    object_points : n x 3 point cloud format of the object
    DBF : distance to the boundary map
    max_bound : maximum coordinates for each dimension with padding

    [OUTPUT]
    G_dist : graph with euclidean distance as edges
    G : graph with penalty as edges
    """

    n = object_points.shape[0]

    #nhood_nodes, edge_dist, edge_weight = create_edges(image, object_points, max_bound)
    nhood_nodes, edge_dist, edge_weight = create_edges(object_points, DBF, vect_mag, vec_dir, max_bound)

    if np.max(edge_weight) < np.finfo(np.float16).max:
        edge_weight = edge_weight.astype(np.float16)

    valid_edge = np.where(nhood_nodes != -1)
    rowcol = (valid_edge[0], nhood_nodes[valid_edge[0], valid_edge[1]])

    print("Creating graph...")
    G_dist = csr_matrix((edge_dist[valid_edge[0], valid_edge[1]], rowcol), shape=(n, n), dtype=np.int32)
    G = csr_matrix((edge_weight[valid_edge[0], valid_edge[1]], rowcol), shape=(n, n), dtype=np.int32)

    return G_dist, G

# def find_vec_mag(max_bound, image):
#     mask = image[0, :, :, :]
#     mask = mask > 0.1
#
#     width, height, depth = mask.shape
#     vec_mag = np.zeros(max_bound)
#     vec_dir = np.zeros((*max_bound, 3))
#     print(vec_mag.shape)
#
#     pixel_count = 0
#
#     for x in range(width):
#         for y in range(height):
#             for z in range(depth):
#                 pixel = (x, y, z)
#                 # pixel = np.array(pixel)
#                 if mask[x, y, z] > 0:
#                     pixel_count += 1
#
#                     flow_vec = (image[1, x, y, z], image[2, x, y, z], image[3, x, y, z])
#                     flow_vec = np.array(flow_vec)
#                     vec_dir[x, y, z] = flow_vec
#                     flow_magnitude = np.linalg.norm(flow_vec)
#                     vec_mag[x, y, z] = flow_magnitude
#
#     return vec_mag, vec_dir

def find_vec_mag(max_bound, image, origin):

    print("DEBUG: image.shape =", image.shape)
    mask = image[0] > 0.5

    vec_field = np.stack((image[1], image[2], image[3]), axis=-1)
    flow_magnitude = np.linalg.norm(vec_field, axis=-1)

    vec_mag = np.zeros(max_bound, dtype=np.float32)
    vec_dir = np.zeros((*max_bound, 3), dtype=np.float32)

    x0, y0, z0 = origin

    mask_idx = np.where(mask)


    x_idx = mask_idx[0] + x0
    y_idx = mask_idx[1] + y0
    z_idx = mask_idx[2] + z0


    x_idx = np.clip(x_idx, 0, max_bound[0] - 1)
    y_idx = np.clip(y_idx, 0, max_bound[1] - 1)
    z_idx = np.clip(z_idx, 0, max_bound[2] - 1)

    vec_mag[x_idx, y_idx, z_idx] = flow_magnitude[mask]
    vec_dir[x_idx, y_idx, z_idx] = vec_field[mask]

    return vec_mag, vec_dir





# Modified TEASAR
def TEASAR(image, object_points, parameters, init_root, init_dest=None, soma=False):
    """
    [INPUT]
    object_points : n x 3 point cloud format of the object
    parameters : list of "scale" and "constant" parameters (first: "scale", second: "constant")
                 larger values mean less sensitive to detecting small branches
    init_root : initial root points (first object point if not defined)
    init_dest : destination points to find path to from given init_root (farthest points if not defined.)
    soma : boolean flag for additional processing (if needed)

    [OUTPUT]
    skeletons : list of skeleton objects (one per root)
    """

    NDIM = object_points.shape[1]
    n = object_points.shape[0]
    print('Number of points ::::: ' + str(n))

    max_bound = np.max(object_points, axis=0) + 2
    print(max_bound)

    object_points = object_points.astype(np.uint32)
    object_nodes = Nodes(object_points, max_bound)

    bin_im = np.zeros(max_bound, dtype='bool')
    bin_im[object_points[:, 0], object_points[:, 1], object_points[:, 2]] = True

    # Distance to the boundary map
    print("Creating DBF...")
    print(bin_im.shape)
    DBF = ndimage.distance_transform_edt(bin_im).astype(np.float32)
    origin = np.min(object_points, axis=0).astype(int)

    print("extracting vector magnitude and direction...")
    vec_mag, vec_dir = find_vec_mag(max_bound, image, origin)

    #vec_mag, vec_dir = find_vec_mag(max_bound, image)

    #G_dist, G = create_graph(image, object_points, max_bound)
    G_dist, G = create_graph(object_points, DBF, vec_mag, vec_dir, max_bound)

    # Define root nodes
    root_nodes = object_nodes.sub2node(init_root)
    n_root = root_nodes.shape[0]
    root_nodes = list(root_nodes)
    manual_root_nodes = root_nodes
    is_disconnected = np.ones(n, dtype='bool')

    r = 0
    c = 0
    nodes = np.array([])
    edges = np.array([])

    # Store shortest paths
    best_path = []
    best_costs = {}
    best_paths = {}
    D_G_all = {}
    pred_G_all = {}
    high_difference_edges = []

    # penalty_threshold = 10000
    # high_penalty_edge_count = 0
    start_time = time()
    G_rev = G.transpose().tocsr()
    for r_idx, root in enumerate(root_nodes):
        #D_G, pred_G = dijkstra(G, directed=True, indices=root, return_predecessors=True)

        D_G, pred_G = dijkstra(G_rev, directed=True, indices=root, return_predecessors=True)

        D_G_all[r_idx] = D_G  # Store distances for each root
        pred_G_all[r_idx] = pred_G  # Store predecessors for each root

    print("all nodes path is found for all roots")
    end_time = time()
    processing_time = end_time - start_time
    print(f"processing time:{processing_time}")

    r_min = np.min(DBF)
    r_max = np.max(DBF)


    # When destination nodes are not set, process connected components dynamically
    if init_dest is None or init_dest.shape[0] == 0:
        while np.any(is_disconnected):
            #print("Processing connected component...")
            n_root = len(root_nodes)

            if r <= n_root - 1:
                root = root_nodes[0]
                D_Gdist = dijkstra(G_dist, directed=True, indices=root)

                cnt_comp = ~np.isinf(D_Gdist)
                is_disconnected = is_disconnected * ~cnt_comp


                cnt_comp = np.where(cnt_comp)[0]
                root_in_component = [root for root in root_nodes if root in cnt_comp]

                print(f"Root nodes in the current connected component: {root_in_component}")
                root_nodes = [root for root in root_nodes if root not in root_in_component]
                r = r + len(root_in_component)
                cnt_comp_im = np.zeros(max_bound, dtype='bool')
                cnt_comp_im[object_points[cnt_comp, 0], object_points[cnt_comp, 1], object_points[cnt_comp, 2]] = 1

                if cnt_comp.shape[0] <= 1:
                    #r = r + 1
                    continue

                # Build skeleton and remove pieces that are completed.
                # Iterate until the entire connected component is completed.
                path_list = []
                while np.any(cnt_comp):
                    dest_node = cnt_comp[np.where(D_Gdist[cnt_comp] == np.max(D_Gdist[cnt_comp]))[0][0]]


                    best_cost = float('inf')
                    best_root = None

                    for r_idx, root in enumerate(manual_root_nodes):
                        D_G = D_G_all[r_idx]
                        pred_G = pred_G_all[r_idx]

                        #start = time()
                        path = find_path(pred_G, dest_node)
                        #end = time()
                        #print(f"find_path time: {end - start:.6f} sec")

                        #path = find_path(dest_node, pred_G)
                        # penalty_threshold = 180
                        # path = find_path_with_threshold(pred_G, root, dest_node, G, penalty_threshold)
                        #print(path)
                        cost = D_G[dest_node]
                        #print(cost)
                        #sys.exit()


                        if cost <= best_cost and path is not None:
                            best_cost = cost
                            best_path = path
                            path_list.append(best_path)
                            # print(best_path)
                            # print(cost)

                            best_root = root


                            if cost <= best_cost and path is not None:
                                best_cost = cost
                                best_path = path
                                path_list.append(best_path)


                                #print("Analyzing cost differences along the path...")

                                cost_threshold = 1000000

                                for i in range(len(best_path) - 1):
                                    node_current = best_path[i]
                                    node_next = best_path[i + 1]

                                    cost_diff = D_G[node_next] - D_G[node_current]


                                    if abs(cost_diff) > cost_threshold:
                                        high_difference_edges.append((node_current, node_next, cost_diff))

                                best_root = root


                    if best_path.size == 0:
                        print("no best path found for a node")
                        break


                    best_costs[dest_node] = best_cost
                    best_paths[dest_node] = (best_path, best_root)


                    # Remove completed parts of the component
                    for i in range(len(best_path)):
                        path_node = best_path[i]
                        path_point = object_points[path_node, :]

                        radius = DBF[path_point[0], path_point[1], path_point[2]]
                        if parameters["adaptive"]:
                            d = adaptive_thr_linear(
                                radius,
                                r_min,
                                r_max,
                                parameters["scale"],
                                parameters["const"]
                            )
                        else:
                            d = thr_linear(
                                DBF[path_point[0], path_point[1], path_point[2]],
                                parameters["scale"],
                                parameters["const"],
                                500)

                        # #d = thr_linear(DBF[path_point[0], path_point[1], path_point[2]], parameters, 500)
                        # radius = DBF[path_point[0], path_point[1], path_point[2]]
                        # d = adaptive_thr_linear(radius, r_min=r_min, r_max=r_max)

                        cube_min = np.zeros(3, dtype=np.uint32)
                        cube_min = path_point - d
                        cube_min[cube_min < 0] = 0
                        cube_min = cube_min.astype(np.uint32)

                        cube_max = np.zeros(3, dtype=np.uint32)
                        cube_max = path_point + d
                        cube_max = cube_max.astype(np.uint32)
                        cube_max = np.minimum(cube_max, max_bound)

                        cnt_comp_im[cube_min[0]:cube_max[0], cube_min[1]:cube_max[1], cube_min[2]:cube_max[2]] = 0

                    cnt_comp_sub = array2point(cnt_comp_im, object_id=None)
                    cnt_comp = object_nodes.sub2node(cnt_comp_sub)

            else:
                disconnected_indices = np.where(is_disconnected == 1)[0]

                # Get the corresponding radii from the distance transform
                disconnected_radii = DBF[object_points[disconnected_indices, 0], object_points[disconnected_indices, 1],object_points[disconnected_indices, 2]]

                # Choose the index of the node with the largest radii
                largest_radius_index = np.argmax(disconnected_radii)
                root = disconnected_indices[largest_radius_index]
                #root = np.where(is_disconnected == 1)[0][0]
                manual_root_nodes.append(root)
                D_Gdist = dijkstra(G_dist, directed=True, indices=root)

                cnt_comp = ~np.isinf(D_Gdist)
                is_disconnected = is_disconnected * ~cnt_comp
                cnt_comp = np.where(cnt_comp)[0]
                cnt_comp_im = np.zeros(max_bound, dtype='bool')
                cnt_comp_im[object_points[cnt_comp, 0], object_points[cnt_comp, 1], object_points[cnt_comp, 2]] = 1
                #D_G, pred_G = dijkstra(G, directed=True, indices=root, return_predecessors=True)
                G_rev = G.transpose().tocsr()
                D_G, pred_G = dijkstra(G_rev, directed=True, indices=root, return_predecessors=True)

                if cnt_comp.shape[0] <= 1:
                    # r = r + 1
                    continue

                # Build skeleton and remove pieces that are completed.
                # Iterate until the entire connected component is completed.
                path_list = []
                while np.any(cnt_comp):
                    dest_node = cnt_comp[np.where(D_Gdist[cnt_comp] == np.max(D_Gdist[cnt_comp]))[0][0]]
                    #print(dest_node)
                    path = find_path(pred_G, dest_node)
                    cost = D_G[dest_node]

                    best_costs[dest_node] = cost
                    best_paths[dest_node] = (path, root)
                    path_list.append(path)
                    # print(dest_node)

                    # Remove completed parts of the component
                    for i in range(len(path)):
                        path_node = path[i]
                        path_point = object_points[path_node, :]

                        radius = DBF[path_point[0], path_point[1], path_point[2]]
                        if parameters["adaptive"]:
                            d = adaptive_thr_linear(
                                radius,
                                r_min,
                                r_max,
                                parameters["scale"],
                                parameters["const"]
                            )
                        else:
                            d = thr_linear(
                                DBF[path_point[0], path_point[1], path_point[2]],
                                parameters["scale"],
                                parameters["const"],
                                500)

                        # #d = thr_linear(DBF[path_point[0], path_point[1], path_point[2]], parameters, 500)
                        #radius = DBF[path_point[0], path_point[1], path_point[2]]
                        # d = adaptive_thr_linear(radius, r_min=r_min, r_max=r_max)

                        cube_min = np.zeros(3, dtype=np.uint32)
                        cube_min = path_point - d
                        cube_min[cube_min < 0] = 0
                        cube_min = cube_min.astype(np.uint32)

                        cube_max = np.zeros(3, dtype=np.uint32)
                        cube_max = path_point + d
                        cube_max = cube_max.astype(np.uint32)
                        cube_max = np.minimum(cube_max, max_bound)

                        cnt_comp_im[cube_min[0]:cube_max[0], cube_min[1]:cube_max[1], cube_min[2]:cube_max[2]] = 0

                    cnt_comp_sub = array2point(cnt_comp_im, object_id=None)
                    cnt_comp = object_nodes.sub2node(cnt_comp_sub)



            c = c + 1

    # Build separate trees
    skeletons = []
    root_associations = {}
    # for root in root_nodes:
    #     manual_root_nodes.append(root)
    #print(manual_root_nodes)
    manual_root_coordinates = object_points[np.array(manual_root_nodes), :]
    #print(manual_root_coordinates)
    #print("High difference edges:")
    print(f'high difference edges: {len(high_difference_edges)}')
    for root in manual_root_nodes:
        root_skeleton_paths = [path for node, (path, r) in best_paths.items() if r == root]

        if not root_skeleton_paths:
            continue  # Skip if no paths are associated with this root

        #print(f"Building tree from root {root}...")

        # Combine all paths for this root to form a tree
        nodes = np.concatenate(root_skeleton_paths)
        edges = np.concatenate([path2edge(path) for path in root_skeleton_paths])

        # Consolidate nodes and edges
        nodes = np.unique(nodes)
        edges = np.unique(edges, axis=0)

        # filtered_edges = []
        #
        # for edge in edges:
        #     node1, node2 = edge
        #     penalty = G[node1, node2]
        #     if penalty >= penalty_threshold:
        #         filtered_edges.append(edge)
        #     else:
        #         high_penalty_edge_count += 1
        #
        # print(f"number of false merges:{high_penalty_edge_count}")
        # filtered_edges = np.array(filtered_edges)
        for node in nodes:
            root_associations[node] = root

        skel_nodes = object_points[nodes, :]
        skel_edges = reorder_nodes(nodes, edges)
        skel_edges = skel_edges.astype('uint32')
        skel_radii = DBF[skel_nodes[:, 0], skel_nodes[:, 1], skel_nodes[:, 2]]
        skel_root = np.array([root_associations[node] for node in nodes])

        skeleton = Skeleton(skel_nodes, skel_edges, skel_radii, skel_root)
        skeleton = consolidate_skeleton(skeleton)

        skeletons.append(skeleton)

    return skeletons








# Skeletonization
def skeletonize(image, object_input, object_id=1, dsmp_resolution=[1, 1, 1], parameters=[0.5, 4], init_root=[],
                init_dest=[], soma=False):
    """
    [INPUT]
    object_input : object to skeletonize (N x 3 point cloud or 3D labeled array)
    object_id : object ID to skeletonize (Don't need this if object_input is in point cloud format)
    dsmp_resolution : downsample resolution
    parameters : list of "scale" and "constant" parameters (first: "scale", second: "constant")
                 larger values mean less senstive to detecting small branches
    init_roots : N x 3 array of initial root coordinates

    [OUTPUT]
    skeleton : skeleton object
    """

    # Don't run skeletonization if the input is an empty array
    print(object_input.shape)
    print(image.shape)
    if object_input.shape[0] == 0:
        return Skeleton()

    init_root = np.array(init_root)
    print(init_root)
    init_dest = np.array(init_dest)

    # Convert object_input to point cloud format
    if object_input.shape[1] == 3:  # object input: point cloud
        obj_points = object_input
    else:  # object input: 3D array
        obj_points = array2point(object_input, object_id=None)

    print(len(obj_points))
    # max_bound = np.max(obj_points, axis=0)
    # print(max_bound)
    # If initial roots is empty, take the first point
    if len(init_root) == 0:
        init_root = obj_points[0, :]

    # If it is not empty, find the closest node in the object.
    else:
        for i in range(init_root.shape[0]):
            root = init_root[i, :]
            root_idx = find_row(obj_points, root)

            if root_idx == -1:
                dist = np.sum((obj_points - root) ** 2, 1)
                print(len(obj_points))
                print(root)
                print(len(dist))
                root = obj_points[np.argmin(dist), :]
                init_root[i, :] = root

    # Same for destinations
    for i in range(init_dest.shape[0]):
        dest = init_dest[i, :]
        dest_idx = find_row(obj_points, dest)

        if dest_idx == -1:
            dist = np.sum((obj_points - dest) ** 2, 1)
            dest = obj_points[np.argmin(dist), :]
            init_dest[i, :] = dest

    # Downsample points
    if sum(dsmp_resolution) > 3:
        print(">>>>> Downsample...")
        obj_points = downsample_points(obj_points, dsmp_resolution)
        init_root = downsample_points(init_root, dsmp_resolution)

        if init_dest.shape[0] != 0:
            init_dest = downsample_points(init_dest, dsmp_resolution)

    # Convert coordinates to bounding box
    # min_bound = np.min(obj_points, axis=0)
    # obj_points = obj_points - min_bound + 1
    # init_root = init_root - min_bound + 1
    #
    # max_bound = np.max(obj_points, axis=0)
    # print(max_bound)
    # if init_dest.shape[0] != 0:
    # 	init_dest = init_dest - min_bound + 1

    # Skeletonize chunk surrounding object
    print(">>>>> Building skeleton...")
    t0 = time()
    skeletons = TEASAR(image, obj_points, parameters, init_root, init_dest, soma)
    t1 = time()
    print(">>>>> Elapsed time : " + str(np.round(t1 - t0, decimals=3)))

    # Convert coordinates back into original coordinates
    for skeleton in skeletons:
        if skeleton.nodes.shape[0] != 0:
            skeleton.nodes = upsample_points(skeleton.nodes, dsmp_resolution)

    return skeletons
