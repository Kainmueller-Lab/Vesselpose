# Import necessary packages
import numpy as np


# Skeleton 
class Skeleton:

	def __init__(self, nodes=np.array([]), edges=np.array([]), radii=np.array([]), root=np.array([])):

		if nodes.size == 0:
			nodes = nodes.reshape((0,3))

		if edges.size == 0:
			edges = edges.reshape((0,3))

		self.nodes = nodes
		self.edges = edges
		self.radii = radii
		self.root = root

	def empty(self):

		return self.nodes.size == 0 or self.edges.size == 0


# Nodes
class Nodes:

	def __init__(self, coord, max_bound):
		
		n = coord.shape[0]

		coord = coord.astype(np.uint32)
		max_bound = max_bound.astype(np.uint32)
		self.max_bound = max_bound
		
		idx = coord[:,0] + max_bound[0]*coord[:,1] + max_bound[0]*max_bound[1]*coord[:,2]
		
		idx2node = np.ones(np.prod(max_bound), dtype=np.int32)*-1
		idx2node[idx] = np.arange(coord.shape[0], dtype=np.int32)
		self.node = idx2node

	def sub2idx(self, sub_array):

		if len(sub_array.shape) == 1:
			sub_array = np.reshape(sub_array,(1,3))

		sub_array = sub_array.astype(np.uint32)

		max_bound = self.max_bound
		
		return sub_array[:,0] + max_bound[0]*sub_array[:,1] + max_bound[0]*max_bound[1]*sub_array[:,2]

	def sub2node(self, sub_array):
		
		idx_array = self.sub2idx(sub_array)

		return self.node[idx_array].astype(np.int32)


# Convert array to point cloud
def array2point(array, object_id):
	"""
	[INPUT]
	array : array with labels
	object_id : object label to extract point cloud

	[OUTPUT]
	points : n x 3 point coordinates 
	"""
	# print(f"Array shape: {array.shape}")
	# print(f"Unique labels in array: {np.unique(array)}")

	if object_id is None:
		object_coord = np.where(array > 0)
	else:
		object_coord = np.where(array == object_id)

	#print(len(object_coord))

	object_x = object_coord[0]
	object_y = object_coord[1]
	object_z = object_coord[2]

	points = np.zeros([len(object_x),3], dtype=np.int64)
	points[:,0] = object_x
	points[:,1] = object_y
	points[:,2] = object_z

	return points


# Downsample points
def downsample_points(points, dsmp_resolution):
	"""
	[INPUT]
	points : n x 3 point coordinates
	dsmp_resolution : [x, y, z] downsample resolution

	[OUTPUT]
	point_downsample : n x 3 downsampled point coordinates
	"""

	if len(points.shape) == 1:
			points = np.reshape(points,(1,3))

	dsmp_resolution = np.array(dsmp_resolution, dtype=np.float)

	point_downsample = points/dsmp_resolution

	point_downsample = np.round(point_downsample)
	point_downsample = np.unique(point_downsample, axis=0)

	return point_downsample.astype(np.int32)


# Upsample points
def upsample_points(points, dsmp_resolution):
	"""
	[INPUT]
	points : n x 3 point coordinates
	dsmp_resolution : [x, y, z] downsampled resolution

	[OUTPUT]
	point_upsample : n x 3 upsampled point coordinates
	"""

	dsmp_resolution = np.array(dsmp_resolution)
	
	point_upsample = points*dsmp_resolution
	
	return point_upsample.astype(np.int32)

	
# Find corresponding row 
def find_row(array, row):
	"""
	[INPUT]
	array : array to search for
	row : row to find

	[OUTPUT]
	idx : row indices
	"""

	row = np.array(row)

	if array.shape[1] != row.size:
		raise ValueError("Dimension do not match!")
	
	NDIM = array.shape[1]

	row_loc = np.ones((array.shape[0],), dtype='bool')
	
	for i in range(NDIM):
		valid = array[:,i] == row[i]
		row_loc = row_loc * valid

	idx = np.where(row_loc==1)[0]

	if len(idx) == 0:
		idx = -1

	return idx


# Find path from predecessor matrix
def find_path(predecessor, end, start = []):
	"""
	[INPUT]
	predecessor : n x n array of predecessors of shortest path from i to j
	end : destination node
	start : start node (Not necessary if the predecessor array is 1D array)

	[OUTPUT]
	path : n x 1 array consisting nodes in path
	"""

	path_list = [end]
	pred = end

	while True:
		if len(predecessor.shape) > 1:
			pred = predecessor[start,pred]

		else:
			pred = predecessor[pred]

		if pred == -9999:
			break
		else:
			path_list.append(pred)

	path_list.reverse()

	return np.array(path_list)

# def find_path(predecessor, end, start=None):
#
#     max_steps = predecessor.shape[-1] if predecessor.ndim > 1 else len(predecessor)
#     path = np.empty(max_steps, dtype=np.int32)
#
#     idx = 0
#     node = end
#
#     while True:
#         path[idx] = node
#
#         if predecessor.ndim > 1:
#             if start is None:
#                 raise ValueError("start must be provided for 2D predecessor array")
#             node = predecessor[start, node]
#         else:
#             node = predecessor[node]
#
#         if node == -9999:
#             break
#
#         idx += 1
#         if idx >= max_steps:
#             print(f"Warning: path too long or corrupted, truncating at {max_steps}")
#             break
#
#     return path[:idx + 1][::-1]


def find_path_with_threshold(pred_G, root, dest_node, G, penalty_threshold):
	"""

    Custom pathfinding function that stops and discards a path if any edge's penalty exceeds the threshold.

    pred_G : Predecessor list from Dijkstra's algorithm
    root : The starting root node
    dest_node : The destination node to reach
    G : Penalty-weighted graph (with individual edge penalties)
    penalty_threshold : The threshold for edge penalties
    """
	path = [dest_node]
	node = dest_node

	# Traverse the path backwards from the destination to the root using the predecessor array
	while node != root:
		prev_node = pred_G[node]
		if prev_node == -9999:  # This indicates no valid predecessor
			return   # No valid path found

		# Check the penalty for the edge between `prev_node` and `node`
		penalty = G[prev_node, node]

		# If the penalty exceeds the threshold, discard the path
		if penalty < penalty_threshold:
			print(f"Edge penalty {penalty} exceeds threshold {penalty_threshold}. Discarding path.")
			return []  # Break and discard the path

		# If the penalty is below the threshold, add the node to the path
		path.append(node)
		node = prev_node

	path.append(root)  # Add the root node
	path.reverse()  # Reverse the path to get it in root-to-destination order
	return np.array(path)


# Thresholded linear function (saturated linear)
def thr_linear(x, slope, const, threshold):
	"""
	[INPUT]
	x : function input
	parameters : [slope, constant] (y = slope*x + constant)
	threshold : threshold or cutoff

	[OUTPUT]
	y : function output
	"""

	#slope, const = linear_parameters

	return min(x * slope + const, threshold)


#scale_range=(0.3, 0.8),  const_range=(2, 7) #VF synthetic data
#scale_range=(0.7, 1.1) , (0.3, 1.5)  const_range=(6, 10), (2, 14) #Trexplorer synthetic data
def adaptive_thr_linear(radius, r_min, r_max, scale_range,  const_range, threshold=500):
    radius_clipped = np.clip(radius, r_min, r_max)
    frac = (radius_clipped - r_min) / (r_max - r_min)

    scale = scale_range[0] + frac * (scale_range[1] - scale_range[0])
    const = const_range[0] + frac * (const_range[1] - const_range[0])

    d = scale * radius + const
    return min(d, threshold)




# Reorder nodes so there is no unused node ids
def reorder_nodes(nodes, edges):
	"""
	[INPUT]
	nodes : list of node numbers
	edges : list of edges

	[OUTPUT]
	edges_reorder : edges with reordered node numbers
	"""

	edges_reorder = np.zeros(edges.shape)
	for i in range(edges.shape[0]):
		edges_reorder[i,0] = np.where(nodes==edges[i,0])[0]
		edges_reorder[i,1] = np.where(nodes==edges[i,1])[0]

	return edges_reorder


# Convert n x 1 path array to list of edges
def path2edge(path):
	"""
	[INPUT]
	path : sequence of nodes (n x 1 array)

	[OUTPUT]
	edges : list of edges that form path (n x 2 array)
	"""

	edges = np.zeros([len(path)-1,2], dtype=np.uint32)
	for i in range(len(path)-1):
		edges[i,0] = path[i]
		edges[i,1] = path[i+1]

	return edges


def consolidate_skeleton(skeleton):

	nodes = skeleton.nodes 
	edges = skeleton.edges
	radii = skeleton.radii

	if nodes.shape[0] == 0 or edges.shape[0] == 0:
		skeleton = Skeleton()

	else:
		# Remove duplicate nodes
		unique_nodes, unique_idx, unique_counts = np.unique(nodes, axis=0, return_index=True, return_counts=True)
		unique_edges = np.copy(edges)

		dup_idx = np.where(unique_counts>1)[0]
		for i in range(dup_idx.shape[0]):
			dup_node = unique_nodes[dup_idx[i],:]
			dup_node_idx = find_row(nodes, dup_node)

			for j in range(dup_node_idx.shape[0]-1):
				start_idx, end_idx = np.where(edges==dup_node_idx[j+1])
				unique_edges[start_idx, end_idx] = unique_idx[dup_idx[i]]


		# Remove unnecessary nodes
		eff_node_list = np.unique(unique_edges)
		eff_node_list = eff_node_list.astype('int')
		
		eff_nodes = nodes[eff_node_list]
		eff_radii = radii[eff_node_list]

		eff_edges = np.copy(unique_edges)
		for i, node in enumerate(eff_node_list, 0):
			row_idx, col_idx = np.where(unique_edges==node)

			eff_edges[row_idx,col_idx] = i

		eff_edges = np.unique(eff_edges, axis=0)

		skeleton.nodes = eff_nodes
		skeleton.edges = eff_edges
		skeleton.radii = eff_radii


	return skeleton

