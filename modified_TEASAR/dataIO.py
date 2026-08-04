import scipy.io as spio
import _pickle as pickle

def save_skeleton(skeleton, filename='./skeleton.pkl'):
	
	SkeletonDict = {}
	SkeletonDict["nodes"] = skeleton.nodes.astype('uint16')
	SkeletonDict["edges"] = skeleton.edges.astype('uint32')
	SkeletonDict["radii"] = skeleton.radii.astype('float32')
	SkeletonDict["root"] = skeleton.root.astype('uint16')

	with open(filename, 'wb') as output:
		pickle.dump(SkeletonDict, output, pickle.HIGHEST_PROTOCOL)


def load_skeleton(filename):

	with open(filename, 'rb') as input:
		skeleton = pickle.load(input)

	return skeleton


##### Python/Matlab #####
def load_points_mat(mat_file):

	mat = spio.loadmat(mat_file)
	p = mat['p']

	return p


def save_skeleton_mat(skeletons, out_filename='./skeleton.mat'):
	"""
    Saves one or multiple skeleton objects to a .mat file.

    Parameters:
    - skeletons : list of Skeleton objects or a single Skeleton object
    - out_filename : str, output file path (default is './skeleton.mat')
    """
	if isinstance(skeletons, list):
		# Handle multiple skeletons
		skeleton_data = {}
		for idx, skel in enumerate(skeletons):
			# Save each skeleton under a unique key
			skeleton_data[f'skeleton_{idx}'] = {
				'nodes': skel.nodes,
				'edges': skel.edges,
				'radii': skel.radii,
				'root': skel.root
			}
	else:
		# Handle a single skeleton
		skeleton_data = {
			'nodes': skeletons.nodes,
			'edges': skeletons.edges,
			'radii': skeletons.radii,
			'root' : skeletons.root
		}

	# Save to .mat file
	spio.savemat(out_filename, skeleton_data)