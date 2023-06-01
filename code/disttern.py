import matplotlib.pyplot as plt
import numpy as np


def proportion_to_point(prop, right_axis=False):
	'''
	Given a proportion along the left or right triangle axis, return its
	cartesian coordinates.
	'''
	h = (1 - prop)
	x = h * np.cos(60 * np.pi / 180)
	y = h * np.sin(60 * np.pi / 180)
	if right_axis:
		x = 1 - x
	return x, y

def intersection(data):
	'''
	Given two line segments, return the point at which they intersect.
	'''
	L = len(data)
	x1, y1, x2, y2 = data.reshape(L * 2, -1).T
	R = np.full([L, 2], np.nan)
	X = np.concatenate([
		(y2 - y1).reshape(L * 2, -1), 
		(x1 - x2).reshape(L * 2, -1)], 
		axis=1
	).reshape(L, 2, 2)
	B = (x1 * y2 - x2 * y1).reshape(L, 2)
	I = np.isfinite(np.linalg.cond(X))
	R[I] = np.matmul(np.linalg.inv(X[I]), B[I][:,:,None]).squeeze(-1)
	return R[0]

def calculate_proportions(reference_objects, target_objects, distance_func):
	'''
	Given three reference objects (placed at the top, left, and right vertices
	of the triangle) and a collection of target objects, compute the
	proportions along the left and right axes that describe where the
	target objects should be located.
	'''
	assert len(reference_objects) == 3
	props = []
	for target in target_objects:
		dist_to_top = distance_func(target, reference_objects[0])
		dist_to_left = distance_func(target, reference_objects[1])
		dist_to_right = distance_func(target, reference_objects[2])
		prop_left = dist_to_top / (dist_to_top + dist_to_left)
		prop_right = dist_to_top / (dist_to_top + dist_to_right)
		props.append((prop_left, prop_right))
	return props

def draw_ternary_axes(axis):
	'''
	Draw the ternary plot axes and grid lines.
	'''
	height = np.sqrt(3) / 2
	centroid = 0.5, height / 3
	ref_l = proportion_to_point(0.5, right_axis=False)
	ref_r = proportion_to_point(0.5, right_axis=True)
	# axis.add_patch( plt.Polygon([(0.5, height), (0, 0), (1, 0)], color='black', fill=False, linewidth=2) )

	axis.add_patch( plt.Polygon([(0.5, height), ref_l, centroid, ref_r], fill=True, color='black', alpha=0.4, linewidth=0) )
	axis.add_patch( plt.Polygon([(0, 0), (0.5, 0), centroid, ref_l], fill=True, color='gray', alpha=0.4, linewidth=0) )
	axis.add_patch( plt.Polygon([(1, 0), ref_r, centroid, (0.5, 0)], fill=True, color='lightgray', alpha=0.4, linewidth=0) )

	# axis.plot([ref_l[0], centroid[0]], [ref_l[1], centroid[1]], color='green')
	# axis.plot([ref_r[0], centroid[0]], [ref_r[1], centroid[1]], color='gray')
	# axis.plot([0.5, centroid[0]], [0, centroid[1]], color='gray')

def make_ternary_plot(reference_objects, target_objects, distance_func, color='MediumSeaGreen', jitter=False):
	'''
	Given three reference objects (placed at the top, left, and right vertices
	of the triangle), a collection of target objects, and a distance
	function, make a distance-based ternary plot. This is a little
	different to an ordinary ternary plot; proximity from the three
	vertices represents distance from three reference objects.
	'''
	props = calculate_proportions(reference_objects, target_objects, distance_func)
	points = np.zeros((len(target_objects), 2), dtype=float)
	for i, (prop_l, prop_r) in enumerate(props):
		x1_l, y1_l = proportion_to_point(prop_l, right_axis=False)
		x1_r, y1_r = proportion_to_point(prop_r, right_axis=True)
		points[i] = intersection(np.array([[(x1_l, y1_l), (1, 0), (x1_r, y1_r), (0, 0)]]))
	
	fig, axis = plt.subplots(1, 1)
	draw_ternary_axes(axis)
	if jitter:
		points[:, 0] += (np.random.random(len(target_objects)) - 0.5) * 0.01
		points[:, 1] += (np.random.random(len(target_objects)) - 0.5) * 0.01
	axis.scatter(points[:, 0], points[:, 1], color=color, alpha=0.5, edgecolor='white')
	axis.axis('off')
	fig.tight_layout()
	plt.show()
