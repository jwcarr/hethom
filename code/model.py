import numpy as np
import voi


transparent = np.array([
	[0, 0, 0, 0],
	[0, 0, 0, 0],
	[0, 0, 0, 0],
	[0, 0, 0, 0],
], dtype=int)

holistic = np.array([
	[ 0, 1, 2, 3],
	[ 7, 6, 5, 4],
	[ 8, 9,10,11],
	[15,14,13,12],
], dtype=int)

redundant = np.array([
	[1, 1, 1, 1],
	[2, 2, 2, 2],
	[3, 3, 3, 3],
	[0, 0, 0, 0],
], dtype=int)

expressive = np.array([
	[0, 1, 2, 3],
	[0, 1, 2, 3],
	[0, 1, 2, 3],
	[0, 1, 2, 3],
], dtype=int)

typology = [transparent, holistic, redundant, expressive]


output = np.array([
	[0, 0, 0, 0],
	[0, 0, 0, 0],
	[1, 1, 1, 1],
	[1, 1, 1, 1],
], dtype=int)


print([voi.variation_of_information(output, grammar) for grammar in typology])
