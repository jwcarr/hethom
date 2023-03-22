import numpy as np
import cairocffi as cairo
import voi

def make_matrix(lexicon, suffix_spellings=None, m=4, n=4):
	matrix = np.zeros((m, n), dtype=int)
	if suffix_spellings is None:
		suffix_spellings = sorted(list(set([word[3:] for word in lexicon.values()])))
	for i in range(m):
		for j in range(n):
			item = f'{i}_{j}'
			word_i = suffix_spellings.index(lexicon[item][3:])
			matrix[i, j] = word_i
	return matrix

def generate_color_palette(matrix):
	n_spellings = len(np.unique(matrix))
	hues = np.linspace(0, 2 * np.pi, n_spellings + 1)
	sats = np.linspace(0, 2 * np.pi, n_spellings + 1)
	brts = np.linspace(0, 2 * np.pi, n_spellings + 1)
	colors = [hsv_to_rgb(h, 0.7, 0.8) for h in hues[:-1]]
	np.random.shuffle(colors)
	return colors

def hsv_to_rgb(h, s, v):
	if s == 0.0:
		return v, v, v
	h /= 2 * np.pi
	i = int(h * 6.0)
	f = (h * 6.0) - i
	p = v * (1.0 - s)
	q = v * (1.0 - s * f)
	t = v * (1.0 - s * (1.0 - f))
	match i % 6:
		case 0:
			return v, t, p
		case 1:
			return q, v, p
		case 2:
			return p, v, t
		case 3:
			return p, q, v
		case 4:
			return t, p, v
		case 5:
			return v, p, q

def draw(matrix, color_palette, output_path):
	surface = cairo.PDFSurface(output_path, 4, 4)
	context = cairo.Context(surface)
	for i, row in enumerate(matrix):
		for j, cell in enumerate(row):
			color = color_palette[cell]
			context.set_source_rgb(*color)
			context.rectangle(j, i, 1, 1)
			context.fill_preserve()
			context.set_source_rgb(1, 1, 1)
			context.set_line_width(0.05)
			context.stroke()
	surface.finish()

def typological_distribution(matrix, probabilistic=False):
	dist = [voi.variation_of_information(matrix, system) for name, system in typology.items()]
	if probabilistic is False:
		return dist
	dist = np.array([np.exp2(-1 * val**2) for val in dist])
	return dist / dist.sum()

typology = {
	'transparent': np.array([
		[0, 0, 0, 0],
		[0, 0, 0, 0],
		[0, 0, 0, 0],
		[0, 0, 0, 0],
	], dtype=int),
	'holistic': np.array([
		[ 0, 1, 2, 3],
		[ 4, 5, 6, 7],
		[ 8, 9,10,11],
		[12,13,14,15],
	], dtype=int),
	'redundant': np.array([
		[1, 1, 1, 1],
		[2, 2, 2, 2],
		[3, 3, 3, 3],
		[0, 0, 0, 0],
	], dtype=int),
	'expressive': np.array([
		[0, 1, 2, 3],
		[0, 1, 2, 3],
		[0, 1, 2, 3],
		[0, 1, 2, 3],
	], dtype=int),
}


if __name__ == '__main__':

	for name, system in typology.items():
		color_palette = generate_color_palette(system)
		print(color_palette)
		draw(system, color_palette, f'/Users/jon/Desktop/{name}.pdf')
