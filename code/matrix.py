import numpy as np
import cairocffi as cairo
import re


RE_STEM_SUFFIX = re.compile(r'^(buvi|zeti|wopi|gafi)(\w*)$')


def parse_stem_and_suffix(word):
	word_match = RE_STEM_SUFFIX.match(word)
	if word_match is None:
		raise ValueError('Cannot parse stem and suffix')
	stem = word_match.group(1)
	suffix = word_match.group(2) or '∅'
	return stem, suffix

def get_suffix_spellings(lexicon):
	suffixes = []
	for word in lexicon.values():
		stem, suffix = parse_stem_and_suffix(word)
		suffixes.append(suffix)
	return sorted(list(set(suffixes)))

def make_matrix(lexicon, m=3, n=3):
	suffix_spellings = get_suffix_spellings(lexicon)
	matrix = np.zeros((m, n), dtype=int)
	for i in range(m):
		for j in range(n):
			stem, suffix = parse_stem_and_suffix(lexicon[ f'{i}_{j}' ])
			matrix[i, j] = suffix_spellings.index(suffix)
	return matrix

def make_matrix_with_cp(lexicon, ss, m=3, n=3):
	suffix_spellings = ss
	matrix = np.zeros((m, n), dtype=int)
	for i in range(m):
		for j in range(n):
			stem, suffix = parse_stem_and_suffix(lexicon[ f'{i}_{j}' ])
			matrix[i, j] = suffix_spellings.index(suffix)
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

def generate_color_palette_many(chain):
	suffix_spellings = []
	for subject_a, _ in chain:
		suffix_spellings.extend(get_suffix_spellings(subject_a['lexicon']))
	suffix_spellings = sorted(list(set(suffix_spellings)))
	first_letters = [suffix[0] for suffix in suffix_spellings]
	suffix_clusters = {first_letter: [s for s in suffix_spellings if s[0] == first_letter] for first_letter in first_letters}
	first_letter_hues = np.linspace(0, 2 * np.pi, len(suffix_clusters) + 1)
	max_cluster_size = max([len(cluster) for cluster in suffix_clusters.values()])
	hue_increment = ((first_letter_hues[1] - first_letter_hues[0]) * 0.8) / max_cluster_size
	suffix_spellings = []
	hues = []
	for cluster, base_hue in zip(suffix_clusters.values(), first_letter_hues):
		for i, suffix in enumerate(cluster):
			suffix_spellings.append(suffix)
			hues.append(base_hue + i * hue_increment)
	hues = np.array(hues)
	hues += np.random.random() * 2*np.pi
	hues[ hues > 2*np.pi ] -= 2*np.pi
	return suffix_spellings, {suffix_spellings.index(suffix): hsv_to_rgb(hue, 0.7, 0.9) for suffix, hue in zip(suffix_spellings, hues)}


def draw(matrix, color_palette, output_path):
	m, n = matrix.shape
	surface = cairo.PDFSurface(output_path, m, n)
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


reference_systems = {
	'transparent': np.array([
		[0, 0, 0],
		[0, 0, 0],
		[0, 0, 0],
	], dtype=int),
	'holistic': np.array([
		[0, 1, 2],
		[3, 4, 5],
		[6, 7, 8],
	], dtype=int),
	'redundant': np.array([
		[0, 0, 0],
		[1, 1, 1],
		[2, 2, 2],
	], dtype=int),
	'expressive': np.array([
		[0, 1, 2],
		[0, 1, 2],
		[0, 1, 2],
	], dtype=int),
}


if __name__ == '__main__':

	for name, system in reference_systems.items():
		color_palette = generate_color_palette(system)
		draw(system, color_palette, f'/Users/jon/Desktop/{name}.pdf')
