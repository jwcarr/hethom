import numpy as np
import cairocffi as cairo

font_size = 50
helvetica = cairo.ToyFontFace('Helvetica', cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
helvetica_sf = cairo.ScaledFont(helvetica, cairo.Matrix(xx=font_size, yy=font_size))

suffix_font_size = 8
suffix_helvetica = cairo.ToyFontFace('Helvetica', cairo.FONT_SLANT_OBLIQUE, cairo.FONT_WEIGHT_NORMAL)
suffix_helvetica_sf = cairo.ScaledFont(suffix_helvetica, cairo.Matrix(xx=suffix_font_size, yy=suffix_font_size))

sound_font_size = 10
suffix_doulos = cairo.ToyFontFace('Doulos SIL', cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
suffix_doulos_sf = cairo.ScaledFont(suffix_doulos, cairo.Matrix(xx=sound_font_size, yy=sound_font_size))

chain_ids = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
padding = 8
sound_shift = 4

def draw_matrix(surface, matrix, color_palette, suffix_spellings, sounds, training_items, cell_size):
	square_size = (cell_size - (2 * padding)) / 3
	context = cairo.Context(surface)
	for (i, j), cell in np.ndenumerate(matrix):
		if sounds and i == 0:
			sound = '-' + sounds[j]
			context.set_source_rgb(0, 0, 0)
			context.set_font_face(suffix_doulos)
			context.set_font_size(sound_font_size)
			text_width = suffix_doulos_sf.text_extents(sound)[4]
			text_height = suffix_doulos_sf.text_extents(sound)[1]
			context.move_to(j*square_size+padding + square_size/2 - text_width/2, i*square_size+padding)
			context.show_text(sound)
		color = color_palette[cell]
		suffix = '-' + suffix_spellings[cell]
		context.set_source_rgb(*color)
		context.rectangle(j*square_size + padding, i*square_size + padding + sound_shift, square_size, square_size)
		context.fill_preserve()
		context.set_source_rgb(1, 1, 1)
		context.set_line_width(2)
		context.stroke()
		if f'{i}_{j}' in training_items:
			context.set_source_rgb(1, 1, 1)
		else:
			context.set_source_rgb(1, 1, 1)
		context.set_font_face(suffix_helvetica)
		context.set_font_size(suffix_font_size)
		text_width = suffix_helvetica_sf.text_extents(suffix)[4]
		text_height = suffix_helvetica_sf.text_extents(suffix)[1]
		context.move_to(j*square_size+padding + square_size/2 - text_width/2, i*square_size+padding + square_size/2 - text_height/2 + sound_shift)
		context.show_text(suffix)


def draw_chain(surface, chain, cell_size):
	context = cairo.Context(surface)
	x = 0
	for generation in chain:
		subsurface = surface.create_for_rectangle(x, 0, cell_size, cell_size)
		if isinstance(generation, str):
			context.set_source_rgb(0, 0, 0)
			context.set_font_face(helvetica)
			context.set_font_size(font_size)
			text_width = helvetica_sf.text_extents(generation)[4]
			text_height = helvetica_sf.text_extents(generation)[1]
			context.move_to(x + cell_size / 2 - text_width / 2, cell_size / 2 - text_height / 2 + sound_shift)
			context.show_text(generation)
		elif isinstance(generation, tuple):
			matrix, color_palette, suffix_spellings, sounds, training_items = generation
			draw_matrix(subsurface, matrix, color_palette, suffix_spellings, sounds, training_items, cell_size)
		x += cell_size


def draw_panel(output_path, chains, figure_width=1000, show_generation_numbers=True, show_chain_ids=True):
	if show_chain_ids:
		for chain, chain_id in zip(chains, chain_ids):
			chain.insert(0, chain_id)
	if show_generation_numbers:
		if show_chain_ids:
			chains = [[''] + [str(gen_i) for gen_i in range(len(chains[0]) - 1)]] + chains
		else:
			chains = [[str(gen_i) for gen_i in range(len(chains[0]))]] + chains
	n_rows = len(chains)
	n_cols = len(chains[0])
	figure_height = figure_width / n_cols * n_rows
	cell_size = figure_width / n_cols
	surface = cairo.PDFSurface(output_path, figure_width, figure_height)
	y = 0
	for chain in chains:
		subsurface = surface.create_for_rectangle(0, y, figure_width, cell_size)
		draw_chain(subsurface, chain, cell_size)
		y += cell_size
	surface.finish()
