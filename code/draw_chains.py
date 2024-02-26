import numpy as np
import cairocffi as cairo

font_size = 20
helvetica = cairo.ToyFontFace('Helvetica', cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
helvetica_sf = cairo.ScaledFont(helvetica, cairo.Matrix(xx=font_size, yy=font_size))

seen_suffix_font_size = 5
seen_suffix_helvetica = cairo.ToyFontFace('Helvetica', cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
seen_suffix_helvetica_sf = cairo.ScaledFont(seen_suffix_helvetica, cairo.Matrix(xx=seen_suffix_font_size, yy=seen_suffix_font_size))

unseen_suffix_font_size = 5
unseen_suffix_helvetica = cairo.ToyFontFace('Helvetica', cairo.FONT_SLANT_OBLIQUE, cairo.FONT_WEIGHT_BOLD)
unseen_suffix_helvetica_sf = cairo.ScaledFont(unseen_suffix_helvetica, cairo.Matrix(xx=unseen_suffix_font_size, yy=unseen_suffix_font_size))

sound_font_size = 5
sound_doulos = cairo.ToyFontFace('Doulos SIL', cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
sound_doulos_sf = cairo.ScaledFont(sound_doulos, cairo.Matrix(xx=sound_font_size, yy=sound_font_size))

padding = 3


def draw_char(surface, char, cell_size):
	context = cairo.Context(surface)
	context.set_source_rgb(0, 0, 0)
	context.set_font_face(helvetica)
	context.set_font_size(font_size)
	text_width = helvetica_sf.text_extents(char)[4]
	text_height = helvetica_sf.text_extents('X')[1]
	context.move_to(cell_size / 2 - text_width / 2, cell_size / 2 - text_height / 2)
	context.show_text(char)


def draw_matrix(surface, generation, cell_size, gen0=False):
	matrix, color_palette, suffix_spellings, sounds, training_items = generation
	square_size = (cell_size - (2 * padding)) / 3
	context = cairo.Context(surface)
	for (i, j), cell in np.ndenumerate(matrix):
		color = color_palette[cell]
		suffix = suffix_spellings[cell]
		context.set_source_rgb(*color)
		context.rectangle(j*square_size + padding, i*square_size + padding, square_size, square_size)
		context.fill_preserve()
		context.set_source_rgb(1, 1, 1)
		context.set_line_width(1)
		context.stroke()
		context.set_source_rgb(1, 1, 1)
		if gen0 or f'{i}_{j}' in training_items:
			context.set_font_face(seen_suffix_helvetica)
			context.set_font_size(seen_suffix_font_size)
			text_width = seen_suffix_helvetica_sf.text_extents(suffix)[4]
			text_height = seen_suffix_helvetica_sf.text_extents('X')[1]
		else:
			context.set_font_face(unseen_suffix_helvetica)
			context.set_font_size(unseen_suffix_font_size)
			text_width = unseen_suffix_helvetica_sf.text_extents(suffix)[4]
			text_height = unseen_suffix_helvetica_sf.text_extents('X')[1]
		context.move_to(j*square_size+padding + square_size/2 - text_width/2, i*square_size+padding + square_size/2 - text_height/2)
		context.show_text(suffix)


def draw_chain(surface, chain, cell_size):
	x = 0
	for gen_i, generation in enumerate(chain):
		subsurface = surface.create_for_rectangle(x, 0, cell_size, cell_size)
		if isinstance(generation, str):
			draw_char(subsurface, generation, cell_size)
		elif isinstance(generation, tuple):
			draw_matrix(subsurface, generation, cell_size, gen_i == 1)
		x += cell_size


def draw_chain_sounds(surface, chain, cell_size):
	square_size = (cell_size - (2 * padding)) / 3
	x = 0
	for generation in chain:
		if isinstance(generation, tuple):
			subsurface = surface.create_for_rectangle(x, 0, cell_size, padding * 2)
			context = cairo.Context(subsurface)
			for i, sound in enumerate(generation[3]):
				context.set_source_rgb(0, 0, 0)
				context.set_font_face(sound_doulos)
				context.set_font_size(sound_font_size)
				text_width = sound_doulos_sf.text_extents(sound)[4]
				text_height = sound_doulos_sf.text_extents('X')[3]
				context.move_to(i * square_size + square_size / 2 + padding - text_width / 2, padding + text_height / 2)
				context.show_text(sound)
		x += cell_size


def draw_panel(output_path, chains, figure_width=1000, chain_ids=None, show_generation_numbers=True, show_sounds=True):
	if chain_ids:
		for chain, chain_id in zip(chains, chain_ids):
			chain.insert(0, chain_id)
	if show_generation_numbers:
		if chain_ids:
			chains = [[''] + [str(gen_i) for gen_i in range(len(chains[0]) - 1)]] + chains
		else:
			chains = [[str(gen_i) for gen_i in range(len(chains[0]))]] + chains
	n_rows = len(chains)
	n_cols = len(chains[0])
	figure_height = figure_width / n_cols * n_rows
	cell_size = figure_width / n_cols
	filename = output_path.name
	if filename.endswith('.eps'):
		surface = cairo.PSSurface(str(output_path), figure_width, figure_height)
		surface.set_eps(True)
	elif filename.endswith('.pdf'):
		surface = cairo.PDFSurface(str(output_path), figure_width, figure_height)
	elif filename.endswith('.svg'):
		surface = cairo.SVGSurface(str(output_path), figure_width, figure_height)
	else:
		raise ValueError('Unrecognized format')
	context = cairo.Context(surface)
	context.set_source_rgb(1, 1, 1)
	context.paint()
	y = 0
	for chain in chains:
		subsurface = surface.create_for_rectangle(0, y, figure_width, cell_size)
		draw_chain(subsurface, chain, cell_size)
		if show_sounds:
			subsurface = surface.create_for_rectangle(0, y - padding, figure_width, (padding * 2))
			draw_chain_sounds(subsurface, chain, cell_size)
		y += cell_size
	surface.finish()
