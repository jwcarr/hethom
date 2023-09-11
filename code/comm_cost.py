from collections import defaultdict
from itertools import product
import numpy as np


def convert_lexicon_meanings_to_tuple(lexicon):
	converted_lexicon = {}
	for item, signal in lexicon.items():
		meaning = tuple(map(int, item.split('_')))
		converted_lexicon[meaning] = signal
	return converted_lexicon


def communicative_cost(lexicon, dims):
	if isinstance(list(lexicon.keys())[0], str):
		lexicon = convert_lexicon_meanings_to_tuple(lexicon)
	reverse_lexicon = defaultdict(set)
	for meaning, signal in lexicon.items():
		reverse_lexicon[signal].add(meaning)
	U = product(*[range(n_values) for n_values in dims])
	U_size = np.product(dims)
	return 1 / U_size * sum([-np.log2(1 / len(reverse_lexicon[lexicon[m]])) for m in U])


if __name__ == '__main__':

	degenerate = {
		(0, 0): 'buvico',
		(0, 1): 'buvico',
		(0, 2): 'buvico',
		(1, 0): 'zetico',
		(1, 1): 'zetico',
		(1, 2): 'zetico',
		(2, 0): 'wopico',
		(2, 1): 'wopico',
		(2, 2): 'wopico',
	}
	holistic = {
		(0, 0): 'buvico',
		(0, 1): 'buviko',
		(0, 2): 'buviqo',
		(1, 0): 'zeticoe',
		(1, 1): 'zetikoe',
		(1, 2): 'zetiqoe',
		(2, 0): 'wopicoh',
		(2, 1): 'wopikoh',
		(2, 2): 'wopiqoh',
	}
	redundant = {
		(0, 0): 'buvico',
		(0, 1): 'buvico',
		(0, 2): 'buvico',
		(1, 0): 'zetiko',
		(1, 1): 'zetiko',
		(1, 2): 'zetiko',
		(2, 0): 'wopiqo',
		(2, 1): 'wopiqo',
		(2, 2): 'wopiqo',
	}
	expressive = {
		(0, 0): 'buvico',
		(0, 1): 'buviko',
		(0, 2): 'buviqo',
		(1, 0): 'zetico',
		(1, 1): 'zetiko',
		(1, 2): 'zetiqo',
		(2, 0): 'wopico',
		(2, 1): 'wopiko',
		(2, 2): 'wopiqo',
	}
	semi_expressive = {
		(0, 0): 'buvico',
		(0, 1): 'buvico',
		(0, 2): 'buviqo',
		(1, 0): 'zetico',
		(1, 1): 'zetico',
		(1, 2): 'zetiqo',
		(2, 0): 'wopico',
		(2, 1): 'wopico',
		(2, 2): 'wopiqo',
	}

	print(communicative_cost(degenerate, (3, 3)))
	print(communicative_cost(holistic, (3, 3)))
	print(communicative_cost(redundant, (3, 3)))
	print(communicative_cost(expressive, (3, 3)))
	print(communicative_cost(semi_expressive, (3, 3)))


	
	


