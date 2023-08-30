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
