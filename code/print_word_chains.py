from pathlib import Path
from itertools import product
from utils import json_load


ROOT = Path(__file__).parent.parent.resolve()


cons = ['f', 's', 'ʃ']
vwls = ['əʊ', 'ə', 'ɛɪ']
def get_sound(item, data):
	sound_file = data['spoken_forms'][item]
	if '_' not in sound_file:
		return 'kəʊ'
	s, c, v = sound_file.split('.')[0].split('_')
	return f'{cons[int(c)]}{vwls[int(v)]}'

def print_word_chains(dataset):
	for condition, data in dataset.items():
		print(condition.upper())
		for chain_i, chain in enumerate(data):
			print('  Chain', chain_i)
			table = [[] for _ in range(9)]
			for item_i, (shape, color) in enumerate(product(range(3), range(3))):
				item = f'{shape}_{color}'
				word = chain[0][0]['lexicon'][item]
				table[item_i].append(word.ljust(9, ' '))
				for subject_a, subject_b in chain[1:]:
					bottleneck = '➤ ' if item in subject_a['training_items'] else '  '
					word = subject_a['lexicon'][item]
					sound = get_sound(item, subject_a).ljust(4) if 'spoken_forms' in subject_a else ''
					table[item_i].append(bottleneck + sound + word.ljust(9, ' '))
			print(''.join([str(gen_i).ljust(16, ' ') for gen_i in range(len(table[0]))]).strip())
			for row in table:
				print(' '.join(row).strip())


if __name__ == '__main__':

	dataset = json_load(ROOT / 'data' / 'exp.json')
	print_word_chains(dataset)
