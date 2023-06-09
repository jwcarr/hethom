import random

task = {
	"stems": [
		"buv",
		"zet",
		"gaf",
		"wop"
	],
	"suffixes": [
		"itha",
		"ise",
		"ishu",
		"ixo"
	],
}

def create_inital_sounds():
	cons = list(range(4))
	vowl = list(range(4))
	random.shuffle(cons)
	random.shuffle(vowl)
	sounds = {}
	for i, stem in enumerate(task['stems']):
		for j, suffix in enumerate(task['suffixes']):
			item = f'{i}_{j}'
			sounds[item] = f'{i}_{cons[j]}_{vowl[j]}'
	return sounds


{1: (0, 1), 2: (2, 3), 3: (0, 2)}

def sound_change(sounds, change_i):
	match change_i:
		case 1:
			p, q = 0, 1
		case 2:
			p, q = 2, 3
		case 3:
			pass

	pnk_suffix = sounds[f'0_{p}'].split('_')[1:]
	gry_suffix = sounds[f'0_{q}'].split('_')[1:]
	
	if random.random() < 0.5:
		new_suffix = '_'.join([pnk_suffix[0], gry_suffix[1]])
	else:
		new_suffix = '_'.join([gry_suffix[0], pnk_suffix[1]])

	for i in range(4): # n_shapes
		for item in [f'{i}_{p}', f'{i}_{q}']:
			sounds[item] = f'{i}_{new_suffix}'

	print(pnk_suffix)
	print(gry_suffix)
	print(new_suffix)
	return sounds


sounds = create_inital_sounds()
for item, sound in sounds.items():
	print(item, sound)

sound_change(sounds, 1)
for item, sound in sounds.items():
	print(item, sound)

sound_change(sounds, 2)
for item, sound in sounds.items():
	print(item, sound)
