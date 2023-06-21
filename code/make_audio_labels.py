'''
This script creates all the spoken audio stimuli for the experiments. It uses
the Mac OS text-to-speech synthesizer, so must be run on a Mac.
'''

import subprocess


def say(utterance, output_path):
	subprocess.check_output(['say', f'"{utterance}"', '-v', 'Tessa', '-o', output_path])


# Audio check sentences

for i, sentence in enumerate(['the quick cat', 'the green bird', 'the hungry dog', 'the angry lion']):
	path = f'../experiment/client/sounds/test{i}.m4a'
	say(sentence, path)

say('This is an attention check; please click on the object.', '../experiment/client/sounds/catch_instruction.m4a')
say('Great. Please continue.', '../experiment/client/sounds/catch_acknowledge.m4a')


# Experiment 1 spoken word stimuli

for i, word in enumerate(['booveekoe', 'zeteekoe', 'gafeekoe', 'woppeekoe']):
	say(word, f'../experiment/client/words/{i}.m4a')


# Experiment 2 spoken word stimuli

for i, stem in enumerate(['boovy', 'zetty', 'gaffy', 'woppy']):
	for c, cons in enumerate(['th', 's', 'sh', 'h']):
		for v, vowl in enumerate(['a', 'ay', 'oe', 'u']):
			word = f'{stem}{cons}{vowl}'
			say(word, f'../experiment/client/words/{i}_{c}_{v}.m4a')
