'''
This script creates all the spoken audio stimuli for the experiments. It uses
the Mac OS text-to-speech synthesizer, so must be run on a Mac.
'''

import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()


def say(utterance, voice, output_path):
	subprocess.check_output(['say', f'"{utterance}"', '-v', voice, '-o', output_path])


# Audio check sentences

for i, sentence in enumerate(['the quick cat', 'the green bird', 'the hungry dog', 'the angry lion']):
	path = ROOT/'experiment'/'client'/'sounds'/f'test{i}.m4a'
	say(sentence, 'Tessa', path)

say('This is an attention check; please click on the object.', 'Tessa', ROOT/'experiment'/'client'/'sounds'/'catch_instruction.m4a')
say('Great. Please continue.', 'Tessa', ROOT/'experiment'/'client'/'sounds'/'catch_acknowledge.m4a')

# Experiment 1, differentiation

for i, word in enumerate(['boovikoe', 'zettikoe', 'woppikoe']):
	say(word, 'Tessa', ROOT/'experiment'/'client'/'words'/f'{i}.m4a')

# Experiment 2, conservation

for i, stem in enumerate(['boovi', 'zetti', 'woppi']):
	for c, cons in enumerate(['f', 's', 'sh']):
		for v, vowl in enumerate(['oe', 'a', 'ay']):
			word = f'{stem}{cons}{vowl}'
			say(word, 'Moira', ROOT/'experiment'/'client'/'words'/f'{i}_{c}_{v}.m4a')
