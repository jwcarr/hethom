from pathlib import Path
import numpy as np
import pandas as pd
from utils import json_load, json_save


ROOT = Path(__file__).parent.parent.resolve()


def calculate_rates(exp_id, with_bonus=False):
	times = []
	amounts = []
	rates = []
	comments_time = []

	for i in range(1, 539):
		subject_id = str(i).zfill(3)
		path = ROOT/'data'/exp_id/f'subject_{subject_id}.json'
		try:
			data = json_load(path)

		except:
			break

		time = data['modified_time'] - data['creation_time']
		times.append(time)

		amount = 200
		if with_bonus:
			amount += data['total_bonus']
		amounts.append(amount)
		
		rate = 3600 / time * amount
		rates.append(rate)

	print('Median completion time (mins):', np.median(times) / 60)
	print('Median earnings (GBP):', np.median(amounts) / 100)
	print('Median hourly rate (GBP):', np.median(rates) / 100)


def most_common_languages(exp_id):
	df = pd.read_csv(ROOT/'private'/'prolific_demographic_data'/f'{exp_id}_demos.csv')
	langs = list(df['Country of residence'])
	unique_langs = list(set(langs))
	counts = {lang: langs.count(lang) for lang in unique_langs}
	unique_langs.sort(key=lambda k: -counts[k])
	for i, lang in enumerate(unique_langs, 1):
		print(i, lang, counts[lang], round(counts[lang] / len(langs) * 100, 2))


calculate_rates('exp3', with_bonus=True)
most_common_languages('exp3')