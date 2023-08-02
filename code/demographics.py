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

		amount = 300
		if with_bonus:
			amount += data['total_bonus']
		amounts.append(amount)
		
		rate = 3600 / time * amount
		rates.append(rate)

		comments_time.append(data['modified_time'] - data['responses'][-1]['time'])

	print('Median completion time (mins):', np.median(times) / 60)
	print('Median earnings (GBP):', np.median(amounts) / 100)
	print('Median hourly rate (GBP):', np.median(rates) / 100)
	print('Median comments time (mins):', np.median(comments_time) / 60)


def most_common_languages(exp_id):
	df = pd.read_csv(ROOT/'private'/f'{exp_id}_demographics.csv')
	langs = list(df['Language'])
	unique_langs = list(set(langs))
	counts = {lang: langs.count(lang) for lang in unique_langs}
	unique_langs.sort(key=lambda k: -counts[k])
	for i, lang in enumerate(unique_langs, 1):
		print(i, lang, counts[lang], round(counts[lang] / len(langs) * 100, 2))


calculate_rates('exp2', with_bonus=False)
# most_common_languages('exp2')