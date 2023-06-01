import json

def json_load(filepath):
	with open(filepath) as file:
		data = json.load(file)
	return data

def json_save(obj, output_file):
	with open(output_file, 'w') as file:
		json.dump(obj, file, indent='\t', ensure_ascii=False)
