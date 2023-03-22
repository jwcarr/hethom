transparent = {
	"0_0": "buviko",
	"0_1": "buviko",
	"0_2": "buviko",
	"0_3": "buviko",
	"1_0": "zetiko",
	"1_1": "zetiko",
	"1_2": "zetiko",
	"1_3": "zetiko",
	"2_0": "gafiko",
	"2_1": "gafiko",
	"2_2": "gafiko",
	"2_3": "gafiko",
	"3_0": "wopiko",
	"3_1": "wopiko",
	"3_2": "wopiko",
	"3_3": "wopiko"
}
redundant = {
	"0_0": "buvikoh",
	"0_1": "buvikoh",
	"0_2": "buvikoh",
	"0_3": "buvikoh",
	"1_0": "zeteeco",
	"1_1": "zeteeco",
	"1_2": "zeteeco",
	"1_3": "zeteeco",
	"2_0": "gafykoe",
	"2_1": "gafykoe",
	"2_2": "gafykoe",
	"2_3": "gafykoe",
	"3_0": "wopycow",
	"3_1": "wopycow",
	"3_2": "wopycow",
	"3_3": "wopycow"
}
expressive = {
	"0_0": "buvikop",
	"0_1": "buvikog",
	"0_2": "buvikob",
	"0_3": "buvikoy",
	"1_0": "zetikop",
	"1_1": "zetikog",
	"1_2": "zetikob",
	"1_3": "zetikoy",
	"2_0": "gafikop",
	"2_1": "gafikog",
	"2_2": "gafikob",
	"2_3": "gafikoy",
	"3_0": "wopikop",
	"3_1": "wopikog",
	"3_2": "wopikob",
	"3_3": "wopikoy",
}
expressive = {
	"0_0": "buvikoh",
	"0_1": "buveeco",
	"0_2": "buvykoe",
	"0_3": "buvycow",
	"1_0": "zetikoh",
	"1_1": "zeteeco",
	"1_2": "zetykoe",
	"1_3": "zetycow",
	"2_0": "gafikoh",
	"2_1": "gafeeco",
	"2_2": "gafykoe",
	"2_3": "gafycow",
	"3_0": "wopikoh",
	"3_1": "wopeeco",
	"3_2": "wopykoe",
	"3_3": "wopycow"
}



def scan(lex):
	words = []
	meanings = []
	for meaning, word in lex.items():
		shape, color = meaning.split('_')
		meanings.append((int(shape), int(color)))
		words.append(word)
	max_length = max([len(word) for word in words])
	stack = []
	for i in range(max_length):
		letters = [word[i] for word in words]
		unique_letters = set(letters)
		substack = []
		for unique_letter in sorted(list(unique_letters)):
			which_items = [meaning for letter, meaning in zip(letters, meanings) if letter == unique_letter]
			shapes = [shape for shape, color in which_items]
			colors = [color for shape, color in which_items]
			substack.append(
				(unique_letter, (set(shapes), set(colors)))
			)
		stack.append(substack)
			# if len(set(shapes)) <= len(set(colors)):
			# 	maj_shape = mode(shapes)
			# 	print(unique_letter, maj_shape)
			# else:
			# 	maj_color = mode(colors)
	return stack

def mode(values):
	values = list(values)
	return max(set(values), key=values.count)

def merge(stack):
	if len(stack) == 1:
		return stack
	sym1 = stack[0]
	sym2 = stack[1]
	merger = False
	for i, (letter, meaning) in enumerate(sym1):
		for j, (letter2, meaning2) in enumerate(sym2):
			if meaning == meaning2:
				stack[0][i] = (letter + letter2, meaning)
				del sym2[j]
				if len(sym2) == 0:
					del stack[1]
				merger = True
				break
	if merger:
		return merge(stack)
	else:
		return stack[0:1] + merge(stack[1:])



stack = scan(redundant)
stack = merge(stack)
print('FINAL STACK')
for row in stack:
	print(row)