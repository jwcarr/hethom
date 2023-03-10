from pathlib import Path
import cairosvg

ROOT = Path(__file__).parent.parent.resolve()

header = '''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<svg width="100%" height="100%" viewBox="0 0 1000 1000" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" xml:space="preserve" xmlns:serif="http://www.serif.com/" style="fill-rule:evenodd;clip-rule:evenodd;stroke-linejoin:round;stroke-miterlimit:2;">
'''

footer = '\n</svg>'

shapes = [
	'<g transform="matrix(1.60121,0,0,1.40577,-296.42,-151.265)"><path d="M497.387,173.739L764.67,394.929L662.577,752.823L332.197,752.823L230.104,394.929L497.387,173.739Z" style="fill:%s;"/></g>',
	'<g transform="matrix(1.2861,0,0,1.36861,-160.848,-168.265)"><path d="M513.838,159.48L570.681,323.881L680.222,273.078L662.657,386.676L846.607,386.676L697.788,488.281L783.054,570.482L662.657,589.886L719.501,754.287L570.681,652.682L513.838,754.287L456.994,652.682L308.175,754.287L365.019,589.886L244.621,570.482L329.887,488.281L181.068,386.676L365.019,386.676L347.453,273.078L456.994,323.881L513.838,159.48Z" style="fill:%s;"/></g>',
	'<g transform="matrix(1.55578,0,0,1.79159,-300.984,-369.947)"><path d="M583.618,280.079L721.162,366.857L652.39,449.055L763.665,480.452L711.128,620.862L599.853,589.465L599.853,691.067L429.839,691.067L429.839,589.465L318.563,620.862L266.026,480.452L377.302,449.055L308.53,366.857L446.074,280.079L514.846,362.277L583.618,280.079Z" style="fill:%s;"/></g>',
	'<g transform="matrix(1.33829,0,0,1.38054,-190.554,-201.325)"><path d="M515.996,182.048C701.577,182.048 852.246,328.106 852.246,508.008C852.246,687.91 701.577,833.968 515.996,833.968C330.415,833.968 179.747,687.91 179.747,508.008C179.747,328.106 330.415,182.048 515.996,182.048ZM515.996,345.028C608.787,345.028 684.121,418.057 684.121,508.008C684.121,597.959 608.787,670.988 515.996,670.988C423.206,670.988 347.872,597.959 347.872,508.008C347.872,418.057 423.206,345.028 515.996,345.028Z" style="fill:%s;"/></g>',
]

colors = ['#E85A71', '#6B6B7F', '#4EA1D3', '#FCBE32']

stimuli_path = ROOT / 'experiment' / 'client' / 'images' / 'shapes'

if not stimuli_path.exists():
	stimuli_path.mkdir()

for i, shape in enumerate(shapes):
	for j, color in enumerate(colors):
		
		image = header + (shape % (color,)) + footer
		filename = f'{i}_{j}.png'
		filepath = stimuli_path / filename
		cairosvg.svg2png(bytestring=image, write_to=str(filepath))
