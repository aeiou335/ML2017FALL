import sys
from PIL import Image
im = Image.open(sys.argv[1])


width, height = im.size

for x in range(width):
	for y in range(height):
		rgb = im.getpixel((x,y))
		new_rgb = (rgb[0]//2, rgb[1]//2, rgb[2]//2)
		im.putpixel((x,y), new_rgb)


im.save("Q2.png")