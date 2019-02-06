
import numpy as np
from PIL import Image
import math


img = Image.open('Images/leena.png')
pixels = img.load()
width, height = img.size

print(img.convert('L'))
for i in range(width):
    for j in range(height):
        print (pixels[i,j])
        # https: // stackoverflow.com / questions / 17615963 / standard - rgb - to - grayscale - conversion
        R = pixels[i,j][0]/255 # convert R value into range 0-1
        G = pixels[i,j][1]/255 # convert G value into range 0-1
        B = pixels[i,j][2]/255 # convert B value into range 0-1

        c_linear = 0.2126 * R + 0.7152 * G + 0.0722 * B

        if c_linear <= 0.0031308:
            c_srgb = 12.92 * c_linear
        else:
            c_srgb = 1.055 * c_linear**(1/2.4) - 0.055

        print (c_srgb)
        px_value = math.ceil(c_linear*255)
        pixels[i, j] = (px_value, px_value, px_value)

img.show('Images/result1.png')