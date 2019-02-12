
import numpy as np
from PIL import Image
import math


class ImageClass:
    def __init__(self):
        self.width = 0
        self.height = 0
        self.pixels = 0
        self.RGB = []
        self.gs_value = []
        self.img = 0

    def load_image(self, path):
        # returns array of RGB values from Image
        self.img = Image.open(path)
        self.pixels = self.img.load()
        print('Original -- ',self.pixels)
        self.width, self.height = self.img.size
        for i in range(self.width):
            rgb_temp = []
            for j in range(self.height):
                r = self.pixels[i, j][0]  # convert R value into range 0-255
                g = self.pixels[i, j][1]
                b = self.pixels[i, j][2]
                rgb_temp.append([r,g,b])
            self.RGB.append(rgb_temp)
        return self.RGB

    def rgb_to_gray_scale(self,pixels):

        # convert RGB image to gray scale
        # pixels in array of RGB form eg. [[225, 137, 127], [225, 137, 127], [227, 137, 122]...]
        for rows in pixels:
            gs_temp = [] # to store gs value for a row
            for pixel in rows:
                # c_linear = 0.2126 * R + 0.7152 * G + 0.0722 * B
                # From https: // stackoverflow.com / questions / 17615963 / standard - rgb - to - grayscale - conversion

                c_linear = 0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2]

                if c_linear <= 0.0031308:
                        c_srgb = 12.92 * c_linear
                else:
                        c_srgb = 1.055 * c_linear**(1/2.4) - 0.055

                # c_srgb returns in 0-1 range, so convert to 255 scale
                px_value = math.ceil(c_linear)
                print(c_srgb)
                gs_temp.append(px_value)
            self.gs_value.append(gs_temp)
        return self.gs_value

    def display_image(self,pixels):
        result_path = 'Images/leena.png'
        self.img = Image.open(result_path)
        self.pixels = self.img.load()
        for i in range(self.width):
            for j in range(self.height):
                self.pixels[i,j] = pixels[i][j]
        self.img.save('result.jpg')

# img = Image.open('Images/leena.png')
# pixels = img.load()
# width, height = img.size
#
# print(img.convert('L'))
# for i in range(width):
#     for j in range(height):
#         print (pixels[i,j])
#         # https: // stackoverflow.com / questions / 17615963 / standard - rgb - to - grayscale - conversion
#         R = pixels[i,j][0]/255 # convert R value into range 0-1
#         G = pixels[i,j][1]/255 # convert G value into range 0-1
#         B = pixels[i,j][2]/255 # convert B value into range 0-1
#
#         c_linear = 0.2126 * R + 0.7152 * G + 0.0722 * B
#
#         if c_linear <= 0.0031308:
#             c_srgb = 12.92 * c_linear
#         else:
#             c_srgb = 1.055 * c_linear**(1/2.4) - 0.055
#
#         print (c_srgb)
#         px_value = math.ceil(c_linear*255)
#         if(px_value > 128):
#             px_value = 255
#         else:
#             px_value = 0
#         print(px_value)
#         pixels[i, j] = (px_value, px_value, px_value)
#
# img.show('Images/result1.png')


img = ImageClass()
RGB = img.load_image('Images/leena.png')

# print(RGB)
gs = img.rgb_to_gray_scale(RGB)
# print(gs)
img.display_image(gs)
