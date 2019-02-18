
import numpy as np
from PIL import Image
import math
import matplotlib.pyplot as plt


class ImageClass:
    def __init__(self, image_path):
        self.img = Image.open(image_path)
        self.pixels = self.img.load()
        self.width, self.height = self.img.size
        self.RGB = []
        self.gs_value = []

    def get_rgb(self):
        # returns array of RGB values from Image
        for i in range(self.width):
            rgb_temp = []
            for j in range(self.height):
                r = self.pixels[i, j][0]  # convert R value into range 0-255
                g = self.pixels[i, j][1]
                b = self.pixels[i, j][2]
                rgb_temp.append([r,g,b])
            self.RGB.append(rgb_temp)
        return self.RGB

    def rgb_to_gray_scale(self):
        pixels = self.get_rgb()
        # convert RGB image to gray scale
        # pixels in array of RGB form eg. [[225, 137, 127], [225, 137, 127], [227, 137, 122]...]
        for rows in pixels:
            gs_temp = []  # to store gs value for a row
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
                gs_temp.append(px_value)
            self.gs_value.append(gs_temp)
        return self.gs_value

    def negative_image(self):
        for i in range(self.width):
            for j in range(self.height):
                r = 255 - self.pixels[i,j][0]  # inversion R
                g = 255 - self.pixels[i,j][1]  # inversion G
                b = 255 - self.pixels[i,j][2]  # inversion B
                self.pixels[i,j] = (r,g,b)
        self.img.show()

    def log_transformation(self, c):  # s = clog(1+r)
        for i in range(self.width):
            for j in range(self.height):
                r = c * math.ceil(math.log10(1 + self.pixels[i, j][0]))
                g = c * math.ceil(math.log10(1 + self.pixels[i, j][1]))
                b = c * math.ceil(math.log10(1 + self.pixels[i, j][2]))
                self.pixels[i, j] = (r, g, b)
        self.img.show()

    def power_transformation(self, c, lambda_value):  # s = c*r^lambda
        for i in range(self.width):
            for j in range(self.height):
                r = c * math.ceil(math.pow(self.pixels[i, j][0], lambda_value))
                g = c * math.ceil(math.pow(self.pixels[i, j][1], lambda_value))
                b = c * math.ceil(math.pow(self.pixels[i, j][2], lambda_value))
                self.pixels[i, j] = (r, g, b)
        self.img.show()

    def display_image(self,pixels):
        result_path = 'Images/leena.png'
        self.img = Image.open(result_path)
        self.pixels = self.img.load()
        for i in range(self.width):
            for j in range(self.height):
                self.pixels[i,j] = pixels[i][j]
        self.img.save('Images/result.jpg')

    def bit_plane_slicing(self):
        gs_values = self.rgb_to_gray_scale()
        binary_values = []  # to hold binary value of each pixel
        for row in gs_values:
            bin_row_temp = []
            for pixel in row:
                bin_temp = "{0:08b}".format(pixel)  # convert into 8 bit binary value
                bin_row_temp.append(bin_temp)
            binary_values.append(bin_row_temp)  # push each row

        #  here we have binary 8 bit value for each of pixels  gray scale
        #  we need to slice each pixel's bit

        slices = [[], [], [], [], [], [], [], []]  # to store slices 0,1,2...7
        for row in binary_values:
            temp_slice_rows = [[], [], [], [], [], [], [], []]
            for pixel in row:
                    for i in range(8):  # go through each bit 0-7
                        temp_slice_rows[i].append(int(pixel[i]))  # get ith value of string, converting to int

            for j in range(8):
                slices[j].append(temp_slice_rows[j])

        # display LSB slice TEST
        for i in range(8): # for eight slices
            for j in range(self.width):
                for k in range(self.height):
                    bit = slices[i][j][k]
                    bit = bit * 255
                    self.pixels[j,k] = bit
            self.img.show()
        return slices

    def histogram_plot(self):
        gs = self.rgb_to_gray_scale()
        print(gs)
        plt.hist(gs, bins=8)
        x_labels = range(0, (17+1)*15, 15)[1:] # [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160,170,180,190]
        plt.xticks(x_labels)
        plt.show()
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


img = ImageClass('Images/Leena.png')
# img.negative_image()
# rgb = img.get_rgb()
# gs = img.rgb_to_gray_scale()
# img.histogram_plot()
# img.display_image(gs)
img.log_transformation(7)
# img.power_transformation(99, 2.5)
# img.bit_plane_slicing()
# print(slices)
