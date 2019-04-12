"""
      Author  : Suresh Pokharel
      Email   : suresh.wrc@gmail.com
      GitHub  : github.com/suresh021
      URL     : psuresh.com.np
"""


from PIL import Image
import math
import matplotlib.pyplot as plt
import cv2
import numpy as np


class Spatial:
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

    def averaging_mask(self,m_size):
         # averaging matrix m_size*m_size
        for i in range(self.width-m_size):
            for j in range(self.height-m_size):
                #  calculate new value for R,G,B
                sum_r =0
                sum_g=0
                sum_b = 0
                for p in range(m_size):
                    for q in range(m_size):
                        sum_r = sum_r + self.pixels[i + p, j + q][0]
                        sum_g = sum_g + self.pixels[i + p, j + q][1]
                        sum_b = sum_b + self.pixels[i + p, j + q][2]
                total_blocks = m_size*m_size
                self.pixels[i,j] = (math.ceil(sum_r/total_blocks), math.ceil(sum_g/total_blocks), math.ceil(sum_b/total_blocks))
        self.img.show()

    def average_mask_cv2(self,m,n):
        img = cv2.imread('Images/leena.jpeg')

        blur = cv2.blur(img, (m, n))

        plt.subplot(121), plt.imshow(img), plt.title('Original')
        plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(blur), plt.title('Blurred')
        plt.xticks([]), plt.yticks([])
        plt.show()

    def gaussian_cv2(self, path, m, n):
        img = cv2.imread(path)
        kernel = np.ones((m, n), np.float32) / (m * n)
        dst = cv2.filter2D(img, -1, kernel)

        plt.subplot(121), plt.imshow(img), plt.title('Original')
        plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(dst), plt.title('Averaging')
        plt.xticks([]), plt.yticks([])
        plt.show()

    def weighted_masking(self, mask, m_size):
         # mask  matrix m_size*m_size
        a = int((m_size-1)/2)
        b = int((m_size-1)/2)


        for i in range(1, self.width-1):
            for j in range(1, self.height-1):
                #  calculate new value for R,G,B
                sum = 0
                for s in range(-1, 1, 1):
                    for t in range(-1, 1, 1):
                        pix = self.pixels[i+s, j+t]
                        gs = self.rgb_to_gs_single(pix)
                        sum = sum + gs * mask[s+1][t+1]
                final_value = int(sum)
                self.pixels[i,j] = (final_value, final_value, final_value)
        self.img.show()


    def rgb_to_gs_single(self, pix):
        return int((0.2126 * pix[0] + 0.7152 * pix[1] + 0.0722 * pix[2]))

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
                    bit = bit * 127
                    self.pixels[j, k] = (bit, bit, bit)
            self.img.show()
        return slices

    def histogram_plot(self):
        gs = self.rgb_to_gray_scale()
        print(gs)
        plt.hist(gs, bins=8)
        x_labels = range(0, (17+1)*15, 15)[1:]  # [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160,170,180,190]
        plt.xticks(x_labels)
        plt.show()



img = Spatial('Images/leena.jpeg')
# img.negative_image()
# rgb = img.get_rgb()
# gs = img.rgb_to_gray_scale()
# img.histogram_plot()
# img.display_image(gs)
# img.log_transformation(7)
# img.power_transformation(99, 2.5)
# mask = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
# img.averaging_mask(15)
# img.average_mask_cv2(9, 9)
img.gaussian_cv2('Images/leena.jpeg', 5, 5)
# img.weighted_masking(mask, 3)
# img.bit_plane_slicing()
