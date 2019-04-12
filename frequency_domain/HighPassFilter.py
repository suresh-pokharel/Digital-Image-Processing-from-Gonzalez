from tkinter import *
import cv2
import matplotlib as mpl

mpl.use('TkAgg')
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import filedialog


class high_pass:
    path = '../Images/leena.jpeg'

    def __init__(self, master):
        self.frame1 = Frame(master)
        self.frame2 = Frame(master)
        self.frame3 = Frame(master)
        hbtn = Button(self.frame2, text="OPEN IMAGE", command=lambda: self.button_click(master))
        hbtn.pack(fill="none", expand=True)
        self.initUI(master)

    def initUI(self, master):
        self.frame1.grid(row=0, column=0)
        self.frame2.grid(row=10, column=0)
        self.frame3.grid(row=0, column =9)

        if len(high_pass.path) > 0:
            img = cv2.imread(high_pass.path, 0)
            min = np.minimum(img.shape[0], img.shape[1])
            img = cv2.resize(img, (min, min))
            img[img >= 225] = 0
            M, N = img.shape
            # computing the 2-d fourier transformation of the image
            fourier_image = np.fft.fft2(img)

            # ideal low pass filter
            u = np.array(range(0, M))
            v = np.array(range(0, N))
            idx = np.where(u > (M / 2))
            u[idx] = u[idx] - M
            idy = np.where(v > N / 2)
            v[idy] = v[idy] - N
            [V, U] = np.meshgrid(v, u)
            D = (U ** 2 + V ** 2) ** (1 / 2)
            # cutoff = 40
            cutoff = [50, 40, 20, 10]

            H = (D > 40)
            G = H * fourier_image
            imback = np.fft.ifft2(G)
            imback = np.uint8(np.real(imback))
            imback[imback >= 225] = 0

            H1 = (D > 20)
            G1 = H1 * fourier_image
            imback1 = np.fft.ifft2(G1)
            imback1 = np.uint8(np.real(imback1))
            imback1[imback1 >= 225] = 0

            fshift = np.fft.fftshift(fourier_image)
            magnitude_spectrum = 20 * np.log(np.abs(fshift))

            fig = Figure(figsize=(8, 8))
            fig.suptitle("Ideal highpass filters")
            a = fig.add_subplot(221)
            a.set_title("Original Image")
            a.imshow(img, cmap='gray')
            b = fig.add_subplot(223)
            b.set_title("Cutoff = 40")
            b.imshow(imback, cmap='gray')
            c = fig.add_subplot(224)
            c.set_title("Cutoff = 20")
            c.imshow(imback1, cmap='gray')
            d = fig.add_subplot(222)
            d.set_title("Fourier Transformation")
            d.imshow(magnitude_spectrum, cmap='gray')

            canvas = FigureCanvasTkAgg(fig, self.frame1)
            canvas.get_tk_widget().grid(row=1, column=0, columnspan=4, rowspan=8)
            canvas.draw()

            order = [1,2]
            H3 = 1 / (1 +(cutoff[1]/D)**(2*order[0]))
            G3 = H3 * fourier_image
            imback2 = np.fft.ifft2(G3)
            imback2 = np.uint8(np.real(imback2))
            imback2[imback2 >= 225] = 0

            H4 = 1 / (1 + (cutoff[1]/D) ** (2*order[1]))
            G4 = H4 * fourier_image
            imback3 = np.fft.ifft2(G4)
            imback3 = np.uint8(np.real(imback3))
            imback3[imback3 >= 225] = 0

            H5 = 1 / (1 + (cutoff[3]/D) ** (2*order[0]))
            G5 = H5 * fourier_image
            imback4 = np.fft.ifft2(G5)
            imback4 = np.uint8(np.real(imback4))
            imback4[imback4 >= 225] = 0

            H6 = 1 / (1 + (cutoff[3]/D) ** (2*order[1]))
            G6 = H6 * fourier_image
            imback5 = np.fft.ifft2(G6)
            imback5 = np.uint8(np.real(imback5))
            imback5[imback5 >= 225] = 0

            fig1 = Figure(figsize=(8, 8))
            fig1.suptitle("Butterworth highpass Filter")
            e = fig1.add_subplot(221)
            e.set_title("Order = 1, Cutoff = 40")
            e.imshow(imback2, cmap='gray')
            f = fig1.add_subplot(222)
            f.set_title("Order = 2, Cutoff = 40")
            f.imshow(imback3, cmap='gray')
            g = fig1.add_subplot(223)
            g.set_title("Order = 1, Cutoff = 10")
            g.imshow(imback4, cmap='gray')
            h = fig1.add_subplot(224)
            h.set_title("Order = 2, Cutoff = 10")
            h.imshow(imback5, cmap='gray')

            canvas = FigureCanvasTkAgg(fig1, self.frame3)
            canvas.get_tk_widget().grid(row=1, column=0, columnspan=4, rowspan=8)
            canvas.draw()

    def button_click(self, master):
        high_pass.path = filedialog.askopenfilename(filetypes=[("Image File", '.png')])
        self.initUI(master)


def main():
    root = Tk()
    root.title("High Pass Filters - Frequency Domain")
    high_pass(root)
    root.mainloop()


if __name__ == '__main__':
    main()