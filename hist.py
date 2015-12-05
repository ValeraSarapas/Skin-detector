#!/usr/bin/env python

import cv2
import matplotlib.pyplot as plt

def show_histogram(im):
    """ Function to display image histogram. 
        Supports single and three channel images. """

    if im.ndim == 2:
        # Input image is single channel
        plt.hist(im.flatten(), 256, range=(0, 250), fc='k')
        plt.show()

    elif im.ndim == 3:
        # Input image is three channels
        fig = plt.figure()
        fig.add_subplot(311)
        #bierzemy tylko 1 składowa, max range, range obrazu, kolor wykresu
        plt.hist(im[...,0].flatten(), 256, range=(0, 250), fc='b')
        fig.add_subplot(312)
        plt.hist(im[...,1].flatten(), 256, range=(0, 250), fc='g')
        fig.add_subplot(313)
        plt.hist(im[...,2].flatten(), 256, range=(0, 250), fc='r')
        plt.show()

fileName = 'face.png'

if __name__ == '__main__':
    im = cv2.imread(fileName)
    if not (im == None):
        show_histogram(im)
