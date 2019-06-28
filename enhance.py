#! python3
from PIL import Image
import numpy as np
from scipy import ndimage, misc, signal
import scipy
import matplotlib.pyplot as plt
import matplotlib.cm
import cv2
import math
import sys

from numba import jit


@jit(nopython=True)
def normalize(image):
   image = np.asarray(image).astype(np.float32)
   image = (image - image.min()) / (image.max() - image.min())
   image = image * 255
   return image.astype(np.uint8)


@jit(nopython=True)
def contrast(image, alpha=150, y=95):
# def contrast(image, alpha=150, y=95):
   image = image.astype(np.float32)

   mean = np.mean(image)
   var = np.std(image)
   image = alpha + y * (image - mean) / var

   image = np.where(image < 0, 0, image)
   image = np.where(image > 255, 255, image)
   image = normalize(image)
   return image.astype(np.uint8)


@jit(nopython=True)
def median_filter(img, filter_size):
   median = img.copy()
   hs = filter_size//2
   for i in range(hs, img.shape[0]-hs):
      for j in range(hs, img.shape[1]-hs):
         pixels = sorted(img[i-hs: i+hs+1, j-hs: j+hs+1].flatten())
         # dislocated median
         median[i, j] = pixels[8+ filter_size*filter_size // 2]
   return median
   # cv2.erode
   # return cv2.medianBlur(img, filter_size)

def binarize(img, blk_sz=3):
   img = img.copy()
   per25 = np.percentile(img, 25)
   per50 = np.percentile(img, 30)
   # print('per25')
   # print(per25)
   # print('per50')
   # print(per50)

   # hist, bins = np.histogram(img, 256, [0, 256])
   # plt.hist(img.ravel(), 256, [0, 256])
   # plt.title('Histogram for gray scale picture')
   # plt.show()
   means = np.zeros(img.shape)

   # number of blocks in a dimension
   blk_no_y, blk_no_x = (int(img.shape[0]//blk_sz), int(img.shape[1]//blk_sz))
   blk_mean = np.zeros((blk_no_y, blk_no_x))
   # for each block i,j
   for i in range(blk_no_y):
      for j in range(blk_no_x):
         block = img[blk_sz*i: blk_sz*(i+1), blk_sz*j: blk_sz*(j+1)]
         blk_mean[i, j] = np.mean(block)

   img = np.where(img < per25, 0, img)
   img = np.where(img >= per50, 255, img)

   for i in range(1, img.shape[0]-1):
      for j in range(1, img.shape[1]-1):
         if(img[i, j] == 0 or img[i, j] == 255):
            continue
         block = img[i-1: i+1+1, j-1: j+1+1]
         block = np.ma.array(block.flatten(), mask=False)
         block.mask[len(block)//2] = True
         
         if(block.mean() >= blk_mean[i//blk_sz, j//blk_sz]):
            img[i, j] = 255
         else:
            # img[i, j] = 255
            # better
            img[i, j] = 255
   return img


def smooth_bin_filter(img, blk_sz, fil_sz, thresh):
   img_smo = img.copy()
   for i in range(fil_sz, img.shape[0]-fil_sz):
      for j in range(fil_sz, img.shape[1]-fil_sz):
         block = img[i - fil_sz: i+fil_sz+1, j-fil_sz: j+fil_sz+1]
         black_no = np.sum(block)
         white_no = fil_sz**2 - black_no
         if(black_no >= thresh):
            img_smo[i, j] = 1
         if(white_no >= thresh):
            img_smo[i, j] = 0

   return img_smo

def smooth_bin(img, blk_sz):
   img_smo = img.copy()
   for fil_sz, thresh in [(3, 18),(1, 5)]:
      img_smo = smooth_bin_filter(img, blk_sz, fil_sz, thresh)

   return img_smo

def region_of_interest(img, blk_sz, gray_out=125):
   img = img.copy()
   #
   # number of blocks in a dimension
   blk_no_y, blk_no_x = (int(img.shape[0]//blk_sz), int(img.shape[1]//blk_sz))
   blk_mean = np.zeros((blk_no_y, blk_no_x))
   blk_std = np.zeros((blk_no_y, blk_no_x))
   # matrix mask of valid blocks, 0: valid; 1: invalid; as in numpy.ma
   roi_blks = np.zeros(img.shape)
   # for each block i,j
   for i in range(blk_no_y):
      for j in range(blk_no_x):
            block = img[blk_sz*i: blk_sz*(i+1), blk_sz*j: blk_sz*(j+1)]
            blk_mean[i, j] = np.mean(block)
            blk_std[i, j] = np.std(block)
   # normalize values
   blk_mean = (blk_mean-np.min(blk_mean))/(np.max(blk_mean) - np.min(blk_mean))
   blk_std = (blk_std-np.min(blk_std))/(np.max(blk_std) - np.min(blk_std))
   # constant
   max_dist_to_center = np.sqrt(
       np.square((blk_no_y/2)) + np.square((blk_no_x/2)))
   # max_dist_to_center = (blk_no_y + blk_no_x)/2
   # for each block i,j
   for i in range(blk_no_y):
      for j in range(blk_no_x):
            w0 = 0.5
            w1 = 0.5
            dist_to_center = np.sqrt(
                np.square(i-(blk_no_y/2)) + np.square(j-(blk_no_x/2)))
            w2 = 1 - (dist_to_center/max_dist_to_center)

            v = w0*(1-blk_mean[i, j]) + w1*blk_std[i, j] + w2
            if (v < 0.8):
               # gray out block
               roi_blks[blk_sz*i:blk_sz *
                        (i+1), blk_sz*j:blk_sz*(j+1)] = 1
               img[blk_sz*i:blk_sz*(i+1), blk_sz*j:blk_sz*(j+1)] = gray_out

   return img, roi_blks
