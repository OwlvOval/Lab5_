# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 00:31:53 2024

@author: Owl
"""

import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from matplotlib import colors
from matplotlib.colors import hsv_to_rgb


def segment_image(image):

    # Convert the image into HSV
    hsv_image = cv.cvtColor(image, cv.COLOR_RGB2HSV)

    # Set the blue range
    #360 гр hue не уместилось в байт, поэтому в opencv hue/2
    lower_magenta = (40, 70, 10)
    upper_magenta = (150, 255, 30)

    # Apply the blue mask
    mask = cv.inRange(hsv_image, lower_magenta, upper_magenta)

    plt.imshow(mask)
    plt.show()

    # Set a white range
    light_white = (60, 20, 0)
    dark_white = (120, 255, 255)

    # Apply the white mask
    mask_white = cv.inRange(hsv_image, light_white, dark_white)

    plt.imshow(mask_white)
    plt.show()
    
    # Combine the two masks
    final_mask = mask + mask_white
    result = cv.bitwise_and(image, image, mask=final_mask)

    # Clean up the segmentation using a blur
    blur = cv.GaussianBlur(result, (5, 5), 0)
    return result


image = cv.imread('./tort.jpg')
image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
plt.imshow(image_rgb)
plt.show()


result = segment_image(image_rgb)

plt.figure(figsize=(15,20))
plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.subplot(1, 2, 2)
#result = cv.cvtColor(result, cv.COLOR_HSV2RGB)
plt.imshow(result)
plt.show()