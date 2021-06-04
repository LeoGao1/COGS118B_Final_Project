import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

def plot_image(images, nb_rows, nb_cols, figsize=(15, 15)):

    fig, axs = plt.subplots(nb_rows, nb_cols, figsize=figsize)

    n = 0
    for i in range(0, nb_rows):
        for j in range(0, nb_cols):
            axs[i, j].axis('off')
            axs[i, j].imshow(images[n])
            n += 1
