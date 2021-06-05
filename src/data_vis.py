import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

def plot_image(images, nb_rows, nb_cols, figsize=(15, 15)):

    fig, axs = plt.subplots(nb_rows, nb_cols, figsize=figsize)

    n = 0
    for i in range(0, nb_rows):
        for j in range(0, nb_cols):
            axs[i, j].axis('off')
            axs[i, j].imshow(images[n])
            n += 1

def performance_vis(temp, classifier_name):

    #create figure
    figure(num=None, figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k')

    plt.subplot(1, 2, 1)
    plt.plot( 'exp_num', 'train_acc', data=temp, marker='', color='skyblue', linewidth=2,label = "train_accuracy")
    plt.plot( 'exp_num', 'test_acc', data=temp, marker='', color='olive', linewidth=2,label = "test_accuracy")
    plt.title(classifier_name+" train accuracy vs test accuracy")
    plt.xlabel("exp_num")
    plt.ylabel("performance")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot( 'exp_num', 'train_f1', data=temp, marker='', color='skyblue', linewidth=2,label = "train_f1")
    plt.plot( 'exp_num', 'test_f1', data=temp, marker='', color='olive', linewidth=2,label = "test_f1")
    plt.title(classifier_name+" train f1 vs test f1")
    plt.xlabel("exp_num")
    plt.ylabel("performance")
    plt.legend()

    return figure
