import os
import cv2
import glob
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import get_embd


def generate_data():
    lists = os.listdir('./test 2')
    # print(len(lists))
    images = []
    labels = []
    paths = []
    for list in lists:
        # print(lists[i])
        path = glob.glob('./test 2/' + list + '/00*.jpg')
        # print(path[0])
        img = misc.imread(path[0])
        img = misc.imresize(img, [112, 112])
        img = img / 127.5 - 1.0
        images.append(img)
        labels.append(list)
        paths.append(path[0])
    return (np.array(images), labels, paths)
    # print(images.shape)
    # print(len(labels))
    # print(labels[4])
    # print(paths[4])
    # # cv2.imshow('image',images[4])
    # # cv2.waitKey(0)
    # plt.imshow(images[4])  # 显示图片
    # plt.axis('off')  # 不显示坐标轴
    # plt.show()
