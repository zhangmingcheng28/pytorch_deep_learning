# -*- coding: utf-8 -*-
"""
@Time    : 2/3/2022 1:40 PM
@Author  : MingCheng
@FileName: 
@Description: 
@Package dependency:
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch


def create_training_data():
    training_data = []

    for category in CATEGORIES:
        path = os.path.join(DATADIR,category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
                training_data.append([new_array,class_num])
            except Exception as e:
                pass

    return np.asarray(training_data, dtype="object")

alexnet = torchvision.models.alexnet(pretrained=False)

DATADIR = 'D:/flowerData/flowers'
CATEGORIES = ["daisy", "dandelion", "rose", "sunflower", "tulip"]
IMG_SIZE = 28

training_data = create_training_data()
