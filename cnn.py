#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 18:28:19 2020

@author: dcoster
"""

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

classifier = Sequential()

classifier.add(Convolution2D(32,3,3, input_shape = (64,64,3),activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2,2)))

classifier.add(Flatten())

classifier.add(Dense())


