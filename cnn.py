#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 16:40:36 2018

@author: vijendrasharma
"""

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten

clf= Sequential()
clf.add( Convolution2D( 32,3,3, input_shape=(64,64,3), activation='relu' ))
# 32 feature maps of size 3*3
#input image 64*64*3channels... activation to increase nonlinearity

clf.add(MaxPooling2D( pool_size=( 2,2) ))

clf.add( Flatten( ))

clf.add(Dense( output_dim=128 , activation='relu'  ))
clf.add(Dense( output_dim=1 , activation='sigmoid' ))

clf.compile( optimizer= 'adam', loss='binary_crossentropy' , metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)
#each pixel varies from 0 to 1

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

clf.fit_generator(training_set,
                         samples_per_epoch = 8000,
                         nb_epoch = 5,
                         validation_data = test_set,
                         nb_val_samples = 2000)



#accuracy on test set achieved 76.89




