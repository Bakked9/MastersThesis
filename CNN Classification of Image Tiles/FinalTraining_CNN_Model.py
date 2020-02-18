# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 08:28:35 2020

@author: BRENDA
"""

# organize dataset into a useful structure
from os import makedirs
from os import listdir
from shutil import copyfile
# create directories
dataset_home = 'C:/Users/BRENDA/Desktop/CNN_FINAL/'
# create label subdirectories
labeldirs = ['Formal/', 'Informal/']
for labldir in labeldirs:
	newdir = dataset_home + labldir
	makedirs(newdir, exist_ok=True)
# copy training dataset images into subdirectories
src_directory = 'C:/Users/BRENDA/Desktop/THESIS/Dataz/Training'
for file in listdir(src_directory):
	src = src_directory + '/' + file
	if file.startswith('Formal'):
		dst = dataset_home + 'Formal/'  + file
		copyfile(src, dst)
	elif file.startswith('Informal'):
		dst = dataset_home + 'Informal/'  + file
		copyfile(src, dst)
        
       


###########PART 6 data augmentation ############
import sys
from matplotlib import pyplot
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

# define cnn model
def define_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(255, 255, 3)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	opt = SGD(lr=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
	return model


# run the test harness for evaluating a model
def run_test_harness():
	# define model
	model = define_model()
	# create data generators
	train_datagen = ImageDataGenerator(rescale=1.0/255.0,
		width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
	test_datagen = ImageDataGenerator(rescale=1.0/255.0)
	# prepare iterators
	train_it = train_datagen.flow_from_directory('C:/Users/BRENDA/Desktop/CNN_FINAL/',
		class_mode='binary', batch_size=64, target_size=(255, 255))
	# fit model
	history = model.fit_generator(train_it, steps_per_epoch=len(train_it),epochs=50, verbose=0)
    # save model
	model.save('C:/Users/BRENDA/Desktop/THESIS/CODE/finalCNN_model.h5')


# entry point, run the test harness
run_test_harness()



#https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-photos-of-dogs-and-cats/