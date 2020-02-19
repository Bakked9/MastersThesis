# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

# Load model
model = load_model('path to model.h5')

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
        "path to Folder with images to Predict",
        target_size=(255, 255),
        batch_size=64,
        class_mode='binary',
        shuffle=False)

# Predict from generator (returns probabilities)
pred=model.predict_generator(test_generator, steps=len(test_generator), verbose=1)

# Get classes by np.round
cl = np.round(pred)
# Get filenames (set shuffle=false in generator is important)
filenames=test_generator.filenames

# Data frame
results=pd.DataFrame({"file":filenames,"pr":pred[:,0], "class":cl[:,0]})




#https://stackoverflow.com/questions/52270177/how-to-use-predict-generator-on-new-images-keras
