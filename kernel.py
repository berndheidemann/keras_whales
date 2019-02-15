from typing import Dict, Any

import pandas as pd
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import GlobalAveragePooling2D
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from skimage.io import imread
from keras import backend as K
from skimage.transform import resize


K.set_image_dim_ordering('tf') # color channel last
trainPath="./data/train.csv"
imgPath="./data/train/"

df=pd.read_csv(trainPath, sep=",").astype(dtype=str)


getClassNumberByClassName={}
getClassByFile={}
c=0
for index, row in df.iterrows():
    if getClassNumberByClassName.get(row["Id"]) == None:
        getClassNumberByClassName[row["Id"]]=c
        c+=1
    getClassByFile[row["Image"]] = row["Id"]

num_classes=len(getClassNumberByClassName.keys())
img_size=64
batch_size=10

datagen = ImageDataGenerator(
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image
        shear_range=0.2, #Scherwinkel
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True ) # randomly flip images


def plotImgs(x, y):
    ax = []
    columns = 3
    rows = 3
    w = img_size
    h = img_size
    fig = plt.figure(figsize=(9, 13))

    for j in range( columns*rows ):
        i = np.random.randint(0, len(x))
        image = x[i] * 255
        title= y[i]
        # create subplot and append to ax
        ax.append( fig.add_subplot(rows, columns, j+1) )
        ax[-1].set_title(title)
        plt.imshow(image.astype('uint8'))
    plt.show()

def image_generator(getClassByFile, batch_size=64):
    ids=[]
    for k, v in getClassByFile.items():
        ids.append(k)

    while True:
        batch_paths = np.random.choice(ids, batch_size)
        batch_input = np.zeros((batch_size, img_size, img_size, 3))
        batch_output = []

        # Read in each input, perform preprocessing and get labels
        for file in batch_paths:
            x = imread(imgPath + file)
           # x.resize((img_size,img_size, 3))

            x = resize(x, (img_size, img_size))


            x = np.reshape(x, (img_size, img_size, 1))

            y = getClassNumberByClassName[getClassByFile[file]]
            #preprocess dataAugment
            batch_input += [x]
            batch_output.append(to_categorical(y, num_classes=num_classes))

        # Return a tuple of (input,output) to feed the network
        batch_x = np.array(batch_input)
        batch_y = np.array(batch_output)

        plotImgs(batch_x, batch_y)
       # print(batch_x.shape)
       # print(batch_y.shape)

        yield (batch_x, batch_y)


model = Sequential()
model.add(Conv2D(128, (3, 3), input_shape=(img_size, img_size, 1), padding='same', activation='relu'))
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(128, (3, 3), strides=(2,2), padding='same', activation='relu'))
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(128, (3, 3), strides=(2,2), padding='same', activation='relu'))
# model.add(Dropout(0.4))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(1000, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(2500, activation='relu'))
# model.add(Dropout(0.4))
model.add(Dense(num_classes, activation='softmax'))
print(model.summary())

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit_generator(
        image_generator(getClassByFile, batch_size=batch_size),
        steps_per_epoch=2000 // batch_size,
        epochs=30,
        validation_data=image_generator(getClassByFile),
       validation_steps=800 // batch_size)
