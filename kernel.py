from pprint import pprint

import pandas as pd
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten, AveragePooling2D
from keras.callbacks import ModelCheckpoint
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
from keras.applications.imagenet_utils import preprocess_input

np.set_printoptions(threshold=np.inf)
channels=3
img_size=100
batch_size=100
ignore_new_whale=True

trainPath="./data/train.csv"
imgPath="./data/train/"


K.set_image_dim_ordering('tf') # color channel last
df=pd.read_csv(trainPath, sep=",").astype(dtype=str)


getClassNumberByClassName={}
getClassByFile={}

c=0
for index, row in df.iterrows():
    if ignore_new_whale and row["Id"] != "new_whale" or not ignore_new_whale:
        if getClassNumberByClassName.get(row["Id"]) == None:
            getClassNumberByClassName[row["Id"]]=c
            c+=1
        getClassByFile[row["Image"]] = row["Id"]

num_classes=len(getClassNumberByClassName.keys())


print(num_classes)

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
        title= "foo"
        # create subplot and append to ax
        ax.append( fig.add_subplot(rows, columns, j+1) )
        ax[-1].set_title(title)
        #print(image.shape)
        plt.imshow(image.astype('uint8'))
    plt.show()


files=[]
for k, v in getClassByFile.items():
    if ignore_new_whale and v!="new_whale" or not ignore_new_whale:
        files.append(k)
x = np.zeros((len(files), img_size, img_size, channels))
y = []


c=0
for file in files:
    y_single = getClassNumberByClassName[getClassByFile[file]]
    if c%500==0:
        print("processing image " + str(c))
    if getClassByFile[file]!="new_whale" or not ignore_new_whale:
        x_single = imread(imgPath + file)
        x_single = resize(x_single, (img_size, img_size))
        x_single = preprocess_input(x_single)
        if len(x_single.shape)==2:
            x_single=np.stack((x_single,) * 3, axis=-1)

        x[c]=x_single
        y.append(y_single)
        c+=1

y = to_categorical(y, num_classes=num_classes)
plotImgs(x[:20], y[:20])

model = Sequential()
# model.add(Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0', input_shape = (img_size, img_size, 3)))
#
# model.add(BatchNormalization(axis = 3, name = 'bn0'))
# model.add(Activation('relu'))
#
# model.add(MaxPooling2D((2, 2), name='max_pool'))
# model.add(Conv2D(64, (3, 3), strides = (1,1), name="conv1"))
# model.add(Activation('relu'))
# model.add(AveragePooling2D((3, 3), name='avg_pool'))
#
# model.add(Flatten())
# model.add(Dense(500, activation="relu", name='rl'))
# model.add(Dropout(0.8))
# model.add(Dense(num_classes, activation='softmax', name='sm'))
model.add(Conv2D(128, (3, 3), input_shape=(img_size, img_size, channels),  padding='same'))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
# model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(256, (3, 3), strides=(2,2), padding='same', activation='relu'))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
# model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(256, (3, 3), strides=(2,2), padding='same', activation='relu'))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
print(model.summary())
checkpoint = ModelCheckpoint('./saves/model-{epoch:02d}-{acc:.4f}.h5', verbose=1, monitor='acc',save_best_only=True, mode='auto')

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(x, y, epochs=10000, batch_size=100, verbose=1, callbacks=[checkpoint])

