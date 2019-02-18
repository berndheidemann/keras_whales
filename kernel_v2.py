#fork of https://www.kaggle.com/pestipeti/keras-cnn-starter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout
from PlotLoss import PlotLosses
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)

train_df = pd.read_csv("./data/train.csv")
train_df.head()


def prepareImages(data, m, dataset):
    print("Preparing images")
    X_train = np.zeros((m, 100, 100, 3))
    count = 0

    for fig in data['Image']:
        # load images into images of size 100x100x3
        img = image.load_img("./data/" + dataset + "/" + fig, target_size=(100, 100, 3))
        x = image.img_to_array(img)
        x = preprocess_input(x)

        X_train[count] = x
        if (count % 500 == 0):
            print("Processing image: ", count + 1, ", ", fig)
        count += 1

    return X_train

def prepare_labels(y):
    values = np.array(y)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    # print(integer_encoded)

    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    # print(onehot_encoded)

    y = onehot_encoded
    # print(y.shape)
    return y, label_encoder

X = prepareImages(train_df, train_df.shape[0], "train")
X /= 255

y, label_encoder = prepare_labels(train_df['Id'])

y.shape


datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.3, # Randomly zoom image
        shear_range=0.3, #Scherwinkel
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images


print(X.shape)

model = Sequential()

model.add(Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0', input_shape = (100, 100, 3)))

model.add(BatchNormalization(axis = 3, name = 'bn0'))
model.add(Activation('relu'))

model.add(MaxPooling2D((2, 2), name='max_pool'))
model.add(Conv2D(64, (3, 3), strides = (1,1), name="conv1"))
model.add(Activation('relu'))
model.add(AveragePooling2D((3, 3), name='avg_pool'))

model.add(Flatten())
model.add(Dense(500, activation="relu", name='rl'))
model.add(Dropout(0.8))
model.add(Dense(y.shape[1], activation='softmax', name='sm'))
checkpoint = ModelCheckpoint('./saves/model-{epoch:02d}-{acc:.4f}.h5', verbose=1, monitor='acc',save_best_only=True, mode='auto')

model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
model.summary()

batch_size=100

print(X.shape)
datagen.fit(X)
#history = model.fit(X, y, epochs=100, batch_size=100, verbose=1, callbacks=[checkpoint, PlotLosses(slowlyCutBeginning=False)])
history=model.fit_generator(datagen.flow(X, y, batch_size=100),
          epochs=3000, steps_per_epoch=X.shape[0]//batch_size, callbacks=[checkpoint, PlotLosses(slowlyCutBeginning=False)])
plt.plot(history.history['acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()

