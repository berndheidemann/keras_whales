from keras.models import load_model
import pickle
import cv2
from keras import backend as K
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

filenames = os.listdir("./data/test")
nb_samples = len(filenames)

calculate_mean_std_on_train=True

trainPath="./data/train.csv"
df=pd.read_csv(trainPath, sep=",").astype(dtype=str)

if calculate_mean_std_on_train:
    print("Preparing images for statistics")
    X_train = np.zeros((df.shape[0], 100, 100, 3))
    count = 0

    for fig in df['Image']:
        # load images into images of size 100x100x3
        img = image.load_img("./data/train/" + fig, target_size=(100, 100, 3))
        x = image.img_to_array(img)
        x = preprocess_input(x)

        X_train[count] = x
        if (count % 500 == 0):
            print("Processing image: ", count + 1, ", ", fig)
        count += 1
    X_train /= 255

    generator = ImageDataGenerator(featurewise_center=True,
                                   featurewise_std_normalization=True)

# Calculate statistics on train dataset
    generator.fit(X_train)



values = np.array(df['Id'])
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)



X_test = np.zeros((nb_samples, 100, 100, 3))
count = 0

for fig in filenames:
    # load images into images of size 100x100x3
    img = image.load_img("./data/test/" + fig, target_size=(100, 100, 3))
    x = image.img_to_array(img)
    x = preprocess_input(x)

    X_test[count] = x
    if (count % 500 == 0):
        print("Processing image: ", count + 1, ", ", fig)
    count += 1

X_test/=255


if calculate_mean_std_on_train:
    # Apply featurewise_center to test-data with statistics from train data
    X_test -= generator.mean
    # Apply featurewise_std_normalization to test-data with statistics from train data
    X_test /= (generator.std + K.epsilon())
else:
    # Apply featurewise_center to test-data with statistics from train data
    X_test -= [[[0.06615361, 0.06615361, 0.04348451]]]
    # Apply featurewise_std_normalization to test-data with statistics from train data
    X_test /= [[[0.14075036, 0.15493952, 0.15001519]]]

'''
generator.mean  ---> [[[0.06615361 0.06615361 0.04348451]]]
(generator.std + K.epsilon()) ---> [[[0.14075036 0.15493952 0.15001519]]]
'''

print("generator.mean" + str(generator.mean))
print("(generator.std + K.epsilon())" + str((generator.std + K.epsilon())))

def plotImgs(x, y):
    ax = []
    columns = 3
    rows = 3
    w = 100
    h = 100
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


model = load_model("./saves/model-955-0.5748.h5")
preds = model.predict(X_test)
submission=[]
plotImgs(X_test, np.zeros((100)))

for i in range(nb_samples):
    print(preds[i].argsort()[-5:])
    preds[i].argsort()[-5:][::-1]
    submission.append([filenames[i], ' '.join(label_encoder.inverse_transform(preds[i].argsort()[-5:][::-1]))])

print(submission)


np.savetxt("submission.csv", X=submission, fmt="%s", delimiter=",", header="Image,Id", comments='')