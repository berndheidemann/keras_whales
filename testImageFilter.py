from keras.applications.imagenet_utils import preprocess_input
import os
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt


nb_img=4

files=os.listdir("./data/train")[0:nb_img]
labels=["original", "preprocessed", "pre & IDG", "only IDG"]

def plotImgs(x, title):
    columns = 4
    rows = 4
    w = 100
    h = 100
    fig = plt.figure(figsize=(9, 13))
    for j in range( columns*rows ):

        image = x[j]
        if j>7:
            image*=255.0
        subplot=fig.add_subplot(rows, columns, j+1)
        subplot.set_title(labels[j//4])
        if j<4 or j>7:
            plt.imshow(image.astype('uint8'))
        else:
            plt.imshow(image)
    plt.title(title)
    plt.show()


images=np.zeros((nb_img*4, 100, 100, 3), dtype=float)
for i,file in enumerate(files):
    img=image.load_img("./data/train/" + file, target_size=(100, 100, 3))
    for j in range(0, 16, 4):
        images[i+j]=np.copy(img)

images=np.asarray(images, dtype=float)

images[4:12]=preprocess_input(images[4:12])


datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=False)
datagen.fit(images[8:16])
for x, y in datagen.flow(images[8:16], [0]*8, batch_size=8, shuffle=False):
    images[8:16]=x
    break

plotImgs(images, "foo")
