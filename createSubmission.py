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

test_datagen = ImageDataGenerator(rescale=1./255, preprocessing_function=preprocess_input)

test_generator = test_datagen.flow_from_directory(
        "./data/test",
        target_size=(100, 100),
        color_mode="rgb",
        shuffle = False,
        batch_size=1)
filenames = test_generator.filenames
nb_samples = len(filenames)
filenames = (os.listdir('./data/test/test'))
print(filenames)

print(filenames[0])
trainPath="./data/train.csv"
df=pd.read_csv(trainPath, sep=",").astype(dtype=str)
getClassNumberByClassName={}
#getClassByFile={}
c=0
getClassNameByNumber={}
for index, row in df.iterrows():
    if getClassNumberByClassName.get(row["Id"]) == None:
        getClassNumberByClassName[row["Id"]]=c
        getClassNameByNumber[c]=row["Id"]
        c+=1
    #getClassByFile[row["Image"]] = row["Id"]


print(getClassNameByNumber)
print(getClassNumberByClassName)

model = load_model("./saves/model-67-0.8666.h5")
preds = model.predict_generator(test_generator, steps=nb_samples)
submission=[]

for i in range(nb_samples):
    file=filenames[i]
    last=-1
    classes=[]
    classes.append("new_whale")
    for j in range(4):
        classId=np.argmax(preds[i],axis=0)
        classes.append(getClassNameByNumber.get(classId))
        preds[i][classId]=0
    submission.append([file, " ".join(classes)])

print(submission)


np.savetxt("submission.csv", X=submission, fmt="%s", delimiter=",", header="Image,Id", comments='')