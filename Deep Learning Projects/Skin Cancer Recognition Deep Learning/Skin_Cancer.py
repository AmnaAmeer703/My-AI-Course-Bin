import pandas as pd
import numpy as np
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings('ignore')

import sys
print(sys.version)

part1 = 'archive\HAM10000_images_part_1'
part2 = 'archive\HAM10000_images_part_2'

df = df = pd.read_csv('archive\HAM10000_metadata.csv')
print(df)

dataset_path = 'archive'

metadata = pd.read_csv(os.path.join(dataset_path, "HAM10000_metadata.csv"))

df = df.rename(columns={'dx':'Labels'})

df = df.drop(columns={'lesion_id','dx_type','age','sex','localization'})
print(df)

df['Labels'] = df['Labels'].replace({'bkl':'benign keratosis','nv':'melanocytic','mel':'melanoma','bcc':'basal cell carcinoma','akiec':'Bowens diease','vasc':'vascular lesions','df':'dermatofibroma'})

import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(df,x='Labels')
plt.xticks(rotation=90)
plt.show()

import cv2
from tqdm import tqdm
IMG_SIZE = 128
X = []
Y = []
for i in tqdm(range(df.shape[0])):
    img_id = df['image_id'][i]
    label = df['Labels'][i]
    img_path = os.path.join(part1, img_id + ".jpg")
    if not os.path.exists(img_path):
        img_path = os.path.join(part2, img_id + ".jpg")
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    X.append(img)
    Y.append(label)
X = np.array(X)
y = np.array(Y)

print('Labels:',df['Labels'].value_counts())


import tensorflow as tf
from tensorflow import keras
from keras import Model
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, Flatten, Dense, Dropout, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D
from keras.applications import VGG16, EfficientNetB4
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
X = X / 255.0
le = LabelEncoder()
y_encoded = le.fit_transform(Y)
y_categorical = to_categorical(y_encoded, num_classes = len(le.classes_))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y_categorical,test_size=0.15,random_state=42,stratify=y_categorical)

datagen = ImageDataGenerator(rotation_range=20,horizontal_flip=True,height_shift_range=0.1,width_shift_range=0.1,shear_range=0.1,zoom_range=0.1,fill_mode='nearest')

class_labels = df['Labels'].value_counts()
class_labels

import random
import matplotlib.image as mpimg
plt.figure(figsize=(12,12))
for i in range(20):
    randomImg = random.choice(
    [
        x for x in os.listdir(part1)
    if os.path.isfile(os.path.join(part1, x))
    ]
)
    imageFileName = part1 + str(randomImg)
    plt.subplot(4,7,i+1);
    plt.imshow(mpimg.imread(imageFileName),cmap='gray')
    plt.axis='off'

img_shape = (128,128,3)

import tensorflow as tf
from tensorflow import keras
from keras import Model
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, Flatten, Dense, Dropout, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D
from keras.applications import VGG16, EfficientNetB4

skin_cancer_model = Sequential([
    Conv2D(32, (3,3),activation='relu',input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2,2),
    BatchNormalization(),

    Conv2D(64, (3,3),activation='relu'),
    MaxPooling2D(2,2),
    BatchNormalization(),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    BatchNormalization(),

    Conv2D(256, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    BatchNormalization(),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.1,seed=123),
    Dense(len(le.classes_),activation='softmax')
])

from keras.optimizers import Adam
model.compile(optimizer=Adam(learning_rate=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])

history = skin_cancer_model.fit(datagen.flow(X_train,y_train,batch_size=32),epochs=20,
                        validation_data = (X_test,y_test),verbose=1)

loss, acc = skin_cancer_model.evaluate(X_test,y_test)
print(f'Test accuracy: {acc*100:.2f}%')

train_score = skin_cancer_model.evaluate(X_train,y_train)
print('Train Loss:', train_score[0])
print('Train Accuracy:', train_score[1])

test_score = skin_cancer_model.evaluate(X_test,y_test)
print('Test Loss:', test_score[0])
print('Test Accuracy:', test_score[1])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.show()

y_pred = skin_cancer_model.predict(X_test)
y_pred_classes = np.argmax(y_pred,axis=1)
y_true = np.argmax(y_test,axis=1)

from sklearn.metrics import confusion_matrix,classification_report
print('classification_report:', classification_report(y_true,y_pred_classes,target_names=le.classes_))
cm = confusion_matrix(y_true,y_pred_classes)
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues')
plt.show()

from keras.models import load_model
skin_Cancer_model.save('skin_cancer_model.h5',include_optimizer=True)

from IPython.display import FileLink
FileLink('model.h5')

