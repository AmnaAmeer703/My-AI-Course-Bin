import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications import VGG16
from tensorflow.keras.regularizers import l2
import pandas as pd
import numpy as np
import os

train_data = 'chest_xray/train'
test_data = 'chest_xray/test'

classes_name = os.listdir(train_data)
ptint(classes_name)

classes = []
for folder in os.listdir(train_data):
    classes.append(folder)
print (classes)

def get_df(train_data):
    test_dataset = []
    for root, dirs,files in os.walk(train_data):
        for file in files:
            file_path = os.path.join(root, file)
            labels = root.split('/')[-1]
            test_dataset.append({'file_path':file_path, 'labels':labels})
    return pd.DataFrame(test_dataset)

df_train = get_df(train_data)
print(df_train)

def get_df(test_data):
    test_dataset = []
    for root, dirs,files in os.walk(test_data):
        for file in files:
            file_path = os.path.join(root, file)
            labels = root.split('/')[-1]
            test_dataset.append({'file_path':file_path, 'labels':labels})
    return pd.DataFrame(test_dataset)

df_test= get_df(test_data)
print(df_test)

Labels = df_train['labels'].value_counts()
print(Labels)

print(df_test['labels'].value_counts())

from sklearn.utils import class_weight

import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(x='labels', data=df)
plt.show()

df_test['labels'].value_counts().plot(kind='bar')

from sklearn.model_selection import train_test_split
val_df, test_df = train_test_split(df_test, test_size = 0.5, random_state=42)

img_height, img_width = 224,224
batch_size = 32

# Image Augmentation
train_datagen = ImageDataGenerator(rescale=1.0/255, shear_range=0.2,zoom_range=0.2,horizontal_flip=50,width_shift_range=0.2,height_shift_range=0.2)
val_datagen = ImageDataGenerator(rescale=1.0/255)
train_ds = train_datagen.flow_from_dataframe(
    df_train,
    target_size = (224,224),
    class_mode='categorical',
    labels = ['PNEUMONIA','NORMAL'],
    color_mode='rgb',
    x_col = 'file_path',
    y_col = 'labels',
    batch_size=32
)
val_ds = val_datagen.flow_from_dataframe(
    df_test,
    target_size=(224,224),
    class_mode='categorical',
    labels = ['PNEUMONIA','NORMAL'],
    color_mode='rgb',
    x_col = 'file_path',
    y_col = 'labels',
    batch_size=32
)
test_ds = val_datagen.flow_from_dataframe(
    df_test,
    target_size = (224,224),
    class_mode='categorical',
    labels = ['PNEUMONIA','NORMAL'],
    color_mode='rgb',
    x_col = 'file_path',
    y_col = 'labels',
    batch_size=32,
    shuffle=False
)

train_ds.class_indices
n_sample_0 = 3883
n_sample_1 = 1349
class_weights = {0:5232/n_sample_0,1:5232/n_sample_1}
class_weight = {0:1/n_sample_0, 1:1/n_sample_1}
print(class_weights)

from tensorflow.keras.models import Model
base_model = VGG16(weights='imagenet',include_top=False,input_shape=(224,224,3))
for layer in base_model.layers:
    layer.trainable=False
    
x = base_model.output
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(2, activation='softmax')(x)
Chest_Xray = Model(base_model.input, output)

Chest_Xray.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
history = Chest_Xray.fit(train_ds,validation_data=val_ds,epochs=5)

import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend()
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend()
plt.show()

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

true_labels = test_ds.classes
class_labels = list(val_ds.class_indices.keys())

train_score = Chest_Xray.evaluate(train_ds)
print('train_loss:', train_score[0])
print('train_accuracy:', train_score[1])

test_score = Chest_Xray.evaluate(test_ds)
print('Test loss:', test_score[0])
print('Test accuracy;', test_score[1])

val_score = Chest_Xray.evaluate(val_ds)
print('val loss:', val_score[0])
print('val acc:', val_score[1])



pred = Chest_Xray.predict(test_ds)
y_pred = np.argmax(pred, axis=1)

test_ds.reset()
predictions = Chest_Xray.predict(test_ds, steps= len(test_ds),verbose=0)
pred_labels = np.where(predictions>0.5,1,0)
y_pred = np.agrmax(predictions, axis=1)

num_label = {'NORMAL':0, 'PNEUMONIA':1}

import seaborn as sns
from sklearn.metrics import confusion_matrix
import seaborn as sns
Confusion_matrix=confusion_matrix(test_ds.classes,y_pred)
sns.heatmap(Confusion_matrix, annot=True,fmt='d',cmap='Blues')
plt.show()

print(Confusion_matrix)

from sklearn.metrics import classification_report
print(classification_report(test_ds.classes, y_pred))

from tensorflow.keras.models import load_model
Chest_Xray.save('Chest_Xray.h5',include_optimizer=True)
Chest_Xray.save('Chest_Xray.keras',include_optimizer=True)

