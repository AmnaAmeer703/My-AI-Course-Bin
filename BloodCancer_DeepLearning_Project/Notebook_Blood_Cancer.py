import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow import keras
from keras import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D, Flatten
from keras.applications import VGG16
import pandas as pd
import numpy as np


Data = 'archive\Blood cell Cancer [ALL]'

Classes = os.listdir(Data)
print(Classes)

def get_df(Data):
    Dataset = []
    for root, dirs, files in os.walk(Data):
        for file in files:
            file_path = os.path.join(root, file)
            labels = root.split('/')[-1]
            Dataset.append({'file_path':file_path, 'labels':labels})
    return pd.DataFrame(Dataset)
            

df = get_df(Data)
print(df)

Labels = df['labels'].value_counts()
print(Labels)

sns.countplot(x='labels', data=df)
plt.show()

import PIL
from PIL import Image
num_image_per_class = 4
num_classes = len(Classes)
plt.figure(figsize=(12, num_classes*6))
for i, class_name in enumerate(Classes):
    class_folder = os.path.join(Data, class_name)
    image = os.listdir(class_folder)

    for j in range(min(num_image_per_class, len(image))):
        img_path = os.path.join(class_folder,image[j])
        img = Image.open(img_path)

    plt.subplot(num_classes, num_image_per_class, i*num_image_per_class + j + 1)
    plt.imshow(img)
    plt.axis = ('off')
plt.tight_layout()
plt.show()

from sklearn.model_selection import train_test_split
train_df, dummy_df = train_test_split(df, train_size=0.8,random_state=42)
val_df, test_df = train_test_split(dummy_df, train_size=0.5, random_state=42)

img_width, img_height = 224,224
batch_size=32
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255,shear_range=0.2,zoom_range=0.2,horizontal_flip=90,height_shift_range=0.2,width_shift_range=0.2)
datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255)
tarin_ds = train_datagen.flow_from_dataframe(
    train_df,
    target_size = (224,224),
    batch_size =32,
    color_mode = 'rgb',
    class_mode = 'categorical',
    labels = ['[Malignant] early Pre-B','[Malignant] Pro-B','[Malignant] Pre-B','Benign'],
    x_col = 'file_path',
    y_col = 'labels',
    shuffle=False
)
test_ds = datagen.flow_from_dataframe(
    test_df,
    target_size = (224,224),
    batch_size=32,
    color_mode = 'rgb',
    class_mode = 'categorical',
    labels = ['[Malignant] early Pre-B','[Malignant] Pro-B','[Malignant] Pre-B','Benign'],
    x_col = 'file_path',
    y_col = 'labels',
    shuffle=False
)
val_ds = datagen.flow_from_dataframe(
    val_df,
    target_size = (224,224),
    batch_size = 32,
    color_mode = 'rgb',
    class_mode = 'categorical',
    labels = ['[Malignant] early Pre-B','[Malignant] Pro-B','[Malignant] Pre-B','Benign'],
    x_col = 'file_path',
    y_col = 'labels',
    shuffle = False
)

base_model = VGG16(weights='imagenet',include_top=False,input_shape=(224,224,3))
for layer in base_model.layers:
    layer.trainable=False

x = base_model.output
x = Flatten()(x)
x = Dense(128,activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(64,activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(4,activation='softmax')(x)
Blood_Cancer = Model(base_model.input,x)

Blood_Cancer.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

history = Blood_Cancer.fit(tarin_ds,validation_data=val_ds, epochs=10)

import seaborn as sns
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()

train_score = Blood_Cancer.evaluate(tarin_ds)
print('Train Loss:', train_score[0])
print('Train Accuracy:',train_score[1])

test_score = Blood_Cancer.evaluate(test_ds)
print('Test Loss:', test_score[0])
print('Test Accuracy:',test_score[1])

val_score = Blood_Cancer.evaluate(val_ds)
print('Val Loss:', val_score[0])
print('Val Accuracy:', val_score[1])

pred = Blood_Cancer.predict(test_ds)
y_pred = np.argmax(pred, axis=1)

from sklearn.metrics import confusion_matrix, classification_report
confusion_matrix=confusion_matrix(test_ds.classes,y_pred)
sns.heatmap(confusion_matrix,fmt='d', cmap='Blues',annot=True)
plt.show()

print(confusion_matrix)

print(classification_report(test_ds.classes,y_pred))

from keras.models import load_model
Blood_Cancer.save('Blood_Cancer.h5',include_optimizer=True)

from IPython.display import FileLink
FileLink('Blood_Cancer.h5')