import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, MaxPooling2D, Conv2D, GlobalAveragePooling2D, Flatten
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

data = 'chicken-disease/Train'
df = pd.read_csv('chicken-disease/train_data.csv')

df['img_path'] = df['images'].apply(lambda x:os.path.join(data, x))

print(df)

Labels = df['label'].value_counts()
print(Labels)

sns.countplot(x='label', data=df)
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
plt.pie(df['label'].value_counts(),labels = ['Salmonella','Coccidiosis','New Castle Disease','Healthy'],autopct = '%1.f%%',shadow=True,explode=(0,0.5,0,0))
plt.show()

from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(df['label']),y=df['label'])
class_weights

df['img_path'] = df['images'].apply(lambda x:os.path.join(data, x))

print(df)

from sklearn.model_selection import train_test_split
train_df, dummy_df = train_test_split(df, train_size=0.8, random_state=42)
val_df, test_df = train_test_split(dummy_df, train_size=0.5, random_state=42)

# Data Augmentation

train_datagen = ImageDataGenerator(rescale=1.0/255, shear_range=0.2,zoom_range=0.2,rotation_range=20,height_shift_range=0.2,width_shift_range=0.2,fill_mode='nearest')
datagen = ImageDataGenerator(rescale=1.0/255)

train_ds = train_datagen.flow_from_dataframe(
    train_df,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    labels = ['Coccidiosis','Healthy','New Castle Disease','Salmonella'],
    color_mode = 'rgb',
    x_col = 'img_path',
    y_col = 'label',
    shuffle = False
)
test_ds = datagen.flow_from_dataframe(
    test_df,
    target_size = (224,224),
    batch_size= 32,
    class_mode = 'categorical',
    labels = ['Coccidiosis','Healthy','New Castle Disease','Salmonella'],
    color_mode= 'rgb',
    x_col = 'img_path',
    y_col = 'label',
    shuffle = False
)
val_ds = datagen.flow_from_dataframe(
    val_df,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    labels = ['Coccidiosis','Healthy','New Castle Disease','Salmonella'],
    color_mode = 'rgb',
    x_col = 'img_path',
    y_col = 'label',
    shuffle = False
)

# CNN MODEL
model = Sequential()
model.add(Conv2D(128,kernel_size=(3,3),padding='same',input_shape=(224,224,3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='same'))

model.add(Conv2D(64,kernel_size=(3,3),padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='same'))

model.add(Conv2D(32,kernel_size=(3,3),padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='same'))

model.add(Conv2D(16,kernel_size=(3,3),padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='same'))

model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(16,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(4, activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.summary()

history = model.fit(train_ds, validation_data=val_ds,epochs=20)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.show()

train_score = model.evaluate(train_ds)
print('Train Loss:', train_score[0])
print('Train Accuracy:', train_score[1])

test_score = model.evaluate(test_ds)
print('Test Loss:', test_score[0])
print('Test Accuracy:', test_score[1])

val_score = model.evaluate(val_ds)
print('Val Loss:', val_score[0])
print('Val Accuracy:', val_score[1])

pred = model.predict(test_ds)
y_pred = np.argmax(pred, axis=1)

from sklearn.metrics import confusion_matrix, classification_report
confusion_matrix = confusion_matrix(test_ds.classes, y_pred)
sns.heatmap(confusion_matrix, fmt='d', annot=True, cmap='Blues')
plt.show()

print(confusion_matrix)

print(classification_report(test_ds.classes, y_pred))

from tensorflow.keras.models import load_model
model.save('model.h5',include_optimizer=True)
model.save('model.keras',include_optimizer=True)

from IPython.display import FileLink
FileLink('model.h5')