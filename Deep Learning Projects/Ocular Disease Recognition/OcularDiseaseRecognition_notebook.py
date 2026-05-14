import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, MaxPooling2D, Conv2D, BatchNormalization
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

classes = os.listdir(data)
print(classes)

def get_df(data):
    Dataset = []
    for root, dirs, files in os.walk(data):
        for file in files:
            file_path = os.path.join(root, file)
            labels = root.split('/')[-1]
            Dataset.append({'file_path':file_path, 'labels':labels})
    return pd.DataFrame(Dataset)

df = get_df(data)
print(df)

class_labels, class_counts = np.unique(df['labels'],return_counts=True)
print(class_labels)

Labels = df['labels'].value_counts()
print(Labels)

sns.countplot(x='labels',data=df)
plt.show()

explode = [0.05,0.05,0.05,0.05]
my_explode = [0,0.2,0,0]
plt.pie(df['labels'].value_counts(), labels=['glaucoma','normal','diabetic_retinopathy','cataract'],autopct='%1.f%%',shadow=True,explode=my_explode)
plt.show()

labels = ['glaucoma','normal','diabetic_retinopathy','cataract']
Labels = df['labels'].value_counts()
colors = ['#FF0000','#0000FF','#FFFF00','#ADFF2F','#FFA500']
explode = (0.05,0.05,0.05,0.05)
plt.pie(Labels,colors=colors,labels=labels,autopct = '%1.f%%',pctdistance=0.85,explode=explode)
centre_circle = plt.Circle((0,0),0.70, fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.title('Eye Diseases')
plt.show()

import PIL
from PIL import Image
num_image_per_class = 4
num_classes = len(classes)
plt.figure(figsize=(12, num_classes*6))
for i, class_name in enumerate(classes):
    class_folder = os.path.join(data, class_name)
    image = os.listdir(class_folder)
    
    for j in range(min(num_image_per_class, len(image))):
        
        image_path = os.path.join(class_folder, image[j])
        img = Image.open(image_path)
        plt.subplot(num_classes, num_image_per_class, i*num_image_per_class + j +1)
        plt.imshow(img)
        plt.asix = ('off')
        plt.title(class_name)
plt.tight_layout()
plt.show()


from sklearn.model_selection import train_test_split
train_df, dummy_df = train_test_split(df, train_size=0.8, random_state=42)
val_df, test_df = train_test_split(dummy_df, train_size=0.5, random_state=42)

# Data Augmentation

train_datagen = ImageDataGenerator(rescale=1.0/255, shear_range=0.2,rotation_range=20,horizontal_flip=True,fill_mode='nearest',height_shift_range=0.2,width_shift_range=0.2)
datagen = ImageDataGenerator(rescale=1.0/255)
train_ds = train_datagen.flow_from_dataframe(
    train_df,
    target_size = (224,224),
    batch_size = 32,
    class_mode = 'categorical',
    labels = ['cataract','diabetic_retionpathy','glaucoma','normal'],
    x_col = 'file_path',
    y_col = 'labels',
    color_mode = 'rgb',
    shuffle = False
)
test_val = datagen.flow_from_dataframe(
    test_df,
    target_size = (224,224),
    batch_size = 32,
    class_mode = 'categorical',
    labels = ['cataract','diabetic_retinopathy','glaucoma','normal'],
    x_col = 'file_path',
    y_col = 'labels',
    color_mode = 'rgb',
    shuffle = False
)
val_ds = datagen.flow_from_dataframe(
    val_df,
    target_size = (224,224),
    batch_szie = 32,
    class_mode = 'categorical',
    labels = ['cataract','diabetic_retionpathy','glaucoma','normal'],
    x_col = 'file_path',
    y_col = 'labels',
    color_mode = 'rgb',
    shuffle = False
)

base_model = VGG16(weights='imagenet',include_top=False,input_shape=(224,224,3))
for layer in base_model.layers:
    layer.trainable = False
    
x = base_model.output
x = Flatten()(x)
x = Dense(128,activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(32, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(16, activation='relu')(x)
x = Dropout(0.2)(x)
output = Dense(4, activation='softmax')(x)
eye = Model(base_model.input, output)

eye.compile(optimizer='adam',loss = 'categorical_crossentropy',metrics=['accuracy'])

eye.summary()

history = eye.fit(train_ds, validation_data=val_ds, epochs=50)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.show()

train_score = eye.evaluate(train_ds)
print('Train Loss:', train_score[0])
print('Train Accuracy:', train_score[1])

test_score = eye.evaluate(test_val)
print('Test Loss:', test_score[0])
print('Test Accuracy:', test_score[1])

val_score = eye.evaluate(val_ds)
print('Val Loss:', val_score[0])
print('Val Accuracy:', val_score[1])

pred = eye.predict(test_val)
y_pred = np.argmax(pred, axis=1)

from sklearn.metrics import confusion_matrix, classification_report
confusion_matrix = confusion_matrix(test_val.classes, y_pred)
sns.heatmap(confusion_matrix, fmt='d',cmap = 'Blues',annot=True)
plt.show()

print(classification_report(test_val.classes, y_pred))

from tensorflow .keras.models import load_model
eye.save('eye.h5',include_optimizer=True)

from IPython.display import FileLink
FileLink('eye.h5')

