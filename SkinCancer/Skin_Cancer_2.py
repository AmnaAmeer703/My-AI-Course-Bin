import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.models import Sequential
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras import layers, models, optimizers
from keras.layers import GlobalAveragePooling2D, BatchNormalization, Dense, Dropout
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_STAGE_1 = 8  
EPOCHS_STAGE_2 = 15

base_dir = 'archive'
metadata_path = os.path.join(base_dir, 'HAM10000_metadata.csv')
image_dir_1 = os.path.join(base_dir, 'HAM10000_images_part_1')
image_dir_2 = os.path.join(base_dir, 'HAM10000_images_part_2')

df = pd.read_csv(metadata_path)
print('df:',df)


image_path_dict = {}
if os.path.exists(image_dir_1):
    for x in os.listdir(image_dir_1):
        image_path_dict[x.split('.')[0]] = os.path.join(image_dir_1, x)
if os.path.exists(image_dir_2):
    for x in os.listdir(image_dir_2):
        image_path_dict[x.split('.')[0]] = os.path.join(image_dir_2, x)

df['path'] = df['image_id'].map(image_path_dict)
df['dx'] = df['dx'].astype(str)
df = df.dropna(subset=['path'])

train_val_df, test_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df['dx'])


train_df, val_df = train_test_split(train_val_df, test_size=0.17, random_state=42, stratify=train_val_df['dx'])

print(f"Image Train : {len(train_df)} | Image Valid: {len(val_df)} | Image Testing: {len(test_df)}")


train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20, width_shift_range=0.1, height_shift_range=0.1,
    horizontal_flip=True, vertical_flip=True, fill_mode='nearest'
)
val_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_dataframe(
    train_df, x_col='path', y_col='dx', target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'
)
val_generator = val_test_datagen.flow_from_dataframe(
    val_df, x_col='path', y_col='dx', target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False
)

test_generator = val_test_datagen.flow_from_dataframe(
    test_df, x_col='path', y_col='dx', target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False
)


base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(7, activation='softmax')(x)
model = Model(base_model.input,x)


model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=15,
    callbacks=[
        ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=3, verbose=1, mode='max'),
        EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True, mode='max')
    ]
)

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from keras.models import load_model
from IPython.display import FileLink

print('Evaluate The Model')
test_loss, test_acc = model.evaluate(test_generator)
print(f"\n✅ Test Accuravy : {test_acc * 100:.2f}%")


predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

print("\n--- Classification Report (Precision, Recall, F1) ---")
print(classification_report(y_true, y_pred, target_names=class_labels))

print('Confusion Matrix')
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix Between y_true and Predicted Values')
plt.ylabel('Real Values')
plt.xlabel('Predicted Values')
plt.show()

# Salving The Model For App
model.save('model.h5',include_optimer=True)
FileLink('model.h5')