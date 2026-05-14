import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
keras = tf.keras
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


utils =  keras.utils
layers = keras.layers

from keras.utils import to_categorical


from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.metrics import Precision, Recall
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
    # Load the CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = df



    # 1. Function for showing images
def show_images(train_images, 
                class_names, 
                train_labels, 
                nb_samples = 12, nb_row = 4):
        
        plt.figure(figsize=(12, 12))
        for i in range(nb_samples):
            plt.subplot(nb_row, nb_row, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(train_images[i], cmap=plt.cm.binary)
            plt.xlabel(class_names[train_labels[i][0]])
        plt.show()


        # Visualize some sample images from the dataset
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']

show_images(train_images, class_names, train_labels)




    # Data normalization
max_pixel_value = 255

train_images = train_images / max_pixel_value
test_images = test_images / max_pixel_value

train_labels = to_categorical(train_labels, len(class_names))
test_labels = to_categorical(test_labels, len(class_names))





    # Variables
INPUT_SHAPE = (32, 32, 3)
FILTER1_SIZE = 32
FILTER2_SIZE = 64
FILTER_SHAPE = (3, 3)
POOL_SHAPE = (2, 2)
FULLY_CONNECT_NUM = 128
NUM_CLASSES = len(class_names)

    # Model architecture implementation
model = Sequential()
model.add(Conv2D(FILTER1_SIZE, FILTER_SHAPE, activation='relu', input_shape=INPUT_SHAPE))
model.add(MaxPooling2D(POOL_SHAPE))
model.add(Conv2D(FILTER2_SIZE, FILTER_SHAPE, activation='relu'))
model.add(MaxPooling2D(POOL_SHAPE))
model.add(Flatten())
model.add(Dense(FULLY_CONNECT_NUM, activation='relu'))
model.add(Dense(NUM_CLASSES, activation='softmax'))





BATCH_SIZE = 32
EPOCHS = 30

METRICS = metrics=['accuracy', 
                    Precision(name='precision'),
                    Recall(name='recall')]

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics = METRICS)

    # Train the model
training_history = model.fit(train_images, train_labels, 
                             epochs=EPOCHS, batch_size=BATCH_SIZE,
                             validation_data=(test_images, test_labels))


print ("Model Summary  : \n" , model.summary())


import matplotlib.pyplot as plt
        
def show_performance_curve(training_result, metric, metric_label):
      train_perf = training_result.history[str(metric)]
      validation_perf = training_result.history['val_'+str(metric)]
      intersection_idx = np.argwhere(np.isclose(train_perf, 
                                                validation_perf, atol=1e-2)).flatten()[0]
      intersection_value = train_perf[intersection_idx]
      plt.plot(train_perf, label=metric_label)
      plt.plot(validation_perf, label = 'val_'+str(metric))
      plt.axvline(x=intersection_idx, color='r', linestyle='--', label='Intersection')
        
      plt.annotate(f'Optimal Value: {intersection_value:.4f}',
                   xy=(intersection_idx, intersection_value),
                   xycoords='data',
                   fontsize=10,
                   color='green')
                    
      plt.xlabel('Epoch')
      plt.ylabel(metric_label)
      plt.legend(loc='lower right')
      
      show_performance_curve(training_history, 'accuracy', 'accuracy')

      show_performance_curve(training_history, 'recall', 'recall')

      show_performance_curve(training_history, 'precision', 'precision')
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
test_predictions = model.predict(test_images)

    # Convert predictions from probabilities to class labels
test_predicted_labels = np.argmax(test_predictions, axis=1)

    # Convert one-hot encoded true labels back to class labels
test_true_labels = np.argmax(test_labels, axis=1)

    # Compute the confusion matrix
cm = confusion_matrix(test_true_labels, test_predicted_labels)

    # Create a ConfusionMatrixDisplay instance
cmd = ConfusionMatrixDisplay(confusion_matrix=cm)

    # Plot the confusion matrix
cmd.plot(include_values=True, cmap='viridis', ax=None, xticks_rotation='horizontal')
plt.show()

read  = input("Wait ....")