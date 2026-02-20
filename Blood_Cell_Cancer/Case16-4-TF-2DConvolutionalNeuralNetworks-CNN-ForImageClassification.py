from contextlib import suppress

with suppress(Exception):

    """"
    https://www.datacamp.com/tutorial/cnn-tensorflow-python

    https://www.datacamp.com/datalab/w/2d1a3f51-ae90-41f7-bfbe-4f47b70c32e8


    """

    import tensorflow as tf
    print("TensorFlow version:", tf.__version__)


    """
    a. Tensors

    """
    # Zero dimensional tensor
    zero_dim_tensor = tf.constant(20)
    print(zero_dim_tensor)
    
    # One dimensional tensor
    one_dim_tensor = tf.constant([12, 20, 53, 26, 11, 56])
    print(one_dim_tensor)
    
    # Two dimensional tensor
    two_dim_array = [[3, 6, 7, 5], 
                    [9, 2, 3, 4],
                    [7, 1, 10,6],
                    [0, 8, 11,2]]
    
    two_dim_tensor = tf.constant(two_dim_array)
    print(two_dim_tensor)


    """
    2. Step-by-step implementation
    a. Load the dataset

    =================

    https://keras.io/api/datasets/cifar10/

    Loads the CIFAR10 dataset.

    This is a dataset of 50,000 32x32 color training images and 10,000 test images, labeled over 10 categories. See more info at the CIFAR homepage.

    The classes are:

    Label	Description
    0	airplane
    1	automobile
    2	bird
    3	cat
    4	deer
    5	dog
    6	frog
    7	horse
    8	ship
    9	truck
    Returns

    Tuple of NumPy arrays: (x_train, y_train), (x_test, y_test).

    """

    import numpy as np
    import matplotlib.pyplot as plt

    keras = tf.keras

    cf10 = keras.datasets.cifar10
    kutils =  keras.utils
    klayers = keras.layers

    from keras.utils import to_categorical


    from keras import Sequential
    from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
    from keras.metrics import Precision, Recall

    # Load the CIFAR-10 dataset
    (train_images, train_labels), (test_images, test_labels) = cf10.load_data()


    """
    b. Exploratory Data Analysis

    """

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

    # One-hot encode the labels
    #from tensorflow.keras.utils import to_categorical
    """
    Also, we notice that the labels are represented in a categorical format like cat, horse, bird, and so one. We need to convert them into a numerical format 
    so that they can be easily processed by the neural network.
    """
    train_labels = to_categorical(train_labels, len(class_names))
    test_labels = to_categorical(test_labels, len(class_names))


    """
    Model architecture

    Model architecture implementation
    The next step is to implement the architecture of the network based on the previous description. 

    First, we define the model using the Sequential() class, and each layer is added to the model with the add() function. 


    from tensorflow.keras import Seq


    """



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


    """
    Model training

    """



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

    """

    Evaluate the model.

    See for details : https://github.com/ShahzadSarwar10/AI-ML-Explorer/blob/main/Week5/Artificial%20Intelligence%20(Machine%20Learning%20%26%20Deep%20Learning)-Week5%20-%20Machine%20Learning%20-%20Important%20Concepts%20-%20Notes_Rev1.pdf

    """
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

    #test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    #show_performance_curve(training_history, 'accuracy', 'accuracy')

    #test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    show_performance_curve(training_history, 'accuracy', 'accuracy')

    show_performance_curve(training_history, 'recall', 'recall')

    show_performance_curve(training_history, 'precision', 'precision')


    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    # Obtain the model's predictions on the test dataset
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