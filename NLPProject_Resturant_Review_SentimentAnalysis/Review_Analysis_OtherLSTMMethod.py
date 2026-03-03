import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras_preprocessing import text
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Activation, Bidirectional, Input, Embedding
from keras.optimizers import Adam
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from keras.metrics import Precision, Recall

df = pd.read_csv('Reviews.csv')
print(df.head(2))

print(df.dtypes)
print(df.describe())
print(df.isnull()*100)
print(df.info())

plt.figure(figsize=(20,10))
plt.xticks(rotation=90)
ax = sns.countplot(x = 'Score',data=df)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01, p.get_height() * 1.01 ))
plt.grid()

rating_mapping = {
    1: 'Negative',
    2: 'Negative',
    3: 'Netural',
    4: 'Positive',
    5: 'Positive'
}

df['rating_score'] = df['Score'].map(rating_mapping)

print(df)

print(df['rating_score'].value_counts())

import re
import nltk
nltk.download('all')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
lemm = WordNetLemmatizer()

features = ['Summary','Text']
X = df[features]
y = df['rating_score']

def toLower(df):
    if isinstance(df,float):
        return '<UNK>'
    else:
        return df.lower()
    
stop_words = stopwords.words('english')

def remove_stopwords(text):
    no_stop =[]
    for word in text.split(' '):
        if word not in stop_words:
            no_stop.append(word)
    return " ".join(no_stop)

def remove_punctuation_func(text):
    return re.sub(r'[^a-zA-Z0-9]', ' ', text)

X['Summary'] = X['Summary'].apply(toLower)
X['Text'] = X['Text'].apply(toLower)

X['Summary'] = X['Summary'].apply(remove_stopwords)
X['Text'] = X['Text'].apply(remove_stopwords)

X['Summary'] = X['Summary'].apply(lambda x: lemm.lemmatize(x))
X['Text'] = X['Text'].apply(lambda x: lemm.lemmatize(x))

X['Cleaned_Summary_Text'] = list(X['Summary']+X['Text'])

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['rating_score'] = le.fit_transform(df['rating_score'])


y = df['rating_score'].values


word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    stop_words='english',
    max_features=1500)
word_vectorizer.fit(X['Cleaned_Summary_Text'].values)
WordFeatures = word_vectorizer.transform(X['Cleaned_Summary_Text'].values)

X_train,X_test,y_train,y_test = train_test_split(WordFeatures,y,random_state=0, test_size=0.2)
print(X_train.shape)
print(X_test.shape)

# Convert the sparse tensor to a dense tensor and reshape
X_train_dense = X_train.toarray()
X_train_reshaped = np.reshape(X_train_dense, (X_train_dense.shape[0], 1, X_train_dense.shape[1]))

X_test_dense = X_test.toarray()
X_test_reshaped = np.reshape(X_test_dense, (X_test_dense.shape[0], 1, X_test_dense.shape[1]))

n_hidden = 64
n_classes = 3


model = Sequential()

model.add(LSTM(n_hidden, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])))

model.add(Dense(n_classes))

model.add(Activation('softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train_reshaped, y_train, validation_data=(X_test_reshaped, y_test), batch_size=32, epochs=10)

train_loss = history.history['loss']
train_accuracy = history.history['accuracy']
val_loss = history.history['val_loss']
val_accuracy = history.history['val_accuracy']

# Plotting training and validation loss
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracy, label='Training Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

train_loss, train_accuracy = model.evaluate(X_train_reshaped, y_train)
print(f'Training Accuracy: {train_accuracy}')

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test_reshaped, y_test)
print(f'Test Accuracy: {test_accuracy}')

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Predict the probabilities for the test set
y_pred_prob = model.predict(X_test_reshaped)

# Convert probabilities to class labels
y_pred = np.argmax(y_pred_prob, axis=1)

# Map the index to the corresponding category string using the label encoder
y_test_labels = le.inverse_transform(y_test)
y_pred_labels = le.inverse_transform(y_pred)

# Compute the confusion matrix
cm = confusion_matrix(y_test_labels, y_pred_labels)

# Plot the confusion matrix
plt.figure(figsize=(16, 14))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, 
            yticklabels=le.classes_,
            cbar=True)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
