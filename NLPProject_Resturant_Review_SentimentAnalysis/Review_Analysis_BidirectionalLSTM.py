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

y = df['rating_score']

vocab_size= 10000
max_len = 40
n_classes = 3


tokenizer = text.Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(X['Cleaned_Summary_Text'])
sequences = tokenizer.texts_to_sequences(X['Cleaned_Summary_Text'])

print("Word Index:", tokenizer.word_index)
print("Sequences:", sequences)

X = pad_sequences(sequences, maxlen=max_len)
print("Padded Sequences:", X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print("X_train:", X_train)
print("y_train:", y_train)
print("X_test:", X_test)
print("y_test:", y_test)

model = Sequential([
    Input(shape=(10000,)),
    Embedding(input_dim=vocab_size, output_dim=128),
    Bidirectional(LSTM(64)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(n_classes, activation='softmax')  # softmax for multi-class single-label
])

METRICS = metrics=['accuracy', 
                   Precision(name='precision'),
                   Recall(name='recall')]

model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=METRICS)
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=1, batch_size=1, validation_data=(X_test, y_test))

train_loss = history.history['loss']
train_accuracy = history.history['accuracy']
val_loss = history.history['val_loss']
val_accuracy = history.history['val_accuracy']


import matplotlib.pyplot as plt
plt.figure(figsize=(12, 5))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Predict the probabilities for the test set
y_pred_prob = model.predict(X_test)

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

target_names = ['Postive','Neutral','Negative']
print(classification_report(y_test, y_pred, target_names=target_names))