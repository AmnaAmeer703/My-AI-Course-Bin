import warnings
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Activation
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from keras.metrics import Precision, Recall

import re
import nltk
nltk.download('all')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
lemm = WordNetLemmatizer()

resume = pd.read_csv('UpdatedResumeDataSet.csv',encoding='utf-8')
print(resume)

print(resume.dtypes)
print(resume.info())
print(resume.shape)
print(resume.isnull()*100)

print(resume['Category'].value_counts())
print(resume['Category'].unique())

plt.figure(figsize=(20,5))
plt.xticks(rotation=90)
ax=sns.countplot(x="Category", data=resume)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))
plt.grid()

features = ['Resume']
X = resume[features]
y = resume['Category']

def toLower(resume):
    if isinstance(resume, float):
        return '<UNK>'
    else:
        return resume.lower()

stop_words = stopwords.words("english")

def remove_stopwords(text):
    no_stop = []
    for word in text.split(' '):
        if word not in stop_words:
            no_stop.append(word)
    return " ".join(no_stop)

def remove_punctuation_func(text):
    return re.sub(r'[^a-zA-Z0-9]', ' ', text)

X['Resume'] = X['Resume'].apply(toLower)

X['Resume'] = X['Resume'].apply(remove_stopwords)

X['Resume'] = X['Resume'].apply(lambda x: lemm.lemmatize(x))

X['Resume'] = X['Resume'].apply(remove_punctuation_func)

X['Resume'] = list(X['Resume'])


resume[resume['Category'] == 'Data Science' ]


from wordcloud import WordCloud
wc = WordCloud(width=500,height=500,min_font_size=10,background_color='black')
spam_wc = wc.generate(resume[resume['Category']=='Data Science']['Resume'].str.cat(sep=""))
plt.imshow(spam_wc)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
resume['Category'] = le.fit_transform(resume['Category'])

X = resume['Resume'].values
y = resume['Category'].values


word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    stop_words='english',
    max_features=1500)
word_vectorizer.fit(X)
WordFeatures = word_vectorizer.transform(X)

X_train,X_test,y_train,y_test = train_test_split(WordFeatures,y,random_state=0, test_size=0.2)
print(X_train.shape)
print(X_test.shape)

# Convert the sparse tensor to a dense tensor and reshape
X_train_dense = X_train.toarray()
X_train_reshaped = np.reshape(X_train_dense, (X_train_dense.shape[0], 1, X_train_dense.shape[1]))

X_test_dense = X_test.toarray()
X_test_reshaped = np.reshape(X_test_dense, (X_test_dense.shape[0], 1, X_test_dense.shape[1]))

n_hidden = 64
n_classes = 25


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


