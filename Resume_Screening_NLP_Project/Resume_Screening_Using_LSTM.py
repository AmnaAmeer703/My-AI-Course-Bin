import warnings
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from keras.utils import pad_sequences, to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Activation, Embedding, Input, Bidirectional
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
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


X = resume['Resume']
y = resume['Category']

vocab_size = 10000
max_len = 40
num_classes = 25

tokenizer = keras.preprocessing.text.Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)

print("Word Index:", tokenizer.word_index)
print("Sequences:", sequences)

X = pad_sequences(sequences, maxlen=max_len)
print("Padded Sequences:", X)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
resume['Category'] = le.fit_transform(resume['Category'])
y = resume['Category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print("X_train:", X_train)
print("y_train:", y_train)
print("X_test:", X_test)
print("y_test:", y_test)

model = Sequential([
    Input(shape=(10000,)),
    Embedding(input_dim=vocab_size, output_dim=128),
    Bidirectional(LSTM(64)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')  # softmax for multi-class single-label
])



METRICS = metrics=['accuracy', 
                   Precision(name='precision'),
                   Recall(name='recall')]

model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=METRICS)
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=5, batch_size=1,verbose=1, validation_data=(X_test, y_test))

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


target_names = ['Java Developer','Testing','DevOps Engineer ','Python Developer ','Web Designing','HR','Hadoop','Blockchain','ETL Developer','Operations Manager','Data Science','Sales','Mechanical Engineer','Arts','Database','Electrical Engineering','Health and fitness','PMO','Business Analyst','DotNet Developer','Automation Testing','Network Security Engineer','SAP Developer','Civil Engineer','Advocate']
print(classification_report(y_test, y_pred, target_names=target_names))

