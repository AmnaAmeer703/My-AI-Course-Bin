import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import re
import nltk
nltk.download('all')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
lemm = WordNetLemmatizer()
import mlflow
import mlflow.sklearn
import logging
import os
import time

mlflow.set_experiment("Experimental Tracking QuickStart")

resume = pd.read_csv('UpdatedResumeDataSet.csv',encoding='utf-8')

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

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
resume['Labels'] = le.fit_transform(resume['Category'])

label_map = dict(zip(le.classes_, le.transform(le.classes_)))

print(label_map)
print(resume['Labels'].value_counts())
print(resume['Labels'].unique())

X = resume['Resume']
y = resume['Labels'].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(stop_words='english',ngram_range=(1,2),max_df=0.7,max_features=50_000)
X_train_CV = cv.fit_transform(X_train).toarray()
X_test_CV = cv.transform(X_test).toarray()

from sklearn.naive_bayes import MultinomialNB

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

logging.info("Starting MLflow run...")

with mlflow.start_run():
    start_time = time.time()
    
    try:
        logging.info("Logging preprocessing parameters...")
        mlflow.log_param("vectorizer", "Bag of Words")
        mlflow.log_param("num_features", 100)
        mlflow.log_param("test_size", 0.25)

        logging.info("Initializing Logistic Regression model...")
        M = MultinomialNB(alpha=0.01) # Increase max_iter to prevent non-convergence issues

        logging.info("Fitting the model...")
        M.fit(X_train_CV, y_train)
        logging.info("Model training complete.")

        logging.info("Logging model parameters...")
        mlflow.log_param("model", "MultiNomialNB")

        logging.info("Making predictions...")
        y_pred = M.predict(X_test)

        logging.info("Calculating evaluation metrics...")
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        logging.info("Logging evaluation metrics...")
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        logging.info("Saving and logging the model...")
        mlflow.sklearn.log_model(M, "model")

        # Log execution time
        end_time = time.time()
        logging.info(f"Model training and logging completed in {end_time - start_time:.2f} seconds.")

        # Save and log the notebook
        # notebook_path = "exp1_baseline_model.ipynb"
        # logging.info("Executing Jupyter Notebook. This may take a while...")
        # os.system(f"jupyter nbconvert --to notebook --execute --inplace {notebook_path}")
        # mlflow.log_artifact(notebook_path)

        # logging.info("Notebook execution and logging complete.")

        # Print the results for verification
        logging.info(f"Accuracy: {accuracy}")
        logging.info(f"Precision: {precision}")
        logging.info(f"Recall: {recall}")
        logging.info(f"F1 Score: {f1}")

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)




