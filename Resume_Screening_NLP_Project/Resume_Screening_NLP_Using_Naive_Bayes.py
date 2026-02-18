import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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
spam_wc = wc.generate(resume[resume['Category']=='Data Science']['cleaned_resume'].str.cat(sep=""))
plt.imshow(spam_wc)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
resume['Labels'] = le.fit_transform(resume['Category'])

label_map = dict(zip(le.classes_, le.transform(le.classes_)))

print(label_map)
print(resume['Labels'].value_counts())
print(resume['Labels'].unique())

X = resume['cleaned_resume']
y = resume['Labels'].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(stop_words='english',ngram_range=(1,2),max_df=0.7,max_features=50_000)
X_train_CV = cv.fit_transform(X_train).toarray()
X_test_CV = cv.transform(X_test).toarray()

from sklearn.naive_bayes import MultinomialNB
M = MultinomialNB(alpha=0.01)
M.fit(X_train_CV,y_train)

pred = M.predict(X_test_CV)
print(pred)
from sklearn import metrics
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

accuracy_score(y_test,pred)
print(accuracy_score(y_test,pred))

cnf_matrix = metrics.confusion_matrix(y_test, pred)
print(" Model Evaluation using Confusion Matrix : " , cnf_matrix)

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()
plt.Text(0.5,257.44,'Predicted label');

from sklearn.metrics import classification_report
target_names = ['Java Developer','Testing','DevOps Engineer ','Python Developer ','Web Designing','HR','Hadoop','Blockchain','ETL Developer','Operations Manager','Data Science','Sales','Mechanical Engineer','Arts','Database','Electrical Engineering','Health and fitness','PMO','Business Analyst','DotNet Developer','Automation Testing','Network Security Engineer','SAP Developer','Civil Engineer','Advocate']
print(classification_report(y_test, pred, target_names=target_names))


import pickle
pickle.dump(M,open('M.pkl','wb'))
pickle.dump(cv,open('cv.pkl','wb'))
pickle.dump(resume,open('resume.pkl','wb'))
            
import IPython.display
from IPython.display import FileLink
FileLink('cv.pkl')
FileLink('M.pkl')
FileLink('resume.pkl')
