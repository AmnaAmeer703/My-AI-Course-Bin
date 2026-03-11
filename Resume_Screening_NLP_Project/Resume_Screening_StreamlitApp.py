import nltk 
import streamlit as st 
import pickle

resume = pickle.load(open('resume.pkl','rb'))
M = pickle.load(open('M.pkl','rb'))
cv = pickle.load(open('cv.pkl', 'rb'))

import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords.words('english')

from nltk.stem.porter import PorterStemmer
import string
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

resume['cleaned_resume'] = resume['Resume'].apply(transform_text)


#webapp
def main():
        st.title('Resume Screening  Classify App')
        upload_file = st.file_uploader('Upload Your Resume', type=['txt','pdf'])

         
        
        if upload_file is not None:
            try:
                   resume_bytes = upload_file.read()
                   resume_text = resume_bytes.decode('utf-8')
            except UnicodeDecodeError:
                   #if Unicode is fail try different code 
                   resume_text = resume_bytes.decode('latin-1')
            
            

            
            cleaned_resume = transform_text(resume_text)
            cleaned_resume = cv.transform([cleaned_resume])
            prediction_id = M.predict(cleaned_resume)[0]
            
            filtered_df = resume[resume['Labels'] == prediction_id]
            category_value = resume['Category'].iloc[0]
            print("Given Resume is for :", category_value)
            st.write('Given Resume is for :-', category_value)


            



                



#python Main 
        
if __name__=="__main__":
        main()