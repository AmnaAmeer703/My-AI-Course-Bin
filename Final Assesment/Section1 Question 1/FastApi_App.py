from fastapi import FastAPI
import uvicorn
import pickle
import pandas as pd

sex = ['female','maale']
smoker = ['yes','no']
region = ['southeast','southwest','northwest','northeast']

app = FastAPI(debug=True)

@app.get('/')
def home():
    return  {'text': 'Insurrance Predictor'}

@app.get('/predict')
def predict(age: int, sex: str, bmi: float, children : int,
            smoker: str, region: str):
      model = pickle.load(open('pipe4.pkl','rb'))
      prediction = pd.DataFrame({'age':[age],'sex':[sex],'bmi':[bmi],'children':[children],'smoker':[smoker],'region':[region]})

      result = model.predict(prediction)
      output = round(result[0],2)
      return {'Charges are {}'.format(output)}

if __name__ == '__main__':
    uvicorn.run(app)