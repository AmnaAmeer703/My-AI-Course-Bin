from fastapi import FastAPI
import uvicorn
import pickle
import pandas as pd

cab_type = ['Uber','Lyft']
destination = ['Financial District','Black Bay','Theatre District','Haymarket Square','Boston University','Fenway','Northeastern University','North End','South Station','North Station','West End','Beacon Hill']
source = ['Financial District','Black Bay','Theatre District','Haymarket Square','Boston University','Fenway','Northeastern University','North End','South Station','North Station','West End','Beacon Hill']
name = ['Shared','Lux','Lyft','Lux Black XL','Lyft XL','Lux Black','Uber XL','Black','Uber X','WAV','Black SUV','UberPool']
Period_Of_Time = ['Early Morning','Morning','Noon','Evening','Night','Late Night']

app = FastAPI(debug=True)

@app.get('/')
def home():
    return  {'text':'Cab Fare Predictor'}

@app.get('/predict')
def predict(distance: float, cab_type: str, destination: str, source: str, surge_multiplier: float, name: str, Period_Of_Time: str):\
    
    model=pickle.load(open('UberLyftPipeline.pkl','rb'))
    Cab_Fares = pd.DataFrame({'distance':[distance],'cab_type':[cab_type],'destination':[destination],'source':[source],'surge_multiplier':[surge_multiplier],'name':[name],'Period_Of_Time':[Period_Of_Time]})
    result = model.predict(Cab_Fares)
    output = round(result[0],2)

    return {'Fare is {}'.format(output)}


if __name__ == '__main__':
    uvicorn.run(app)