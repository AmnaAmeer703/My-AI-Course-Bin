from fastapi import FastAPI
import uvicorn
import pickle
import pandas as pd

Product_Name = ['HomePod mini','iMac with Retina Display','iPhone 13 Pro Max','iPad Pro 12.9-inch','Magic Trackpad','Leather Case for iPhone','HomePod','AirPods (2nd Generation)','iCloud','Smart Cover for iPad','Apple Watch Series 9','Apple Watch Series 7','Lightning to USB Cable','Mac Mini','iPad mini (6th Generation)','iMac 27-inch','iPad Pro (M1)','Silicone Case for iPhone','Apple Watch Ultra','iPad (10th Generation)','AirPods Pro','iPhone 12 Pro Max','HomePod (2nd Generation)','Apple Watch Hermès','Apple Watch Series 5','iPhone 14 Plus','Apple Watch Nike Edition','AirPods Pro (2nd Generation)',
'AirPods Max','iPhone 12 mini','iPad mini (5th Generation)','MacBook (Retina)','AirTag','MacBook Pro 16-inch','iPhone 13','Apple Watch Series 6','iMac 24-inch','HomePod mini','iMac with Retina Display','iPhone 13 Pro Max','iPad Pro 12.9-inch','Magic Trackpad','Leather Case for iPhone','HomePod','AirPods (2nd Generation)','iCloud','Smart Cover for iPad','Apple Watch Series 9','Apple Watch Series 7','Lightning to USB Cable','Mac Mini','iPad mini (6th Generation)','iMac 27-inch','iPad Pro (M1)','Silicone Case for iPhone','Apple Watch Ultra','iPad (10th Generation)','AirPods Pro','iPhone 12 Pro Max','HomePod (2nd Generation)','Apple Watch Hermès','Apple Watch Series 5','iPhone 14 Plus','Apple Watch Nike Edition','AirPods Pro (2nd Generation)',
'AirPods Max','iPhone 12 mini','iPad mini (5th Generation)','MacBook (Retina)','AirTag','MacBook Pro 16-inch','iPhone 13','Apple Watch Series 6','iMac 24-inch','Apple TV (3rd Generation)','Apple News+','Beats Fit Pro','iPad Pro 11-inch','MagSafe Charger','Apple Music','Magic Mouse',
'Mac Pro (Rack)','MagSafe Battery Pack','MacBook','iPhone 13 mini','MacBook Air (Retina)','iPhone 14','Apple News+','Beats Fit Pro','iPad Pro 11-inch','MagSafe Charger','Apple Music','Magic Mouse','Mac Pro (Rack)','MagSafe Battery Pack','MacBook','iPhone 13 mini','MacBook Air (Retina)','iPhone 14']
Store_Name = ['Apple Chadstone','Apple Covent Garden','Apple The Dubai Mall','Apple Central World','Apple Orchard Road','Apple Champs-Elysees','Apple Dubai Mall','Apple Fukuoka','Apple SoHo','Apple Fifth Avenue','Apple Southland','Apple Schildergasse','Apple Kurfuerstendamm','Apple Via del Corso','Apple North Michigan Avenue','Apple Kyoto','Apple Leidseplein','Apple Jewel Changi Airport','Apple Highpoint','Apple Beijing SKP','Apple Kumamoto','Apple Taipei 101','Apple Sanlitun','Apple Gangnam','Apple Iconsiam','Apple Sainte-Catherine','Apple Yeouido','Apple Nanjing East','Apple Brompton Road','Apple Eaton Centre','Apple Yorkdale','Apple Omotesando','Apple Santa Fe','Apple Opera','Apple Grand Central','Apple Marunouchi','Apple Union Square','Apple The Grove','Apple Cotai Central','Apple Rosenstrasse','Apple Ala Moana','Apple Pioneer Place','Apple Rideau Centre','Apple Bondi','Apple Shanghai IFC','Apple The Americana at Brand','Apple Brisbane','Apple Parque La Colina','Apple Shinjuku','Apple Galeries Lafayette','Apple Michigan Avenue','Apple Taikoo Li','Apple Via Santa Fe','Apple Metrotown','Apple Regent Street','Apple Park Visitor Center','Apple Yas Mall','Apple Kaerntner Strasse','Apple Mall of the Emirates','Apple Downtown Brooklyn','Apple South Coast Plaza','Apple Antara','Apple Passeig de Gracia','Apple Andino','Apple Walnut Street','Apple Sydney','Apple Piazza Liberty','Apple Beverly Center','Apple Causeway Bay']
City = ['Dubai','London','Paris','New York','Melbourne','Tokyo','Bangkok','Singapore','Mexico City','Beijing','Seoul','Toronto','Chicago','Shanghai','Bogota','Los Angeles','Fukuoka','Cheltenham','Cologne','Berlin','Rome','Kyoto','Amsterdam','Kumamoto','Taipei','Montreal','San Francisco','Macau','Munich','Honolulu','Portland','Ottawa','Bondi','Glendale','Brisbane','Chengdu','Burnaby','Cupertino','Abu Dhabi','Vienna','Brooklyn','Costa Mesa','Barcelona','Philadelphia','Sydney','Milan','Hong Kong']
Country = ['United','States','Australia','China','Japan','Canada','UAE','United Kingdom','France','Germany','Thailand','Singapore','Mexico','South Korea','Italy','Colombia','Netherlands','Taiwan','Austria','Spain']
category_name = ['Accessories','Smartphone','Audio','Tablet','Desktop','Laptop','Wearable','Subscription Service','Streaming Device','Smart Speaker']

app = FastAPI(debug=True)

@app.get('/')
def home():
    return  {'text':'Apple Retail Sale Predictor'}

@app.get('/predict')
def predict(Product_Name: str, Price: int, quantity: int, Store_Name: str, City: str, Country: str, category_name: str,
             sale_year: int, sale_month: int, Launch_Year: int, Launch_Month: int):
    model = pickle.load(open('pipe3.pkl','rb'))
    
    Apple_Sales = pd.DataFrame({'Product_Name':[Product_Name],'Price':[Price],'quantity':[quantity],'Store_Name':[Store_Name],'City':[City],'Country':[Country],'category_name':[category_name],'sale_year':[sale_year],'sale_month':[sale_month],'Launch_Year':[Launch_Year],'Launch_Month':[Launch_Month]})

    result = model.predict(Apple_Sales)
    output = round(result[0],2)
    return {'Total_Sale is {}'.format(output)}

if __name__ == '__main__':
    uvicorn.run(app)