import pandas as pd
import pymongo

product = pd.read_csv("Apple_Retail_Dataset/products.csv")
sales = pd.read_csv("Apple_Retail_Dataset/sales.csv")
warranty = pd.read_csv("Apple_Retail_Dataset/warranty.csv")
stores = pd.read_csv("Apple_Retail_Dataset/stores.csv")
category = pd.read_csv("Apple_Retail_Dataset/category.csv")

product = product.rename(columns={'Product_ID':'product_id'})

stores = stores.rename(columns={'Store_ID':'store_id'})

category = category.rename(columns={'category_id':'Category_ID'})

# Merge product and sales csv file
product_sales = pd.merge(product,sales, on = 'product_id')

# Merge Product_sales and stores CSV file
product_sales_stores = pd.merge(product_sales,stores, on = 'store_id')

# Merge Product_sales_stores and category Csv File
product_sales_stores_category = pd.merge(product_sales_stores,category, on = 'Category_ID')

df = product_sales_stores_category

# We can create a new column Total_Sales by multiply 2 columns 'quantity' and 'prices'
df['Total_Sales'] = df['Price'] * df['quantity']


data = df.to_dict(orient='records')

DB_NAME = "Proj1"
COLLECTION_NAME = "Proj1-Data"
CONNECTION_URL = "mongodb+srv://########:#######y@cluster0.sf954yq.mongodb.net/?appName=Cluster0"

# above, either remove your credentials or delete the mongoDB resource bofore pushing it to github.

client = pymongo.MongoClient(CONNECTION_URL)
data_base = client[DB_NAME]
collection = data_base[COLLECTION_NAME]

rec = collection.insert_many(data)

df = pd.DataFrame(list(collection.find()))
df.head(2)