import sys
from src.entity.config_entity import AppleRetailSalesPredictorConfig
from src.entity.s3_estimator import Proj1Estimator
from src.exception import MyException
from src.logger import logging
from pandas import DataFrame


class AppleData:
    def __init__(self,
                Product_Name,
                quantity,
                Store_Name,
                City,
                category_name,
                sale_year,
                sale_month,
                Launch_Year,
                Launch_Month
                ):
    
        try:
            self.Product_Name = Product_Name
            self.quantity = quantity
            self.Store_Name = Store_Name
            self.City = City
            self.category_name = category_name
            self.sale_year = sale_year
            self.sale_month = sale_month
            self.Launch_Year = Launch_Year
            self.Launch_Month = Launch_Month

        except Exception as e:
            raise MyException(e, sys) from e

    def get_apple_input_data_frame(self)-> DataFrame:
        
        try:
            
            apple_input_dict = self.get_vehicle_data_as_dict()
            return DataFrame(apple_input_dict)
        
        except Exception as e:
            raise MyException(e, sys) from e


    def get_apple_data_as_dict(self):
        
        logging.info("Entered get_apple_data_as_dict method as AppleData class")

        try:
            input_data = {
                "Product_Name": [self.Product_Name],
                "quantity": [self.quantity],
                "Store_Name": [self.Store_Name],
                "City": [self.City],
                "category_name": [self.category_name],
                "sale_year": [self.sale_year],
                "sale_month": [self.sale_month],
                "Launch_Year": [self.Launch_Year],
                "Launch_Month": [self.Launch_Month]
            }

            logging.info("Created apple data dict")
            logging.info("Exited get_apple_data_as_dict method as AppleData class")
            return input_data

        except Exception as e:
            raise MyException(e, sys) from e

class AppleRetailSalesDataRegressor:
    def __init__(self,prediction_pipeline_config: AppleRetailSalesPredictorConfig = AppleRetailSalesPredictorConfig(),) -> None:
        
        try:
            self.prediction_pipeline_config = prediction_pipeline_config
        except Exception as e:
            raise MyException(e, sys)

    def predict(self, dataframe) -> str:

        try:
            logging.info("Entered predict method of AppleRetailSalesDataRegressor class")
            model = Proj1Estimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path,
            )
            result =  model.predict(dataframe)
            
            return result
        
        except Exception as e:
            raise MyException(e, sys)