import sys
from src.entity.config_entity import UberLyftPredictorConfig
from src.entity.s3_estimator import Proj3Estimator
from src.exception import MyException
from src.logger import logging
from pandas import DataFrame


class UberLyftData:
    def __init__(self,
                distance,
                cab_type,
                destination,
                source,
                serge_multiplier,
                name,
                Perid_Of_Time
                ):
       
        try:
            self.distance = distance
            self.cab_type = cab_type
            self.destination = destination
            self.source = source
            self.serge_multiplier = serge_multiplier
            self.name = name
            self.Perid_Of_Time = Perid_Of_Time
    
        except Exception as e:
            raise MyException(e, sys) from e

    def get_uberlyft_input_data_frame(self)-> DataFrame:
        
        try:
            
            uberlyft_input_dict = self.get_uberlyft_data_as_dict()
            return DataFrame(uberlyft_input_dict)
        
        except Exception as e:
            raise MyException(e, sys) from e


    def get_uberlyft_data_as_dict(self):
        
        logging.info("Entered get_uberlyft_data_as_dict method as UberLyftData class")

        try:
            input_data = {
                "distance": [self.distance],
                "cab_type": [self.cab_type],
                "destination": [self.destination],
                "source": [self.source],
                "serge_multiplier": [self.serge_multiplier],
                "name": [self.name],
                "Period_Of_Time": [self.Perid_Of_Time]
            }

            logging.info("Created uberlyft data dict")
            logging.info("Exited get_uberlyft_data_as_dict method as UberLyftData class")
            return input_data

        except Exception as e:
            raise MyException(e, sys) from e

class UberLyftDataRegressor:
    def __init__(self,prediction_pipeline_config: UberLyftPredictorConfig = UberLyftPredictorConfig(),) -> None:
        
        try:
            self.prediction_pipeline_config = prediction_pipeline_config
        except Exception as e:
            raise MyException(e, sys)

    def predict(self, dataframe) -> str:
        
        try:
            logging.info("Entered predict method of UberLyftDataRegressor class")
            model = Proj3Estimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path,
            )
            result =  model.predict(dataframe)
            
            return result
        
        except Exception as e:
            raise MyException(e, sys)