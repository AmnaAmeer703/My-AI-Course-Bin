import sys
from src.entity.config_entity import UberLyftPredictorConfig
from src.entity.s3_estimator import Proj1Estimator
from src.exception import MyException
from src.logger import logging
from pandas import DataFrame


class UberLyftData:
    def __init__(self,
                distance,
                cab_type,
                destination,
                source,
                surge_multiplier,
                name,
                Period_Of_Time
                ):
        """
        Uber&Lyft Data constructor
        Input: all features of the trained model for prediction
        """
        try:
            self.distance = distance
            self.cab_type = cab_type
            self.destination = destination
            self.source = source
            self.surge_multiplier = surge_multiplier
            self.name = name
            self.Period_Of_Time = Period_Of_Time


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
                "surge_multiplier": [self.surge_multiplier],
                "name": [self.name],
                "Period_Of_Time": [self.Period_Of_Time]
            }

            logging.info("Created uberlyft data dict")
            logging.info("Exited get_uberlyft_data_as_dict method as UberLyftData class")
            return input_data

        except Exception as e:
            raise MyException(e, sys) from e

class UberLyftDataRegressor:
    def __init__(self,prediction_pipeline_config: UberLyftPredictorConfig = UberLyftPredictorConfig(),) -> None:
        """
        :param prediction_pipeline_config: Configuration for prediction the value
        """
        try:
            self.prediction_pipeline_config = prediction_pipeline_config
        except Exception as e:
            raise MyException(e, sys)

    def predict(self, dataframe) -> str:
        try:
            logging.info("Entered predict method of VehicleDataClassifier class")
            model = Proj1Estimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path,
            )
            result =  model.predict(dataframe)
            
            return result
        
        except Exception as e:
            raise MyException(e, sys)