import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join("artifacts","train.csv")
    test_data_path:str = os.path.join("artifacts","test.csv")
    raw_data_path:str = os.path.join("artifacts","data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion method: initiate_data_ingestion start")
        try:
            df=pd.read_csv(r'C:\Users\naray\Documents\python_projects\MLProjects\Student_Performance_ML_Project\src\notebook\data\stud.csv')
            # df=pd.read_csv('src\notebook\data\stud.csv')
            logging.info("Data Ingestion method: Read the data into Dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path))
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Data Ingestion method: Train Test Split Initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Data Ingestion method: Data Ingestion completed")

            return (self.ingestion_config.train_data_path,self.ingestion_config.test_data_path)
        except Exception as e:
            logging.info("Data Ingestion method: Error =>",e)
            raise CustomException(e,sys)


if __name__=="__main__":
    obj=DataIngestion()
    obj.initiate_data_ingestion()