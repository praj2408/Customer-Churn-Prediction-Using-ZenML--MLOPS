import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Union

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()



class DataStrategy(ABC):
    """
    Abstract class defining strategy for handling data

    Args:
        ABC (_type_): _description_
    """
    
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass
    
    
    
class DataPreProcessStrategy(DataStrategy):
    """
    Abstract class defining strategy for handling data preprocessing
    """
    try:
        def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
            """
            Preprocessing the data
            """
            ## Drop these columns
            data = data.drop("customerID", axis=1, inplace=True)
            
            ## Preprocessing the data
            data.gender = pd.get_dummies(data.gender, drop_first=True, dtype="int64")
            data.Partner = pd.get_dummies(data.Partner, drop_first=True, dtype="int64")
            data.Dependents = pd.get_dummies(data.Dependents, drop_first=True, dtype="int64")
            data.PhoneService = pd.get_dummies(data.PhoneService, drop_first=True, dtype="int64")
            data.MultipleLines = le.fit_transform(data.MultipleLines)
            data.InternetService = le.fit_transform(data.InternetService)
            data.OnlineSecurity = le.fit_transform(data.OnlineSecurity)
            data.OnlineBackup = le.fit_transform(data.OnlineBackup)
            data.DeviceProtection = le.fit_transform(data.DeviceProtection)
            data.PaperlessBilling = pd.get_dummies(data.PaperlessBilling, drop_first=True)
            data.TechSupport = le.fit_transform(data.TechSupport)
            data.StreamingTV = le.fit_transform(data.StreamingTV)
            data.StreamingMovies = le.fit_transform(data.StreamingMovies)
            data.Contract = le.fit_transform(data.Contract)
            data.PaymentMethod = le.fit_transform(data.PaymentMethod)
            data.Churn = le.fit_transform(data.Churn)
            print(data)
            return data
    except Exception as e: 
        raise e
    

class DataDivideStrategy(DataStrategy):
    """Split data into training and testing data

    Args:
        DataStrategy (_type_): _description_
    """
    
    def handle_data(self, data: pd.DataFrame) ->Union[pd.DataFrame ,pd.Series] :
        
        """
        Train test Split
        """
        try:
            X = data.drop("Churn", axis=1)
            y = data["Churn"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            raise e
        
        
class DataCleaning():
    """Class for cleaning data which processes the data and divides it into train and test
    """
    
    def __init__(self, data: pd.DataFrame, strategy: DataStrategy) -> None:
        self.data = data
        self.strategy = strategy
        
    def handle_data(self)-> Union[pd.DataFrame, pd.Series]:
        """
        Handle the data
        Returns:
            Union[pd.DataFrame, pd.Series]: _description_
        """
        
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            raise e