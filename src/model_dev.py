import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()



class Model(ABC):
    
    """
    Abstract class for all models
    """
    def train(self, X_train, y_train):
        
        """Traing a model
        
        Args: 
            X_train : training data
            y_train : testing data
        
        """
        pass
    
    
class LogisticRegressionModel(Model):
    """Linear regression model
    """
    def train(self, X_train, y_train, **kwargs):
        """Trains the model

        Args:
            X_train (dataframe): Training set
            y_train (dataseries): Test set
        """
        
        try:
            reg = LogisticRegression()
            reg.fit(X_train, y_train)
            logging.info("Model training completed successfully")
            return reg
        except Exception as e:
            logging.error("Error in training the model: {}".format(e))
            raise e
        
        
        
