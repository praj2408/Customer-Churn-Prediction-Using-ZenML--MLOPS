import logging
import pandas as pd

from zenml import step
from src.model_dev import LogisticRegressionModel
from sklearn.base import ClassifierMixin
from config import ModelNameConfig

@step
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series) -> ClassifierMixin:
    
    """
    Trains the model on the ingested data
    
    Args:
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series

    """
    model = None
    if config.model  


