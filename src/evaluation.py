import logging
from abc import ABC, abstractmethod
import numpy as np

class Evaluation(ABC):
    """Abstract class defining strategy for evaluation our models

    Args:
        ABC (_type_): _description_
    """
    @abstractmethod
    def calculate_scores(self, y_true:np.array, y_pred:np.array):
        """
        Calculates the scores for the model

        Args:
            y_true (np.array): True labels
            y_pred (np.array): Predicted labels
        
        Returns:
            None
        """
        


class MSE(Evaluation):
    """Evaluation Strategy that uses Mean Squared Error

    Args:
        Evaluation (_type_): _description_
    """