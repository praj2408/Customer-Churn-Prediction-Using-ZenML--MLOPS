import logging
from abc import ABC, abstractmethod

class Evaluation(ABC):
    """Abstract class defining strategy for evaluation our models

    Args:
        ABC (_type_): _description_
    """
    @abstractmethod
    def c