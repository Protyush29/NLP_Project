import gensim
import numpy as np
from shared.data_extractor import DataExtractor

class Doc2Vec:
    def __init__(self):
        self.train_data = None
        self.test

    def data_collection(self):
        """
        uses data extracted from the data extractor class
            return : list of pandas dataframe
        """
        data =